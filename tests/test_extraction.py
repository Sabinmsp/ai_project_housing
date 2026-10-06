import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import openai
import pytest

from triage import extraction
from triage.extraction import (
    FAULT_NAMES,
    PROMPT_EXAMPLES,
    SYSTEM_PROMPT,
    OfflineExtractor,
    _FAULT_PATTERNS,
    build_user_prompt,
    extract,
    response_schema,
)
from triage.intake import create_report
from triage.models import ExtractedFacts, ExtractionStatus, ReportExtraction, SourceTag
from triage.report_files import parse_report_text, read_text
from triage.tiers import TIER_TABLE

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)

# Words the model must never see (invariants 1 and 3). Schema: substring; prompt: whole word.
BANNED_WORDS = (
    "dangerous", "standard", "tier", "points", "score", "rank", "priority", "severity", "urgen",
    "language", "dialect", "english", "tone",
    "severe", "serious", "urgent", "urgency", "emergency",
    "repaired first", "nt.gov.au", "s63", "Residential Tenancies",
)


def report(text):
    return create_report(tenant_id="T", raw_text=text, source_tag="tenant_direct",
                         community="Maningrida", original_report_timestamp=T0)


class ScriptedClient:
    """Returns canned responses in order, counting calls."""

    name = "scripted"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.seen_prompts = []

    def complete_json(self, system, user, schema):
        self.calls += 1
        self.seen_prompts.append(system + user)
        return self.responses.pop(0)


GOOD = json.dumps({
    "fault_description": "toilet blocked",
    "taxonomy_match": ["blocked or broken toilet"],
    "alternative_mentioned": False,
    "coping_mentioned": False,
    "impact_status": "ongoing",
    "hazard_status": "none",
    "mechanism_type": None,
    "harm_claimed": False,
    "fault_or_sign": "fault",
    "claim_mismatch": None,
    "worsening_mentioned": False,
    "quoted_spans": [{"field": "fault_description", "text": "toilet blocked"},
                     {"field": "taxonomy_match", "text": "toilet blocked"}],
})


def response(*faults: dict) -> str:
    """A model response: the ReportExtraction wrapper around zero or more fault records."""
    return json.dumps({"faults": list(faults)})


GOOD_RESPONSE = response(json.loads(GOOD))


# --- the model can never see scoring information -------------------------

def test_prompt_contains_no_tier_or_scoring_information():
    prompt = (SYSTEM_PROMPT + build_user_prompt("toilet blocked")).lower()
    for word in ("dangerous", "standard", "tier", "points", "score", "rank", "priority"):
        assert word not in prompt, word
    assert not any(ch.isdigit() for ch in build_user_prompt("toilet blocked"))


def test_response_schema_has_no_numeric_fields():
    for schema in (ExtractedFacts.model_json_schema(), response_schema()):
        schema_text = json.dumps(schema)
        assert '"integer"' not in schema_text
        assert '"number"' not in schema_text


def test_schema_sent_to_model_has_no_scoring_language():
    """Panel A covers everything the model sees, the schema included."""
    text = json.dumps(response_schema()).lower()
    for word in BANNED_WORDS:
        assert word.lower() not in text, word


def test_response_schema_is_strict():
    """Strict structured output: every object lists all its properties as
    required and allows no others, so the provider enforces the shape."""
    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    found = list(objects(response_schema()))
    assert len(found) == 3  # ReportExtraction, ExtractedFacts and QuotedSpan
    for obj in found:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


# --- validation boundary --------------------------------------------------

def test_good_response_first_try():
    c = ScriptedClient(GOOD_RESPONSE)
    res = extract(report("toilet blocked"), c)
    assert res.status is ExtractionStatus.OK and res.attempts == 1 and c.calls == 1


def test_malformed_then_good_retries_once():
    c = ScriptedClient("not json", GOOD_RESPONSE)
    res = extract(report("toilet blocked"), c)
    assert res.status is ExtractionStatus.OK and res.attempts == 2 and len(res.errors) == 1


def test_two_failures_flag_for_human():
    c = ScriptedClient("{}garbage", '{"fault_description": 5}')
    res = extract(report("toilet blocked"), c)
    assert res.status is ExtractionStatus.FLAGGED_FOR_HUMAN
    assert res.extraction is None and c.calls == 2


def test_unknown_fault_name_is_a_validation_failure():
    bad = json.loads(GOOD)
    bad["taxonomy_match"] = ["toilet emergency (severe)"]
    c = ScriptedClient(response(bad), response(bad))
    assert extract(report("toilet blocked"), c).status is ExtractionStatus.FLAGGED_FOR_HUMAN


def test_claim_without_span_rejected():
    bad = json.loads(GOOD)
    bad["coping_mentioned"] = True
    with pytest.raises(ValueError):
        ExtractedFacts.model_validate(bad)


def test_hazard_needs_mechanism_type():
    bad = json.loads(GOOD)
    bad["hazard_status"] = "described"
    bad["quoted_spans"].append({"field": "hazard", "text": "water in light"})
    with pytest.raises(ValueError):
        ExtractedFacts.model_validate(bad)


@pytest.mark.parametrize("field, value", [
    ("severity", "high"), ("priority", "urgent"), ("rank", 1), ("tier", "dangerous"),
    ("urgency", "high"), ("urgency_tally", 4), ("points", 3), ("score", 0.9),
    ("safety_flag", True),
])
def test_llm_cannot_add_scoring_fields(field, value):
    """The model has no way to hand a score, tier or rank to later stages."""
    bad = json.loads(GOOD)
    bad[field] = value
    with pytest.raises(ValueError):
        ExtractedFacts.model_validate(bad)
    c = ScriptedClient(response(bad), response(bad))
    assert extract(report("toilet blocked"), c).status is ExtractionStatus.FLAGGED_FOR_HUMAN


def test_taxonomy_match_without_span_rejected():
    bad = json.loads(GOOD)
    bad["quoted_spans"] = [s for s in bad["quoted_spans"] if s["field"] != "taxonomy_match"]
    with pytest.raises(ValueError, match="taxonomy_match"):
        ExtractedFacts.model_validate(bad)


def test_intermittent_without_span_rejected():
    bad = json.loads(GOOD)
    bad["impact_status"] = "intermittent"
    with pytest.raises(ValueError, match="impact_status"):
        ExtractedFacts.model_validate(bad)
    bad["quoted_spans"].append({"field": "impact_status", "text": "on and off"})
    assert ExtractedFacts.model_validate(bad).impact_status == "intermittent"


# --- 2026-10-04 schema (master §3.2.2) ---------------------------------------

NEW_FIELDS = ["hazard_status", "harm_claimed", "fault_or_sign", "claim_mismatch", "worsening_mentioned"]


def good_with(spans=(), **fields):
    data = json.loads(GOOD)
    data.update(fields)
    data["quoted_spans"] += [{"field": f, "text": "quoted"} for f in spans]
    return data


@pytest.mark.parametrize("field", NEW_FIELDS)
def test_new_field_missing_rejected(field):
    data = json.loads(GOOD)
    del data[field]
    with pytest.raises(ValueError, match=field):
        ExtractedFacts.model_validate(data)


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="hazard_mechanism"):
        ExtractedFacts.model_validate(good_with(hazard_mechanism="water in light"))


# Each case is valid once its span is added, so the rejection is the missing span alone.
@pytest.mark.parametrize("fields, span", [
    ({"hazard_status": "described", "mechanism_type": "active"}, "hazard"),
    ({"hazard_status": "unclear"}, "hazard"),
    ({"harm_claimed": True}, "harm_claimed"),
    ({"fault_or_sign": "sign"}, "fault_or_sign"),
    ({"worsening_mentioned": True}, "worsening_mentioned"),
])
def test_claim_without_span_rejected_new_fields(fields, span):
    with pytest.raises(ValueError, match=span):
        ExtractedFacts.model_validate(good_with(**fields))
    ExtractedFacts.model_validate(good_with(spans=[span], **fields))


@pytest.mark.parametrize("spans, missing", [
    ([], "mismatch_claim"),
    (["mismatch_detail"], "mismatch_claim"),
    (["mismatch_claim"], "mismatch_detail"),
])
@pytest.mark.parametrize("direction", ["over", "under"])
def test_mismatch_needs_both_spans(spans, missing, direction):
    with pytest.raises(ValueError, match=missing):
        ExtractedFacts.model_validate(good_with(spans=spans, claim_mismatch=direction))
    both = ["mismatch_claim", "mismatch_detail"]
    assert ExtractedFacts.model_validate(good_with(spans=both, claim_mismatch=direction)).claim_mismatch == direction


def test_described_without_mechanism_type_rejected():
    with pytest.raises(ValueError, match="mechanism_type"):
        ExtractedFacts.model_validate(good_with(spans=["hazard"], hazard_status="described"))


@pytest.mark.parametrize("status", ["unclear", "none"])
def test_mechanism_type_without_described_rejected(status):
    spans = ["hazard"] if status == "unclear" else []
    with pytest.raises(ValueError, match="mechanism_type"):
        ExtractedFacts.model_validate(good_with(spans=spans, hazard_status=status, mechanism_type="active"))


def test_response_schema_requires_every_field():
    schema = response_schema()
    assert schema["required"] == ["faults"]
    assert set(schema["$defs"]["ExtractedFacts"]["required"]) == set(ExtractedFacts.model_fields)


# --- offline reader: Panel B, same fault three phrasings ------------------

@pytest.mark.parametrize("text, alt, cope", [
    ("toilet blocked", False, False),
    ("toilet blocked, going down the servo", False, True),
    ("toilet blocked, using the other one", True, False),
])
def test_panel_b_phrasings(text, alt, cope):
    res = extract(report(text), OfflineExtractor())
    (f,) = res.extraction.faults
    assert f.taxonomy_match == ["blocked or broken toilet"]
    assert f.alternative_mentioned is alt
    assert f.coping_mentioned is cope


def test_no_fault_named_is_out_of_scope():
    res = extract(report("hi can someone call me back"), OfflineExtractor())
    assert res.status is ExtractionStatus.NO_FAULT_NAMED


def test_unknown_fault_gives_empty_match_but_is_not_out_of_scope():
    # Documents an offline stand-in limit, not intended behaviour: the LLM path is
    # expected to match "fan not working properly" here.
    text = "the ceiling fan wobbles a bit"
    res = extract(report(text), OfflineExtractor())
    (f,) = res.extraction.faults
    assert f.taxonomy_match == []
    assert res.status is ExtractionStatus.OK  # goes on to the review band
    assert f.fault_description and f.fault_description in text


def test_active_hazard_detected():
    text = "roof leaking bad, water coming through the light fitting in kids room"
    (f,) = extract(report(text), OfflineExtractor()).extraction.faults
    assert f.mechanism_type == "active"
    assert "roof leak" in f.taxonomy_match


def test_conditional_hazard_detected():
    f = extract(report("roof is leaking, if it rains it drips near the power board"),
                OfflineExtractor()).extraction.faults[0]
    assert f.mechanism_type == "conditional"


def test_offline_spans_are_real_substrings():
    text = "Toilet blocked and hot water not working, using bucket from neighbour's"
    (f,) = extract(report(text), OfflineExtractor()).extraction.faults
    assert f.quoted_spans
    for s in f.quoted_spans:
        assert s.text in text


def test_fault_names_unique():
    assert len(FAULT_NAMES) == len(set(FAULT_NAMES))


@pytest.mark.parametrize("text, expected", [
    ("stove element not working", ["stove element not working"]),
    ("stove's not working", ["stove or oven not working"]),
])
def test_stove_element_is_not_the_stove(text, expected):
    assert OfflineExtractor.read(text).faults[0].taxonomy_match == expected


def test_every_pattern_is_keyed_by_a_table_name():
    assert set(_FAULT_PATTERNS) <= set(TIER_TABLE)


def test_every_table_name_has_a_pattern():
    assert set(TIER_TABLE) <= set(_FAULT_PATTERNS)


def test_extraction_imports_only_fault_names_from_tiers():
    """Invariant 1: the tier column never reaches the module that builds the prompt."""
    tree = ast.parse(Path(extraction.__file__).read_text())
    from_tiers = [alias.name for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("tiers")
                  for alias in node.names]
    assert from_tiers == ["FAULT_NAMES"]


def test_taxonomy_match_without_fault_description_rejected():
    bad = json.loads(GOOD)
    bad["fault_description"] = None
    bad["quoted_spans"] = [s for s in bad["quoted_spans"] if s["field"] != "fault_description"]
    with pytest.raises(ValueError, match="fault_description"):
        ExtractedFacts.model_validate(bad)
    c = ScriptedClient(response(bad), response(bad))
    assert extract(report("toilet blocked"), c).status is ExtractionStatus.FLAGGED_FOR_HUMAN



# --- ReportExtraction (step 3.1b): one entry per fault ------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def test_report_extraction_requires_faults():
    with pytest.raises(ValueError, match="faults"):
        ReportExtraction.model_validate({})


def test_report_extraction_rejects_extra_field():
    with pytest.raises(ValueError, match="severity"):
        ReportExtraction.model_validate({"faults": [], "severity": "high"})


def test_report_extraction_is_frozen():
    extraction = ReportExtraction.model_validate({"faults": []})
    with pytest.raises(ValueError):
        extraction.faults = [ExtractedFacts.model_validate(json.loads(GOOD))]


def test_report_extraction_rejects_item_without_fault():
    no_fault = json.loads(GOOD) | {"fault_description": None, "taxonomy_match": [], "quoted_spans": []}
    with pytest.raises(ValueError, match="names no fault"):
        ReportExtraction.model_validate({"faults": [json.loads(GOOD), no_fault]})


def test_empty_faults_is_no_fault_named():
    assert ReportExtraction.model_validate({"faults": []}).faults == ()
    res = extract(report("hi can someone call me back"), ScriptedClient(response()))
    assert res.status is ExtractionStatus.NO_FAULT_NAMED and res.extraction.faults == ()


def test_two_faults_parse_as_two_entries():
    gas = json.loads(GOOD) | {
        "fault_description": "smell gas", "taxonomy_match": ["gas leak"],
        "quoted_spans": [{"field": "fault_description", "text": "smell gas"},
                         {"field": "taxonomy_match", "text": "smell gas"}],
    }
    res = extract(report("toilet blocked and I can smell gas"), ScriptedClient(response(json.loads(GOOD), gas)))
    assert res.status is ExtractionStatus.OK
    assert [f.taxonomy_match for f in res.extraction.faults] == [["blocked or broken toilet"], ["gas leak"]]


def test_model_facing_schema_is_a_list_of_extracted_facts():
    schema = response_schema()
    assert list(schema["properties"]) == ["faults"]
    faults = schema["properties"]["faults"]
    assert faults["type"] == "array" and faults["items"] == {"$ref": "#/$defs/ExtractedFacts"}
    item = schema["$defs"]["ExtractedFacts"]
    assert set(item["properties"]) == set(ExtractedFacts.model_fields)


def test_offline_extractor_matches_pre_list_baseline():
    """faults[0] for every report in reports/ equals what read() returned before the list change."""
    baseline = json.loads((FIXTURES / "offline_extraction_baseline.json").read_text())
    reports_dir = Path(__file__).parent.parent / "reports"
    files = sorted(reports_dir.glob("*.pdf"))
    assert [p.name for p in files] == sorted(baseline)
    for path in files:
        extraction = OfflineExtractor.read(parse_report_text(read_text(path)).raw_text)
        assert len(extraction.faults) == 1, path.name
        assert extraction.faults[0] == ExtractedFacts.model_validate(baseline[path.name]), path.name


def test_offline_extractor_no_fault_gives_empty_list():
    assert OfflineExtractor.read("hi can someone call me back").faults == ()



def test_faults_cannot_be_changed_in_place():
    good = ExtractedFacts.model_validate(json.loads(GOOD))
    extraction = ReportExtraction.model_validate({"faults": [good]})
    assert isinstance(extraction.faults, tuple)
    assert not hasattr(extraction.faults, "append")
    with pytest.raises(TypeError):
        extraction.faults[0] = good  # type: ignore[index]
    with pytest.raises(ValueError):
        extraction.faults = (good, good)  # frozen model
    assert extraction.faults == (good,)



# --- SYSTEM_PROMPT and build_user_prompt (step 3.2) ----------------------------------

REPORTS_DIR = Path(__file__).parent.parent / "reports"
LABELLED = Path(__file__).parent.parent / "data" / "synthetic" / "labelled.jsonl"


def model_facing_texts() -> dict[str, str]:
    # The template is checked with a placeholder, never tenant text, which may contain these words.
    return {"SYSTEM_PROMPT": SYSTEM_PROMPT, "template": build_user_prompt("REPORT_TEXT")}


@pytest.mark.parametrize("word", BANNED_WORDS)
def test_prompt_and_template_have_no_banned_word(word):
    pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
    for name, text in model_facing_texts().items():
        assert not pattern.search(text), f"{word!r} in {name}"


def prompt_sections() -> list[tuple[str, str]]:
    """(heading name, body) for every "### name" heading in SYSTEM_PROMPT."""
    parts = re.split(r"^### (\S+)$", SYSTEM_PROMPT, flags=re.M)
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts), 2)]


@pytest.mark.parametrize("name", ["faults", *ExtractedFacts.model_fields])
def test_prompt_has_one_non_empty_section_per_field(name):
    matching = [body for heading, body in prompt_sections() if heading == name]
    assert len(matching) == 1, f"### {name} appears {len(matching)} times"
    assert matching[0], f"### {name} has an empty body"


@pytest.mark.parametrize("rule, phrase", [
    ("empty list", "If the report names no fault, return an empty faults list"),
    ("one entry per fault", "Return one entry per distinct fault"),
    ("no match", "No match is a correct answer"),
    ("gas", 'A gas leak reported as present is "described" with mechanism_type "active"'),
    ("all candidates", "If the words could fit more than one entry, list every entry they fit; do not choose between them."),
    ("span text is report words", "Every span's text is words copied exactly from the REPORT TEXT — never a fault list name, "
     "never a field value such as 'ongoing', 'fault' or 'sign'. Add a span only where a field below says one is required."),
    ("taxonomy span", "When the list is not empty it needs its own span with field taxonomy_match, "
     "even if those are the same words as fault_description."),
    ("gas smell is the fault", "A smell of gas is the fault itself (a gas leak), so fault_or_sign is 'fault'."),
])
def test_prompt_states_required_rule(rule, phrase):
    assert " ".join(phrase.split()) in " ".join(SYSTEM_PROMPT.split()), rule


def test_user_prompt_has_report_verbatim():
    text = 'dunny\'s cooked,\n  "won\'t go down"  since Tuesday'
    assert f"<<<\n{text}\n>>>" in build_user_prompt(text)


def test_source_tag_never_reaches_the_model():
    # The tag stays on the Report; nothing model-facing names it.
    for name, prompt in model_facing_texts().items():
        for tag in SourceTag:
            assert tag.value.lower() not in prompt.lower(), f"{tag.value!r} in {name}"


def section(name: str) -> str:
    (body,) = [b for heading, b in prompt_sections() if heading == name]
    return " ".join(body.split())


def test_mechanism_type_active_means_happening_now():
    body = section("mechanism_type")
    assert "happening now" in body
    assert "could happen now" not in body


# Phrases held back for evaluating the model; if the prompt contained them, a test report
# using them would be answered from the prompt rather than understood.
RESERVED_TEST_TERMS = ("dunny", "won't go down", "chocked")


@pytest.mark.parametrize("term", RESERVED_TEST_TERMS)
def test_reserved_test_terms_never_in_prompt(term):
    for name, prompt in model_facing_texts().items():
        assert term.lower() not in prompt.lower(), f"{term!r} in {name}"


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def test_prompt_examples_never_come_from_report_files():
    corpus = [_normalise(read_text(p)) for p in sorted(REPORTS_DIR.iterdir()) if p.suffix in (".pdf", ".txt")]
    if LABELLED.exists():
        for line in LABELLED.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                corpus += [_normalise(v) for v in record.values() if isinstance(v, str)]
    assert corpus
    for key, example in PROMPT_EXAMPLES.items():
        for text in corpus:
            assert _normalise(example) not in text, key


def test_every_prompt_example_is_in_the_prompt():
    for key, example in PROMPT_EXAMPLES.items():
        assert f'"{example}"' in SYSTEM_PROMPT, key


class FakeOpenAI:
    """Swapped in for the SDK constructor; records what the client was built with."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.mark.parametrize("value", ["", None])
def test_unset_or_empty_model_and_base_url_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch,
                                                                 value: str | None) -> None:
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    for name in ("TRIAGE_MODEL", "TRIAGE_BASE_URL"):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    client = extraction.OpenAICompatibleClient(api_key="x")
    assert client.model == "gpt-4o"
    assert client._client.kwargs["base_url"] is None


def test_set_model_and_base_url_are_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("TRIAGE_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("TRIAGE_BASE_URL", "https://example.test/v1")
    client = extraction.OpenAICompatibleClient(api_key="x")
    assert (client.model, client._client.kwargs["base_url"]) == ("gpt-4o-mini", "https://example.test/v1")


def test_repo_wide_guard_refuses_the_real_api_client() -> None:
    # conftest's autouse guard: this file is not test_demo.py, and still cannot reach the API.
    with pytest.raises(AssertionError, match="real API client constructed in a test"):
        extraction.OpenAICompatibleClient(api_key="x")
