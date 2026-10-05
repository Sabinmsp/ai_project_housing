import json
from datetime import datetime, timezone

import pytest

from triage.extraction import (
    FAULT_NAMES,
    SYSTEM_PROMPT,
    OfflineExtractor,
    build_user_prompt,
    extract,
    response_schema,
)
from triage.intake import create_report
from triage.models import ExtractedFacts, ExtractionStatus

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


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
    "hazard_mechanism": None,
    "mechanism_type": None,
    "quoted_spans": [{"field": "fault_description", "text": "toilet blocked"},
                     {"field": "taxonomy_match", "text": "toilet blocked"}],
})


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
    for word in ("dangerous", "standard", "tier", "points", "score", "rank", "priority",
                 "severity", "urgen"):
        assert word not in text, word


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
    assert len(found) == 2  # ExtractedFacts and QuotedSpan
    for obj in found:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


# --- validation boundary --------------------------------------------------

def test_good_response_first_try():
    c = ScriptedClient(GOOD)
    res = extract(report("toilet blocked"), c)
    assert res.status is ExtractionStatus.OK and res.attempts == 1 and c.calls == 1


def test_malformed_then_good_retries_once():
    c = ScriptedClient("not json", GOOD)
    res = extract(report("toilet blocked"), c)
    assert res.status is ExtractionStatus.OK and res.attempts == 2 and len(res.errors) == 1


def test_two_failures_flag_for_human():
    c = ScriptedClient("{}garbage", '{"fault_description": 5}')
    res = extract(report("toilet blocked"), c)
    assert res.status is ExtractionStatus.FLAGGED_FOR_HUMAN
    assert res.facts is None and c.calls == 2


def test_unknown_fault_name_is_a_validation_failure():
    bad = json.loads(GOOD)
    bad["taxonomy_match"] = ["toilet emergency (severe)"]
    c = ScriptedClient(json.dumps(bad), json.dumps(bad))
    assert extract(report("toilet blocked"), c).status is ExtractionStatus.FLAGGED_FOR_HUMAN


def test_claim_without_span_rejected():
    bad = json.loads(GOOD)
    bad["coping_mentioned"] = True
    with pytest.raises(ValueError):
        ExtractedFacts.model_validate(bad)


def test_hazard_needs_mechanism_type():
    bad = json.loads(GOOD)
    bad["hazard_mechanism"] = "water in light"
    bad["quoted_spans"].append({"field": "hazard_mechanism", "text": "water in light"})
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
    c = ScriptedClient(json.dumps(bad), json.dumps(bad))
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


# --- offline reader: Panel B, same fault three phrasings ------------------

@pytest.mark.parametrize("text, alt, cope", [
    ("toilet blocked", False, False),
    ("toilet blocked, going down the servo", False, True),
    ("toilet blocked, using the other one", True, False),
])
def test_panel_b_phrasings(text, alt, cope):
    res = extract(report(text), OfflineExtractor())
    f = res.facts
    assert f.taxonomy_match == ["blocked or broken toilet"]
    assert f.alternative_mentioned is alt
    assert f.coping_mentioned is cope


def test_no_fault_named_is_out_of_scope():
    res = extract(report("hi can someone call me back"), OfflineExtractor())
    assert res.status is ExtractionStatus.NO_FAULT_NAMED


def test_unknown_fault_gives_empty_match_but_is_not_out_of_scope():
    text = "the ceiling fan wobbles a bit"
    res = extract(report(text), OfflineExtractor())
    assert res.facts.taxonomy_match == []
    assert res.status is ExtractionStatus.OK  # goes on to the review band
    assert res.facts.fault_description and res.facts.fault_description in text


def test_active_hazard_detected():
    text = "roof leaking bad, water coming through the light fitting in kids room"
    f = extract(report(text), OfflineExtractor()).facts
    assert f.mechanism_type == "active"
    assert "serious roof leak" in f.taxonomy_match


def test_conditional_hazard_detected():
    f = extract(report("roof is leaking, if it rains it drips near the power board"),
                OfflineExtractor()).facts
    assert f.mechanism_type == "conditional"


def test_offline_spans_are_real_substrings():
    text = "Toilet blocked and hot water not working, using bucket from neighbour's"
    f = extract(report(text), OfflineExtractor()).facts
    assert f.quoted_spans
    for s in f.quoted_spans:
        assert s.text in text


def test_fault_names_unique():
    assert len(FAULT_NAMES) == len(set(FAULT_NAMES))
