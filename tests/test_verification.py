from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from triage.models import ExtractedFacts
from triage.verification import claim_spans, verify_spans

REPORT = "The toilet is BLOCKED again.\nWe're using a bucket and it's  coming and going."


def facts(spans: list[tuple[str, str]], **fields: Any) -> ExtractedFacts:
    base: dict[str, Any] = {
        "fault_description": "toilet is blocked",
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
    }
    claim = [("fault_description", "toilet is blocked"), ("taxonomy_match", "toilet is blocked")]
    return ExtractedFacts.model_validate(
        base | fields | {"quoted_spans": [{"field": f, "text": t} for f, t in claim + spans]}
    )


def test_all_quotes_found_gives_nothing_unverified() -> None:
    assert verify_spans(REPORT, facts([])) == frozenset()


def test_fabricated_quote_on_claimed_field_is_unverified() -> None:
    f = facts([("coping_mentioned", "staying at my aunty's")], coping_mentioned=True)
    assert verify_spans(REPORT, f) == {("coping_mentioned", "staying at my aunty's")}


@pytest.mark.parametrize("quote", ["TOILET is blocked", "toilet  is   blocked", "toilet is\nblocked", " toilet is blocked "])
def test_case_and_whitespace_differences_still_verify(quote: str) -> None:
    f = facts([("coping_mentioned", "using a bucket"), ("impact_status", quote)], coping_mentioned=True,
              impact_status="intermittent")
    assert verify_spans(REPORT, f) == frozenset()


def test_near_match_is_not_the_tenants_words() -> None:
    # Every word is in the report but not as one phrase: no fuzzy or partial matching.
    f = facts([("coping_mentioned", "bucket using")], coping_mentioned=True)
    assert verify_spans(REPORT, f) == {("coping_mentioned", "bucket using")}


def test_whitespace_only_quote_is_unverified() -> None:
    f = facts([("coping_mentioned", "   ")], coping_mentioned=True)
    assert verify_spans(REPORT, f) == {("coping_mentioned", "   ")}


def test_ongoing_span_is_dropped_not_unverified() -> None:
    f = facts([("impact_status", "ongoing")])
    assert verify_spans(REPORT, f) == frozenset()
    assert ("impact_status", "ongoing") not in {(s.field, s.text) for s in claim_spans(f)}


def test_fabricated_sign_span_is_unverified_not_dropped() -> None:
    f = facts([("fault_or_sign", "smells of sewage")], fault_or_sign="sign")
    assert verify_spans(REPORT, f) == {("fault_or_sign", "smells of sewage")}


def test_fabricated_hazard_span_with_described_status_is_unverified() -> None:
    f = facts([("hazard", "water on the wires")], hazard_status="described", mechanism_type="active")
    assert verify_spans(REPORT, f) == {("hazard", "water on the wires")}


def test_fabricated_hazard_span_with_unclear_status_is_unverified_not_dropped() -> None:
    f = facts([("hazard", "invented words")], hazard_status="unclear")
    assert verify_spans(REPORT, f) == {("hazard", "invented words")}


def test_taxonomy_span_quoting_a_list_name_is_unverified() -> None:
    data = facts([]).model_dump()
    data["quoted_spans"] = [{"field": "fault_description", "text": "toilet is blocked"},
                            {"field": "taxonomy_match", "text": "blocked or broken toilet"}]
    assert verify_spans(REPORT, ExtractedFacts.model_validate(data)) == {("taxonomy_match", "blocked or broken toilet")}


def test_fault_description_span_never_dropped() -> None:
    data = facts([]).model_dump()
    data["quoted_spans"][0]["text"] = "toilet's cactus"
    assert verify_spans(REPORT, ExtractedFacts.model_validate(data)) == {("fault_description", "toilet's cactus")}


# "Never" rule: a span that backs no claim never changes the result, whatever its text.
NO_CLAIM_FIELDS = ["impact_status", "alternative_mentioned", "coping_mentioned", "harm_claimed",
                   "worsening_mentioned", "fault_or_sign", "hazard", "mismatch_claim", "mismatch_detail"]
BASES = [
    facts([]),  # every claim-free field at its no-quote value
    facts([("coping_mentioned", "made up coping")], coping_mentioned=False),
    facts([("fault_or_sign", "invented")], fault_or_sign="fault"),
]


@given(st.sampled_from(BASES), st.sampled_from(NO_CLAIM_FIELDS), st.text(min_size=1))
def test_no_claim_span_never_changes_result(base: ExtractedFacts, field: str, text: str) -> None:
    data = base.model_dump()
    data["quoted_spans"].append({"field": field, "text": text})
    with_extra = ExtractedFacts.model_validate(data)
    assert verify_spans(REPORT, with_extra) == verify_spans(REPORT, base)
