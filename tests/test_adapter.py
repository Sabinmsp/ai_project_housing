from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from triage.adapter import to_rank_input
from triage.evaluation import NO_ALTERNATIVE, NO_HAZARD
from triage.models import EnrichedJob
from triage.ranking import rank

MON = datetime(2026, 9, 28, 9, 0, tzinfo=timezone(timedelta(hours=9, minutes=30)))


def enriched(**overrides: Any) -> EnrichedJob:
    fields: dict[str, Any] = {
        "request_id": "R-0000AAAA",
        "community": "Wadeye",
        "original_report_timestamp": MON,
        "tier": "standard",
        "tier_entry": "power point not working",
        "base_points": 2,
        "severity_bump": 1,
        "urgency_tally": 3,
        "tally_reasons": (NO_ALTERNATIVE,),
        "safety_level": "none",
        "safety_reason": NO_HAZARD,
        "flags": (),
    }
    return EnrichedJob(**{**fields, **overrides})


@pytest.mark.parametrize(
    ("safety_level", "safety_flag", "expected"),
    [("active", True, 2), ("conditional", False, 1), ("none", False, 0)],
)
def test_safety_level_mapped_to_int(safety_level: str, safety_flag: bool, expected: int) -> None:
    job = enriched(safety_level=safety_level, safety_flag=safety_flag)
    assert to_rank_input(job).safety_level == expected


def test_request_id_becomes_job_id() -> None:
    assert to_rank_input(enriched(request_id="R-C5581F02")).job_id == "R-C5581F02"


def test_urgency_tally_becomes_tally() -> None:
    assert to_rank_input(enriched()).tally == 3


def test_none_tally_stays_none() -> None:
    job = enriched(tier=None, tier_entry=None, base_points=None, severity_bump=None, urgency_tally=None)
    assert to_rank_input(job).tally is None


def test_report_timestamp_becomes_original_timestamp() -> None:
    assert to_rank_input(enriched()).original_timestamp == MON


def test_distance_carried_for_display() -> None:
    assert to_rank_input(enriched(distance_cost_km=412.5)).distance_km == 412.5


def test_missing_distance_stays_none() -> None:
    assert to_rank_input(enriched(distance_cost_km=None)).distance_km is None


@pytest.mark.parametrize(
    ("safety_level", "safety_flag"),
    [("active", False), ("conditional", True), ("none", True)],
)
def test_safety_flag_mismatch_raises(safety_level: str, safety_flag: bool) -> None:
    with pytest.raises(ValueError, match="disagrees"):
        to_rank_input(enriched(safety_level=safety_level, safety_flag=safety_flag))


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        to_rank_input(enriched(original_report_timestamp=datetime(2026, 9, 28, 9, 0)))


def test_end_to_end_order() -> None:
    blocked_toilet = enriched(request_id="R-TOILET", urgency_tally=3, distance_cost_km=5.0)
    roof_into_light = enriched(
        request_id="R-ROOF",
        tier="dangerous",
        tier_entry="roof leak",
        base_points=3,
        severity_bump=0,
        urgency_tally=3,
        safety_level="conditional",
        original_report_timestamp=MON + timedelta(days=1),
    )
    no_hot_water = enriched(
        request_id="R-HOTWATER",
        base_points=3,
        urgency_tally=4,
        original_report_timestamp=MON - timedelta(days=1),
        distance_cost_km=600.0,
    )
    result = rank([to_rank_input(j) for j in (blocked_toilet, roof_into_light, no_hot_water)])
    assert [entry.job_id for entry in result.ranked] == ["R-ROOF", "R-HOTWATER", "R-TOILET"]


# --- EnrichedJob rules the adapter relies on ----------------------------------


@pytest.mark.parametrize(
    "field", ["tier", "tier_entry", "base_points", "severity_bump", "urgency_tally", "tally_reasons", "safety_level", "safety_reason", "flags"]
)
def test_enriched_job_field_without_default_must_be_given(field: str) -> None:
    fields = enriched().model_dump()
    del fields[field]
    with pytest.raises(ValidationError, match=field):
        EnrichedJob.model_validate(fields)


def test_safety_flag_must_match_level_at_construction() -> None:
    with pytest.raises(ValidationError, match="disagrees"):
        enriched(safety_level="active", safety_flag=False)


def test_review_band_needs_no_tally_and_no_safety() -> None:
    untiered = {"tier": None, "tier_entry": None, "base_points": None, "severity_bump": None, "urgency_tally": None, "tally_reasons": ()}
    assert enriched(**untiered).in_review_band
    assert not enriched(**untiered, safety_level="conditional").in_review_band
    assert not enriched().in_review_band


def test_tier_entry_must_back_the_stated_tier() -> None:
    with pytest.raises(ValidationError, match="tier_entry"):
        enriched(tier_entry="gas leak")  # a dangerous entry on a standard job
    with pytest.raises(ValidationError, match="tier_entry"):
        enriched(tier_entry="serious roof leak")  # not a table name
