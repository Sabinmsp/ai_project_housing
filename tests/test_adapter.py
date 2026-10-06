from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from triage.adapter import to_rank_input
from triage.models import EnrichedJob
from triage.ranking import rank

MON = datetime(2026, 9, 28, 9, 0, tzinfo=timezone(timedelta(hours=9, minutes=30)))


def enriched(**overrides: Any) -> EnrichedJob:
    fields: dict[str, Any] = {
        "request_id": "R-0000AAAA",
        "community": "Wadeye",
        "original_report_timestamp": MON,
        "tier": "standard",
        "base_points": 2,
        "no_redundancy": 1,
        "urgency_tally": 3,
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
    job = enriched(tier=None, base_points=None, no_redundancy=0, urgency_tally=None)
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
        base_points=3,
        no_redundancy=0,
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
