from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_DNS, uuid4, uuid5

import pytest
from pydantic import ValidationError

from triage.models import RankedJob, RankInput


def valid_job() -> dict[str, Any]:
    return {
        "job_id": uuid4(),
        "safety_level": 1,
        "tally": 3,
        "original_timestamp": datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
        "distance_km": 12.5,
    }


def test_valid_job_accepted() -> None:
    job = RankInput(**valid_job())
    assert job.tally == 3


def test_non_uuid_job_id_rejected() -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "job_id": "DAR-0012"})


# parametrize runs the test once per value in the list.
@pytest.mark.parametrize("level", [-1, 3])
def test_safety_level_out_of_range_rejected(level: int) -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "safety_level": level})


@pytest.mark.parametrize("tally", [1, 5])
def test_tally_out_of_range_rejected(tally: int) -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "tally": tally})


def test_tally_none_accepted() -> None:
    job = RankInput(**{**valid_job(), "tally": None})
    assert job.tally is None


def test_missing_timestamp_rejected() -> None:
    data = valid_job()
    del data["original_timestamp"]
    with pytest.raises(ValidationError):
        RankInput(**data)


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "original_timestamp": datetime(2026, 10, 1, 9, 0)})


def test_negative_distance_rejected() -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "distance_km": -0.1})


def test_mutation_raises() -> None:
    job = RankInput(**valid_job())
    with pytest.raises(ValidationError):
        job.tally = 4


@pytest.mark.parametrize("level", [True, "2"])
def test_safety_level_not_coerced(level: object) -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "safety_level": level})


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "override": True})


@pytest.mark.parametrize("distance", [float("inf"), float("nan")])
def test_non_finite_distance_rejected(distance: float) -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "distance_km": distance})


def test_non_random_uuid_rejected() -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "job_id": uuid5(NAMESPACE_DNS, "DAR-0012")})


@pytest.mark.parametrize("timestamp", [0, "2026-10-01T09:00:00"])
def test_numeric_or_naive_timestamp_rejected(timestamp: object) -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "original_timestamp": timestamp})


def test_iso_timestamp_with_timezone_accepted() -> None:
    job = RankInput(**{**valid_job(), "original_timestamp": "2026-10-01T09:00:00+00:00"})
    assert job.original_timestamp == datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def test_json_dump_round_trips() -> None:
    job = RankInput(**valid_job())
    assert RankInput(**job.model_dump(mode="json")) == job


@pytest.mark.parametrize("tally", ["4", 4.0])
def test_tally_not_coerced(tally: object) -> None:
    with pytest.raises(ValidationError):
        RankInput(**{**valid_job(), "tally": tally})


@pytest.mark.parametrize(
    "fields",
    [{"decided_by": ""}, {"decided_by": "   "}, {"flags": ("",)}, {"flags": ("ok", "  ")}],
)
def test_blank_reason_strings_rejected(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RankedJob(**{"position": 1, "job_id": uuid4(), "decided_by": "top of list", **fields})
