from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from triage.models import RankInput


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
