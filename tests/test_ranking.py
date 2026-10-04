from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from triage.models import RankInput
from triage.ranking import sort_key

NOW = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)


def job(safety_level: int, tally: int | None, timestamp: datetime = NOW, job_id: UUID | None = None) -> RankInput:
    return RankInput(
        job_id=job_id or uuid4(),
        safety_level=safety_level,
        tally=tally,
        original_timestamp=timestamp,
    )


def test_higher_safety_beats_higher_tally() -> None:
    gas = job(2, 3)
    wire = job(1, 4, NOW - timedelta(days=5))
    assert sorted([wire, gas], key=sort_key) == [gas, wire]


def test_untiered_safety_job_before_tiered_at_same_level() -> None:
    roof = job(2, None)
    gas = job(2, 4)
    assert sorted([gas, roof], key=sort_key) == [roof, gas]


def test_older_first_when_safety_and_tally_equal() -> None:
    older = job(1, 3, NOW - timedelta(days=2))
    newer = job(1, 3)
    assert sorted([newer, older], key=sort_key) == [older, newer]


def test_job_id_breaks_full_tie_deterministically() -> None:
    low = job(1, 3, job_id=UUID(int=1))
    high = job(1, 3, job_id=UUID(int=2))
    forward = sorted([high, low], key=sort_key)
    assert forward == [low, high]
    assert sorted([low, high], key=sort_key) == forward


def test_review_band_job_raises() -> None:
    with pytest.raises(ValueError):
        sort_key(job(0, None))
