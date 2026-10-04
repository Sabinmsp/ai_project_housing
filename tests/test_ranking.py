from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from triage.models import RankInput
from triage.ranking import NO_TIER_FLAG, rank, sort_key

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


def test_untiered_non_safety_job_only_in_review_band() -> None:
    brown_water = job(0, None)
    result = rank([brown_water, job(1, 3)])
    assert result.review_band == (brown_water.job_id,)
    assert brown_water.job_id not in [entry.job_id for entry in result.ranked]


def test_untiered_safety_job_flagged_tiered_jobs_not() -> None:
    roof = job(2, None)
    gas = job(2, 4)
    tap = job(0, 2)
    flags = {entry.job_id: entry.flags for entry in rank([gas, tap, roof]).ranked}
    assert flags == {roof.job_id: (NO_TIER_FLAG,), gas.job_id: (), tap.job_id: ()}


def test_review_band_oldest_first() -> None:
    newest = job(0, None)
    oldest = job(0, None, NOW - timedelta(days=3))
    middle = job(0, None, NOW - timedelta(days=1))
    result = rank([newest, oldest, middle])
    assert result.review_band == (oldest.job_id, middle.job_id, newest.job_id)


def test_positions_are_one_to_n_without_gaps() -> None:
    jobs = [job(2, 3), job(0, None), job(1, None), job(0, 4), job(1, 2)]
    positions = [entry.position for entry in rank(jobs).ranked]
    assert positions == [1, 2, 3, 4]


def test_duplicate_job_id_raises() -> None:
    shared = uuid4()
    with pytest.raises(ValueError):
        rank([job(1, 3, job_id=shared), job(2, 4, job_id=shared)])


def test_empty_input_returns_empty_result() -> None:
    result = rank([])
    assert result.review_band == ()
    assert result.ranked == ()
