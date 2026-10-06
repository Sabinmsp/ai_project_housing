from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from triage.models import RankInput
from triage.ranking import NO_TIER_FLAG, SLOT_REASONS, UNTIERED_TIE_REASON, rank, sort_key

NOW = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)


def job(safety_level: int, tally: int | None, timestamp: datetime = NOW, job_id: str | None = None) -> RankInput:
    return RankInput(
        job_id=job_id or str(uuid4()),
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
    low_id, high_id = "R-0000AAAA", "R-0000BBBB"
    low = job(1, 3, job_id=low_id)
    high = job(1, 3, job_id=high_id)
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
    shared = "R-0000AAAA"
    with pytest.raises(ValueError):
        rank([job(1, 3, job_id=shared), job(2, 4, job_id=shared)])


def test_empty_input_returns_empty_result() -> None:
    result = rank([])
    assert result.review_band == ()
    assert result.ranked == ()


MON = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)
TUE, WED, THU = (MON + timedelta(days=n) for n in (1, 2, 3))


def test_realistic_queue() -> None:
    gas = job(2, 4, WED)
    roof = job(2, None, THU)
    wire = job(1, 4, MON)
    toilet_a = job(0, 4, MON)
    toilet_b = job(0, 4, TUE)
    stove = job(0, 3, MON)
    brown_water = job(0, None, MON)

    result = rank([stove, toilet_b, brown_water, gas, toilet_a, roof, wire])

    expected = [roof, gas, wire, toilet_a, toilet_b, stove]
    assert [entry.job_id for entry in result.ranked] == [j.job_id for j in expected]
    assert result.review_band == (brown_water.job_id,)
    flagged = [entry.job_id for entry in result.ranked if NO_TIER_FLAG in entry.flags]
    assert flagged == [roof.job_id]
    assert [entry.decided_by for entry in result.ranked] == [
        "top of list",
        "below #1: untiered safety job above, needs a human tier call first",
        "below #2: higher safety level above",
        "below #3: higher safety level above",
        "tied on safety and urgency; earlier report wins",
        "below #5: higher urgency score above",
    ]


def test_escalated_job_keeps_its_place_by_original_timestamp() -> None:
    # This proves rank() honours the preserved original_timestamp; it does not prove
    # escalation preserves it (invariant 4), because model_copy here sets that up by hand.
    # model_copy(update=...) skips validation, so the update must already be valid.
    reported_monday = job(1, 2, MON)
    escalated = reported_monday.model_copy(update={"tally": 3})
    reported_tuesday = job(1, 3, TUE)

    result = rank([reported_tuesday, escalated])

    assert escalated.original_timestamp == MON
    assert [entry.job_id for entry in result.ranked] == [escalated.job_id, reported_tuesday.job_id]


def test_conditional_low_tally_above_no_safety_high_tally() -> None:
    conditional = job(1, 2)
    no_safety = job(0, 4)
    result = rank([no_safety, conditional])
    assert [entry.job_id for entry in result.ranked] == [conditional.job_id, no_safety.job_id]


def test_untiered_conditional_job_first_in_its_level() -> None:
    gas = job(2, 2)
    ac_sparking = job(1, None)
    wire = job(1, 4)
    result = rank([wire, ac_sparking, gas])
    assert [entry.job_id for entry in result.ranked] == [gas.job_id, ac_sparking.job_id, wire.job_id]
    assert result.ranked[1].flags == (NO_TIER_FLAG,)


def test_all_untiered_non_safety_jobs_go_to_review_band() -> None:
    newest = job(0, None)
    oldest = job(0, None, NOW - timedelta(days=2))
    middle = job(0, None, NOW - timedelta(days=1))
    result = rank([newest, oldest, middle])
    assert result.ranked == ()
    assert result.review_band == (oldest.job_id, middle.job_id, newest.job_id)


def decided_by(jobs: list[RankInput]) -> list[str]:
    return [entry.decided_by for entry in rank(jobs).ranked]


def test_decided_by_top_of_list() -> None:
    assert decided_by([job(1, 3)]) == ["top of list"]


def test_decided_by_safety() -> None:
    assert decided_by([job(1, 4), job(2, 2)])[1] == "below #1: higher safety level above"


def test_decided_by_tally_missing() -> None:
    assert decided_by([job(2, 4), job(2, None)])[1] == (
        "below #1: untiered safety job above, needs a human tier call first"
    )


def test_decided_by_tally() -> None:
    low = job(1, 2)
    high = job(1, 4)
    result = rank([low, high])
    assert [entry.job_id for entry in result.ranked] == [high.job_id, low.job_id]
    assert result.ranked[1].decided_by == "below #1: higher urgency score above"


def test_decided_by_timestamp() -> None:
    assert decided_by([job(1, 3), job(1, 3, MON)])[1] == "tied on safety and urgency; earlier report wins"


def test_decided_by_job_id() -> None:
    low_id, high_id = "R-0000AAAA", "R-0000BBBB"
    jobs = [job(1, 3, job_id=high_id), job(1, 3, job_id=low_id)]
    assert decided_by(jobs)[1] == "identical; order arbitrary but fixed"


def test_decided_by_untiered_tie() -> None:
    assert decided_by([job(1, None), job(1, None, MON)])[1] == UNTIERED_TIE_REASON


def test_decided_by_untiered_same_timestamp_falls_to_job_id() -> None:
    assert decided_by([job(1, None), job(1, None)])[1] == "identical; order arbitrary but fixed"


def test_decided_by_untiered_different_safety() -> None:
    assert decided_by([job(1, None), job(2, None)])[1] == "below #1: higher safety level above"


def test_one_decided_by_reason_per_sort_key_slot() -> None:
    assert len(SLOT_REASONS) == len(sort_key(job(1, 3)))


@pytest.mark.parametrize("update", [{"safety_level": 7}, {"override": True}])
def test_rank_revalidates_model_copy_input(update: dict[str, object]) -> None:
    bad = job(1, 3).model_copy(update=update)
    with pytest.raises(ValueError):
        rank([job(2, 4), bad])
