from datetime import datetime, timedelta, timezone

from hypothesis import given
from hypothesis import strategies as st

from triage.models import RankInput, RankResult
from triage.ranking import rank

# st.datetimes takes naive bounds; timezones= then attaches the zone to each value.
LATEST = datetime(2026, 10, 4, 9, 0)
EARLIEST = LATEST - timedelta(days=90)

distances = st.floats(min_value=0, max_value=2000)

jobs_strategy = st.lists(
    # st.builds calls RankInput(...) with a value drawn from each strategy.
    st.builds(
        RankInput,
        job_id=st.uuids(),
        safety_level=st.integers(min_value=0, max_value=2),
        tally=st.none() | st.integers(min_value=2, max_value=4),
        original_timestamp=st.datetimes(EARLIEST, LATEST, timezones=st.just(timezone.utc)),
        distance_km=distances,
    ),
    max_size=40,
    # unique_by stops Hypothesis generating duplicate IDs, which rank() rejects.
    unique_by=lambda job: job.job_id,
)


def ranked_jobs(jobs: list[RankInput], result: RankResult) -> list[RankInput]:
    by_id = {job.job_id: job for job in jobs}
    return [by_id[entry.job_id] for entry in result.ranked]


@given(jobs_strategy)
def test_p1_no_job_ranked_above_higher_safety(jobs: list[RankInput]) -> None:
    levels = [job.safety_level for job in ranked_jobs(jobs, rank(jobs))]
    assert levels == sorted(levels, reverse=True)


@given(jobs_strategy)
def test_p2_equal_safety_and_tally_older_first(jobs: list[RankInput]) -> None:
    ordered = ranked_jobs(jobs, rank(jobs))
    for i, above in enumerate(ordered):
        for below in ordered[i + 1 :]:
            if (above.safety_level, above.tally) == (below.safety_level, below.tally):
                assert above.original_timestamp <= below.original_timestamp


# st.data() lets the test draw further values mid-test, sized to the list it already has.
@given(jobs_strategy, st.data())
def test_p3_distance_never_changes_result(jobs: list[RankInput], data: st.DataObject) -> None:
    new_distances = data.draw(st.lists(distances, min_size=len(jobs), max_size=len(jobs)))
    moved = [job.model_copy(update={"distance_km": d}) for job, d in zip(jobs, new_distances)]
    assert rank(moved) == rank(jobs)


@given(jobs_strategy, st.data())
def test_p4_shuffling_never_changes_result(jobs: list[RankInput], data: st.DataObject) -> None:
    shuffled = data.draw(st.permutations(jobs))
    assert rank(shuffled) == rank(jobs)


@given(jobs_strategy)
def test_p5_band_exactly_when_untiered_and_no_safety(jobs: list[RankInput]) -> None:
    result = rank(jobs)
    band = set(result.review_band)
    ranked = {entry.job_id for entry in result.ranked}
    assert band.isdisjoint(ranked)
    for job in jobs:
        in_band = job.tally is None and job.safety_level == 0
        assert (job.job_id in band) == in_band
        assert (job.job_id in ranked) == (not in_band)
