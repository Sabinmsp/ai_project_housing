"""Ranking: a pure, deterministic sort of jobs into a queue plus a review band.

Pipeline: intake -> extraction -> verification -> evaluation -> ranking -> explain.
Input: RankInput per job (from adapter.to_rank_input). Output: RankResult, with a
decided_by reason per position for the coordinator's why-trace. No model, no I/O.
"""

from datetime import datetime

from triage.models import RankedJob, RankInput, RankResult

NO_TIER_FLAG = "No tier: fault isn't on the reference list — needs a tier call"
REVIEW_BAND_REASON = "not on the reference list, no safety trigger — awaiting a tier call"
# Index i explains slot i of sort_key; a test pins the two lengths together.
SLOT_REASONS = (
    "below #{above}: higher safety level above",
    "below #{above}: untiered safety job above, needs a human tier call first",
    "below #{above}: higher urgency score above",
    "tied on safety and urgency; earlier report wins",
    "identical; order arbitrary but fixed",
)
# Replaces the timestamp-slot reason when neither job has an urgency score to tie on.
UNTIERED_TIE_REASON = "both untiered at the same safety level; earlier report wins"


def _in_review_band(job: RankInput) -> bool:
    # Master §4.4: no tier and no safety trigger waits for a tier call instead of being ranked.
    return job.tally is None and job.safety_level == 0


def sort_key(job: RankInput) -> tuple[int, int, int, datetime, str]:
    """The sort key for one ranked job (invariant 5); lower sorts first.

    Raises:
        ValueError: the job belongs in the review band, so it must never be sorted.
    """
    if _in_review_band(job):
        raise ValueError(f"job {job.job_id} has no tier and no safety trigger; belongs in review band")
    # Invariant 5, Logistics G1: distance, logistics and overrides are never read here, so a
    # remote job can't sink for being far from town.
    # sorted() is ascending, so values are negated where higher must come first.
    return (
        -job.safety_level,  # Safety G3: safety sits above urgency and can't be outweighed
        # A safety job with no tier (e.g. roof collapse) must never sink for lack of a tier call.
        0 if job.tally is None else 1,
        -(job.tally or 0),  # Urgency G1/G2
        job.original_timestamp,  # FIFO on when the tenant reported it, never re-stamped (invariant 6)
        job.job_id,  # deterministic; neutral only because IDs never encode region (invariant 7)
    )


def rank(jobs: list[RankInput]) -> RankResult:
    """Split jobs into the review band (oldest first) and the ranked queue.

    Raises:
        ValueError: a job fails validation or two jobs share a job_id.
    """
    # model_copy(update=...) skips validation, so re-check every job. vars() is used, not
    # model_dump(), because model_dump() silently drops unknown fields such as "override".
    # Pydantic's ValidationError is a ValueError subclass.
    jobs = [RankInput.model_validate(vars(job)) for job in jobs]
    ids = [job.job_id for job in jobs]
    duplicates = sorted({job_id for job_id in ids if ids.count(job_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate job_id in input: {', '.join(map(str, duplicates))}")

    band = [job for job in jobs if _in_review_band(job)]
    rest = [job for job in jobs if not _in_review_band(job)]
    # sorted() returns a new list, so the caller's list is never reordered.
    band_sorted = sorted(band, key=lambda job: (job.original_timestamp, job.job_id))

    ordered = sorted(rest, key=sort_key)
    ranked = tuple(
        RankedJob(
            position=position,
            job_id=job.job_id,
            # A safety job without a tier is ranked, but flagged for a tier call (invariant 5).
            flags=(NO_TIER_FLAG,) if job.tally is None else (),
            decided_by=_decided_by(position, ordered[position - 2] if position > 1 else None, job),
        )
        for position, job in enumerate(ordered, start=1)
    )
    return RankResult(review_band=tuple(job.job_id for job in band_sorted), ranked=ranked)


def _decided_by(position: int, above: RankInput | None, job: RankInput) -> str:
    if above is None:
        return "top of list"
    # job_ids are unique, so some slot always differs.
    slot = next(i for i, (a, b) in enumerate(zip(sort_key(above), sort_key(job))) if a != b)
    # Slot 3 is the timestamp; reaching it with tally None means both jobs are untiered.
    if slot == 3 and job.tally is None:
        return UNTIERED_TIE_REASON
    return SLOT_REASONS[slot].format(above=position - 1)
