from datetime import datetime
from uuid import UUID

from triage.models import RankedJob, RankInput, RankResult

NO_TIER_FLAG = "No tier: fault isn't on the reference list — needs a tier call"


def _in_review_band(job: RankInput) -> bool:
    return job.tally is None and job.safety_level == 0


def sort_key(job: RankInput) -> tuple[int, int, int, datetime, UUID]:
    # 5.3: no tier and no safety trigger goes to the review band, so reaching here is a bug.
    if _in_review_band(job):
        raise ValueError(f"job {job.job_id} has no tier and no safety trigger; belongs in review band")
    # Invariant 3: distance is never read here (Logistics G1, equity).
    # sorted() is ascending, so values are negated where higher must come first.
    return (
        -job.safety_level,  # Safety G3
        0 if job.tally is None else 1,  # untiered safety job first in its level (invariant 6)
        -(job.tally or 0),  # Urgency G1/G2
        job.original_timestamp,  # FIFO
        job.job_id,  # deterministic; neutral only because IDs never encode region
    )


def rank(jobs: list[RankInput]) -> RankResult:
    ids = [job.job_id for job in jobs]
    duplicates = sorted({job_id for job_id in ids if ids.count(job_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate job_id in input: {', '.join(map(str, duplicates))}")

    band = [job for job in jobs if _in_review_band(job)]
    rest = [job for job in jobs if not _in_review_band(job)]
    # sorted() returns a new list, so the caller's list is never reordered.
    band_sorted = sorted(band, key=lambda job: (job.original_timestamp, job.job_id))

    ranked = tuple(
        RankedJob(
            position=position,
            job_id=job.job_id,
            # Invariant 6: a safety job without a tier is ranked, but flagged for a tier call.
            flags=(NO_TIER_FLAG,) if job.tally is None else (),
        )
        # enumerate(..., start=1) yields 1-based positions.
        for position, job in enumerate(sorted(rest, key=sort_key), start=1)
    )
    return RankResult(review_band=tuple(job.job_id for job in band_sorted), ranked=ranked)
