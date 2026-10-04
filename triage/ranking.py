from datetime import datetime
from uuid import UUID

from triage.models import RankInput


def sort_key(job: RankInput) -> tuple[int, int, int, datetime, UUID]:
    # 5.3: no tier and no safety trigger goes to the review band, so reaching here is a bug.
    if job.tally is None and job.safety_level == 0:
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
