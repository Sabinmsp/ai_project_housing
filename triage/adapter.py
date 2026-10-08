"""Adapter from evaluation's EnrichedJob to ranking's RankInput.

Pipeline: intake -> extraction -> verification -> evaluation -> ranking -> explain.
The one place a safety level name becomes the integer the sort key uses.
"""

from triage.models import EnrichedJob, RankInput

# Safety G3: 2 active, 1 conditional, 0 none.
_SAFETY_LEVELS = {"active": 2, "conditional": 1, "none": 0}


def to_rank_input(job: EnrichedJob) -> RankInput:
    """The fields ranking needs from one job.

    Raises:
        ValueError: safety_flag disagrees with safety_level.
    """
    # A disagreement means upstream data is corrupt, and picking either side could
    # under-rank a hazard.
    if job.safety_flag != (job.safety_level == "active"):
        raise ValueError(
            f"job {job.request_id}: safety_flag={job.safety_flag} disagrees with safety_level={job.safety_level!r}"
        )
    return RankInput(
        job_id=job.request_id,
        safety_level=_SAFETY_LEVELS[job.safety_level],
        tally=job.urgency_tally,
        original_timestamp=job.original_report_timestamp,
        distance_km=job.distance_cost_km,
    )
