from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from triage.adapter import to_rank_input
from triage.models import EnrichedJob, RankedJob, RankResult, Reason, VerifiedSpan
from triage.ranking import REVIEW_BAND_REASON

TIER_SOURCE = "NT Government published fault list"
SAFETY_REASONS = {
    2: "safety level 2: active hazard, full override",
    1: "safety level 1: conditional hazard, elevated, check-in window applies",
    0: "safety level 0: no hazard mechanism described",
}


class ReasoningTrace(BaseModel):
    """Structured explanation. Both renderers read this and nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    fault_description: str | None
    taxonomy_match: tuple[str, ...]
    # None: untiered safety job, ranked but awaiting a tier call (invariant 6).
    tier: Literal["dangerous", "standard"] | None
    base_points: int | None
    no_redundancy: int
    no_redundancy_reason: str
    defaults_applied: tuple[str, ...]
    urgency_tally: int | None
    safety_level: int
    safety_reason: str
    evidence_spans: tuple[VerifiedSpan, ...]
    flags: tuple[Reason, ...]
    original_timestamp: datetime
    position: int
    queue_length: int
    decided_by: str
    distance_km: float | None
    logistics_notes: tuple[str, ...]


def _verified_fault_text(job: EnrichedJob) -> str | None:
    # Panel D: no trace value originates inside the model. fault_description is the model's
    # wording and may be paraphrased, so only a Stage 3 verified span is used.
    return next((s.text for s in job.spans if s.verified and s.field == "fault_description"), None)


def _logistics_notes(job: EnrichedJob) -> tuple[str, ...]:
    notes: list[str] = []
    if job.capacity_block_flag:
        notes.append(
            "No qualified trade available today; rank kept"
            + (f", next actionable: {job.next_actionable}" if job.next_actionable else "")
        )
    if job.shared_route_opportunities:
        notes.append("Shared route possible with: " + ", ".join(job.shared_route_opportunities))
    if job.starvation_line:
        notes.append(f"{job.community}: {job.starvation_line}")
    return tuple(notes)


def build_trace(entry: RankedJob, job: EnrichedJob, queue_length: int) -> ReasoningTrace:
    # Stage 4b severity default: +1 unless the text names a genuine alternative.
    if job.tier is None:
        nr_reason, defaults = "not scored: no tier", ()
    elif job.no_redundancy:
        nr_reason = "no working alternative named in the report"
        defaults = ("no-redundancy default applied: +1 (no alternative named)",)
    else:
        nr_reason = "report names a working alternative"
        defaults = ("no-redundancy default removed: +0 (alternative named)",)

    # RankedJob doesn't carry safety_level; the adapter is the one place that maps it.
    safety_level = to_rank_input(job).safety_level
    return ReasoningTrace(
        job_id=entry.job_id,
        fault_description=_verified_fault_text(job),
        taxonomy_match=tuple(job.taxonomy_match),
        tier=job.tier,
        base_points=job.base_points,
        no_redundancy=job.no_redundancy,
        no_redundancy_reason=nr_reason,
        defaults_applied=defaults,
        urgency_tally=job.urgency_tally,
        safety_level=safety_level,
        safety_reason=SAFETY_REASONS[safety_level],
        evidence_spans=tuple(s for s in job.spans if s.verified),
        # Stage 3-5 flags first, then ranking's own (e.g. the no-tier flag).
        flags=(*job.flags, *entry.flags),
        original_timestamp=job.original_report_timestamp,
        position=entry.position,
        queue_length=queue_length,
        decided_by=entry.decided_by,
        distance_km=job.distance_cost_km,
        logistics_notes=_logistics_notes(job),
    )


def build_traces(result: RankResult, jobs: dict[str, EnrichedJob]) -> list[ReasoningTrace]:
    # A missing job_id raises KeyError: a ranked job with no source data is a bug.
    return [build_trace(entry, jobs[entry.job_id], len(result.ranked)) for entry in result.ranked]


def render_tenant_sms(trace: ReasoningTrace) -> str:
    # Pure template: every word is fixed text or a trace field, so it cannot invent a reason.
    # §5.2: own-job facts only, so no position, queue length, decided_by or distance.
    fault = trace.fault_description or "your repair"
    lines = [f'Housing repair {trace.job_id}: we have your report about "{fault}".']
    if trace.safety_level > 0:
        lines.append("It is marked as a safety job.")
    if trace.tier is None:
        lines.append("A coordinator is confirming its priority.")
    else:
        lines.append(f"It is classed {trace.tier} on the {TIER_SOURCE}.")
    lines.append(f"Reply with {trace.job_id} if things get worse.")
    return " ".join(lines)


def _table(rows: list[tuple[str, str, str]]) -> str:
    width = max(len(key) for key, _, _ in rows)
    return "\n".join(f"{key.ljust(width)}  {value}  {note}".rstrip() for key, value, note in rows)


def render_coordinator(trace: ReasoningTrace) -> str:
    rows = [("job_id", trace.job_id, "")]
    rows += [("taxonomy_match", m, "(NT fault list name)") for m in trace.taxonomy_match]
    if trace.tier is None:
        rows.append(("tier", "untiered", "(needs a coordinator tier call)"))
    else:
        rows += [
            ("tier", trace.tier, f"({TIER_SOURCE})"),
            ("base_points", str(trace.base_points), "(from tier, not text)"),
            ("no_redundancy", f"+{trace.no_redundancy}", f"({trace.no_redundancy_reason})"),
            *[("default", d, "") for d in trace.defaults_applied],
            ("urgency_tally", str(trace.urgency_tally), f"({trace.base_points} + {trace.no_redundancy})"),
        ]
    rows.append(("safety_level", str(trace.safety_level), f"({trace.safety_reason})"))
    rows += [(f"span:{s.field}", f'"{s.text}"', "verified [3]") for s in trace.evidence_spans]
    rows += [("flag", f, "") for f in trace.flags]
    rows += [
        ("original_timestamp", trace.original_timestamp.isoformat(), "FIFO input, never overwritten"),
        ("position", f"{trace.position} of {trace.queue_length}", ""),
        ("decided_by", trace.decided_by, ""),
        (
            "distance",
            "distance unknown" if trace.distance_km is None else f"{trace.distance_km:g} km",
            "not in sort_key",
        ),
    ]
    rows += [("logistics", n, "display only") for n in trace.logistics_notes]
    return _table(rows)


def render_review_entry(job: EnrichedJob) -> str:
    return _table(
        [
            ("job_id", job.request_id, ""),
            ("fault", _verified_fault_text(job) or "(no verified fault text)", ""),
            ("community", job.community, ""),
            ("original_timestamp", job.original_report_timestamp.isoformat(), "FIFO input, never overwritten"),
            ("review_band", REVIEW_BAND_REASON, ""),
        ]
    )
