"""Stage 6: Ranking + why-trace.

A pure function: no I/O, no model. Only Stage 2 calls a language model;
this module imports nothing that could. The hierarchy is enforced by tuple
comparison, not by team discipline:

    sort_key = (safety_flag, urgency_tally, -original_timestamp)

Tuple comparison exhausts position 0 before reading position 1, so safety
cannot be outweighed. Logistics fields have no position in the tuple at all.
"""
from __future__ import annotations

from typing import Optional

from .models import EnrichedJob, RankingResult, ReasoningTrace, ReviewBandEntry

TIER_SOURCE = "NT Government published fault list"


def sort_key(job: EnrichedJob) -> tuple[bool, int, int]:
    """The only three inputs to order. Nothing else on the job is read."""
    assert job.urgency_tally is not None, "review-band jobs have no sort key"
    return (
        job.safety_flag,
        job.urgency_tally,
        -int(job.original_report_timestamp.timestamp()),
    )


def rank(jobs: list[EnrichedJob]) -> RankingResult:
    review = [j for j in jobs if j.in_review_band]
    scored = [j for j in jobs if not j.in_review_band]

    ranked = sorted(scored, key=sort_key, reverse=True)

    review_band = [
        ReviewBandEntry(
            request_id=j.request_id,
            community=j.community,
            fault_description=_verified_fault_text(j),
            original_report_timestamp=j.original_report_timestamp,
        )
        for j in sorted(review, key=lambda j: j.original_report_timestamp)
    ]

    traces = [build_trace(j, pos, len(ranked)) for pos, j in enumerate(ranked, start=1)]
    return RankingResult(review_band=review_band, ranked=ranked, traces=traces)


def _verified_fault_text(job: EnrichedJob) -> Optional[str]:
    """The fault in the tenant's own words, as verified by Stage 3.

    EnrichedJob.fault_description is the model's wording and may be stitched
    together or paraphrased. Panel D: no value in the trace originates inside
    the model, so the trace carries only a verified span.
    """
    return next((s.text for s in job.spans
                 if s.verified and s.field == "fault_description"), None)


def _logistics_notes(job: EnrichedJob) -> list[str]:
    notes: list[str] = []
    if job.capacity_block_flag:
        notes.append("No qualified trade available today; rank kept"
                     + (f", next actionable: {job.next_actionable}" if job.next_actionable else ""))
    if job.shared_route_opportunities:
        notes.append("Shared route possible with: " + ", ".join(job.shared_route_opportunities))
    if job.starvation_line:
        notes.append(f"{job.community}: {job.starvation_line}")
    return notes


def build_trace(job: EnrichedJob, position: int, queue_length: int) -> ReasoningTrace:
    assert job.tier is not None and job.base_points is not None and job.urgency_tally is not None

    # Stage 4b severity default: +1 unless the text names a genuine alternative.
    if job.no_redundancy:
        nr_reason = "no working alternative named in the report"
        defaults = ["no-redundancy default applied: +1 (no alternative named)"]
    else:
        nr_reason = "report names a working alternative"
        defaults = ["no-redundancy default removed: +0 (alternative named)"]

    safety_reason = {
        "active": "mechanism_type=active: full override",
        "conditional": "mechanism_type=conditional: elevated only, check-in window applies",
        "none": "no hazard mechanism described",
    }[job.safety_level]

    return ReasoningTrace(
        request_id=job.request_id,
        fault_description=_verified_fault_text(job),
        taxonomy_match=list(job.taxonomy_match),
        tier=job.tier,
        tier_source=TIER_SOURCE,
        base_points=job.base_points,
        no_redundancy=job.no_redundancy,
        no_redundancy_reason=nr_reason,
        defaults_applied=defaults,
        urgency_tally=job.urgency_tally,
        safety_flag=job.safety_flag,
        safety_reason=safety_reason,
        evidence_spans=[s for s in job.spans if s.verified],
        flags=list(job.flags),
        original_report_timestamp=job.original_report_timestamp,
        sort_key=sort_key(job),
        position=position,
        queue_length=queue_length,
        distance_cost_km=job.distance_cost_km,
        logistics_notes=_logistics_notes(job),
    )


# ---------------------------------------------------------------------------
# Renderers: two audiences, one trace
# ---------------------------------------------------------------------------

def render_tenant_sms(trace: ReasoningTrace) -> str:
    """Pure template. Every word is either fixed text or a trace field, so it
    provably cannot invent a reason. Distance is never shown to the tenant."""
    fault = trace.fault_description or "your repair"
    lines = [f"Housing repair {trace.request_id}: we have your report about \"{fault}\"."]
    if trace.safety_flag:
        lines.append("It is marked as a safety job, so it sits above all non-safety jobs.")
    lines.append(f"It is classed {trace.tier} on the {TIER_SOURCE}.")
    if trace.no_redundancy:
        lines.append("It is higher priority because you have no other working one.")
    lines.append(f"Queue position: {trace.position} of {trace.queue_length}.")
    lines.append(f"Reply with {trace.request_id} if things get worse.")
    return " ".join(lines)


def render_coordinator(trace: ReasoningTrace) -> str:
    """Deterministic coordinator view, laid out like Panel D."""
    rows = [("request_id", trace.request_id, "")]
    for m in trace.taxonomy_match:
        rows.append(("taxonomy_match", m, "(NT fault list name)"))
    rows += [
        ("tier", trace.tier, f"({trace.tier_source})"),
        ("base_points", str(trace.base_points), "(from tier, not text)"),
        ("no_redundancy", f"+{trace.no_redundancy}", f"({trace.no_redundancy_reason})"),
        *[("default", d, "") for d in trace.defaults_applied],
        ("urgency_tally", str(trace.urgency_tally),
         f"({trace.base_points} + {trace.no_redundancy})"),
        ("safety_flag", str(trace.safety_flag).lower(), f"({trace.safety_reason})"),
    ]
    for s in trace.evidence_spans:
        rows.append((f"span:{s.field}", f"\"{s.text}\"", "verified [3]"))
    for f in trace.flags:
        rows.append(("flag", f, ""))
    rows.append(("original_timestamp", trace.original_report_timestamp.isoformat(),
                 "FIFO input, never overwritten"))
    rows.append(("sort_key", str(trace.sort_key), ""))
    rows.append(("position", f"{trace.position} of {trace.queue_length}", ""))
    if trace.distance_cost_km is not None:
        rows.append(("distance_cost", f"{trace.distance_cost_km:g} km", "not in sort_key"))
    for n in trace.logistics_notes:
        rows.append(("logistics", n, "display only"))
    width = max(len(r[0]) for r in rows)
    return "\n".join(f"{k.ljust(width)}  {v}  {note}".rstrip() for k, v, note in rows)
