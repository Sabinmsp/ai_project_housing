"""Stage 6: Ranking + why-trace.

A pure function: no I/O, no model. The hierarchy is enforced by tuple
comparison, not by team discipline:

    sort_key = (safety_flag, urgency_tally, -original_timestamp)

Tuple comparison exhausts position 0 before reading position 1, so safety
cannot be outweighed. Logistics fields have no position in the tuple at all.
"""
from __future__ import annotations

import json
from typing import Optional

from .extraction import LLMClient
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
            fault_description=j.fault_description,
            original_report_timestamp=j.original_report_timestamp,
        )
        for j in sorted(review, key=lambda j: j.original_report_timestamp)
    ]

    traces = [build_trace(j, pos, len(ranked)) for pos, j in enumerate(ranked, start=1)]
    return RankingResult(review_band=review_band, ranked=ranked, traces=traces)


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

    if job.no_redundancy:
        nr_reason = "no working alternative named in the report"
    else:
        nr_reason = "report names a working alternative"

    safety_reason = {
        "active": "active hazard described in the report",
        "conditional": "conditional hazard: elevated, check-in window applies",
        "none": "no hazard mechanism described",
    }[job.safety_level]

    return ReasoningTrace(
        request_id=job.request_id,
        fault_description=job.fault_description,
        tier=job.tier,
        tier_source=TIER_SOURCE,
        base_points=job.base_points,
        no_redundancy=job.no_redundancy,
        no_redundancy_reason=nr_reason,
        urgency_tally=job.urgency_tally,
        safety_flag=job.safety_flag,
        safety_reason=safety_reason,
        evidence_spans=[s for s in job.spans if s.verified],
        flags=list(job.flags),
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
    rows = [
        ("request_id", trace.request_id, ""),
        ("tier", trace.tier, f"({trace.tier_source})"),
        ("base_points", str(trace.base_points), "(from tier, not text)"),
        ("no_redundancy", f"+{trace.no_redundancy}", f"({trace.no_redundancy_reason})"),
        ("urgency_tally", str(trace.urgency_tally),
         f"({trace.base_points} + {trace.no_redundancy})"),
        ("safety_flag", str(trace.safety_flag).lower(), f"({trace.safety_reason})"),
    ]
    for s in trace.evidence_spans:
        rows.append((f"span:{s.field}", f"\"{s.text}\"", "verified [3]"))
    for f in trace.flags:
        rows.append(("flag", f, ""))
    rows.append(("sort_key", str(trace.sort_key), ""))
    rows.append(("position", f"{trace.position} of {trace.queue_length}", ""))
    if trace.distance_cost_km is not None:
        rows.append(("distance_cost", f"{trace.distance_cost_km:g} km", "not in sort_key"))
    for n in trace.logistics_notes:
        rows.append(("logistics", n, "display only"))
    width = max(len(r[0]) for r in rows)
    return "\n".join(f"{k.ljust(width)}  {v}  {note}".rstrip() for k, v, note in rows)


COORDINATOR_PROSE_PROMPT = """Rewrite this reasoning trace as 2 to 4 plain \
sentences for a housing maintenance coordinator. Use ONLY facts present in \
the trace. Do not add reasons, do not speculate, do not change any number. \
Return JSON: {"prose": "..."}"""


def render_coordinator_prose(trace: ReasoningTrace, client: Optional[LLMClient]) -> str:
    """Optional LLM renderer. Receives the trace and nothing else. Falls back
    to the deterministic view if no client or if the output drops a number."""
    if client is None:
        return render_coordinator(trace)
    schema = {"type": "object", "properties": {"prose": {"type": "string"}},
              "required": ["prose"], "additionalProperties": False}
    try:
        raw = client.complete_json(COORDINATOR_PROSE_PROMPT,
                                   trace.model_dump_json(), schema)
        prose = json.loads(raw)["prose"]
    except Exception:
        return render_coordinator(trace)
    # Guard: the prose must at least carry the tally and position unchanged.
    if str(trace.urgency_tally) not in prose or str(trace.position) not in prose:
        return render_coordinator(trace)
    return prose
