"""Explain: the why-trace for each ranked job, and its renderers.

Pipeline: intake -> extraction -> verification -> evaluation -> ranking -> explain.
Input: RankResult plus each EnrichedJob. Output: a ReasoningTrace per job, rendered as
the coordinator view, the tenant SMS or the tenant's WHY answer; review-band jobs get
their own entry.
Renderers are fixed templates over trace fields, so they can't invent a reason.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from triage.adapter import to_rank_input
from triage.models import (
    EnrichedJob,
    ExtractionResult,
    ExtractionStatus,
    RankedJob,
    RankResult,
    Reason,
    ReRead,
    SecondReading,
    VerifiedSpan,
)
from triage.ranking import REVIEW_BAND_REASON
from triage.tiers import TIER_TABLE


class ReasoningTrace(BaseModel):
    """Structured explanation. Both renderers read this and nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    fault_description: str | None
    taxonomy_match: tuple[str, ...]
    # None: untiered safety job, ranked but awaiting a tier call (invariant 5).
    tier: Literal["dangerous", "standard"] | None
    tier_entry: str | None
    base_points: int | None
    severity_bump: int | None
    tally_reasons: tuple[Reason, ...]
    urgency_tally: int | None
    safety_level: int
    safety_reason: Reason
    evidence_spans: tuple[VerifiedSpan, ...]
    flags: tuple[Reason, ...]
    original_timestamp: datetime
    position: int
    queue_length: int
    decided_by: str
    distance_km: float | None
    nearest_office: str | None
    # Coordinator-only: tenant renderers never read it.
    second_reader: SecondReading | None
    reread: ReRead | None
    logistics_notes: tuple[str, ...]
    # Coordinator-only: the tenant sent WHY <ref> for this job.
    tenant_asked_why: bool


def _verified_fault_text(job: EnrichedJob) -> str | None:
    # No trace text comes from the model's own wording: fault_description may be paraphrased,
    # so only a verified span (the tenant's words) is shown.
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


def build_trace(entry: RankedJob, job: EnrichedJob, queue_length: int, asked_why: bool) -> ReasoningTrace:
    """The trace for one ranked position. Only verified spans are carried."""
    # RankedJob doesn't carry safety_level; the adapter is the one place that maps it.
    safety_level = to_rank_input(job).safety_level
    return ReasoningTrace(
        job_id=entry.job_id,
        fault_description=_verified_fault_text(job),
        taxonomy_match=tuple(job.taxonomy_match),
        tier=job.tier,
        tier_entry=job.tier_entry,
        base_points=job.base_points,
        severity_bump=job.severity_bump,
        tally_reasons=job.tally_reasons,
        urgency_tally=job.urgency_tally,
        safety_level=safety_level,
        safety_reason=job.safety_reason,
        evidence_spans=tuple(s for s in job.spans if s.verified),
        # Evaluation and logistics flags first, then ranking's own (e.g. the no-tier flag).
        flags=(*job.flags, *entry.flags),
        original_timestamp=job.original_report_timestamp,
        position=entry.position,
        queue_length=queue_length,
        decided_by=entry.decided_by,
        distance_km=job.distance_cost_km,
        nearest_office=job.nearest_office,
        second_reader=job.second_reader,
        reread=job.reread,
        logistics_notes=_logistics_notes(job),
        tenant_asked_why=asked_why,
    )


def build_traces(
    result: RankResult, jobs: dict[str, EnrichedJob], asked_why: frozenset[str] = frozenset()
) -> list[ReasoningTrace]:
    """One trace per ranked job, in rank()'s order; asked_why holds the refs a tenant sent WHY for.

    Raises:
        KeyError: a ranked job_id is missing from jobs, which is a bug upstream.
    """
    return [
        build_trace(entry, jobs[entry.job_id], len(result.ranked), entry.job_id in asked_why)
        for entry in result.ranked
    ]


SmsPath = Literal["safety_active", "safety_conditional", "urgent", "routine", "review", "flagged", "out_of_scope"]

_KEEP_UPDATED = "We'll keep you updated."

# Master §5.2: fixed wording only. No tier label, source, verdict on the report, comparison,
# count of other jobs, timeframe or ranking mechanics. "routine", never "general".
SMS_PATH_SENTENCES: dict[SmsPath, str] = {
    "safety_active": "It is marked as a safety job and is being handled as a priority. " + _KEEP_UPDATED,
    "safety_conditional": "It is marked for a safety check, which a coordinator will do as a priority. "
    + _KEEP_UPDATED,
    "urgent": "It's being treated as an urgent repair under NT rules. " + _KEEP_UPDATED,
    "routine": "It's being treated as a routine repair. " + _KEEP_UPDATED,
    "review": "It has been sent to a coordinator for a direct review before scheduling. "
    "We'll update you once that's done.",
    "flagged": "A coordinator will read your request directly and update you.",
    "out_of_scope": "A coordinator will contact you to talk through what needs fixing.",
}

# Safety advice: exact text supplied by Prabin, never generated or paraphrased. Keyed on the
# taxonomy match only, never on hazard_status: a live probe showed casual gas reports
# sometimes come back without a hazard, and the advice must still reach them.
# Source: https://worksafe.nt.gov.au/safety-and-prevention/gas-safety
GAS_ADVICE = (
    "If you smell gas: leave the building or area and call Fire and Emergency Services on 000. "
    "If it is safe to do so, turn off the gas at the cylinder or meter. Do not enter the gas affected area."
)
# Source: https://www.powerwater.com.au/customers/safety-and-emergencies
ELECTRICAL_ADVICE = "Stay away from the exposed wiring. In an emergency call 000."
SAFETY_ADVICE: dict[str, str] = {"gas leak": GAS_ADVICE, "exposed electrical wires": ELECTRICAL_ADVICE}

SMS_FAULT_MAX = 60  # keeps the whole message near one SMS segment


def _plain_fault(text: str | None) -> str | None:
    """The tenant's own fault words, whitespace collapsed and cut to SMS_FAULT_MAX; None if blank."""
    words = " ".join((text or "").split())
    if not words:
        return None
    return words if len(words) <= SMS_FAULT_MAX else words[: SMS_FAULT_MAX - 1].rstrip() + "…"


def _ranked_path(safety_level: int, tier: str | None) -> SmsPath:
    # Category word follows the tier, so every dangerous entry (wires, sewage, drains) is urgent.
    if safety_level == 2:
        return "safety_active"
    if safety_level == 1:
        return "safety_conditional"
    if tier == "dangerous":
        return "urgent"
    if tier == "standard":
        return "routine"
    return "review"


def _advice(taxonomy_match: Sequence[str]) -> tuple[str, ...]:
    """The fixed advice lines for a job's taxonomy matches, in SAFETY_ADVICE order."""
    return tuple(text for entry, text in SAFETY_ADVICE.items() if entry in taxonomy_match)


def _reply_line(ref: str) -> str:
    # §5.2: escalation offer on every message, with no cutoff.
    return f"Reply HELP with {ref} if anything changes or gets worse."


def _sms(ref: str, fault: str | None, path: SmsPath, advice: tuple[str, ...]) -> str:
    words = _plain_fault(fault)
    # No quoted placeholder: without the tenant's own fault words, the message names none.
    opener = f'Housing repair {ref}: we have your report about "{words}".' if words else f"Housing repair {ref}: we have your message."
    # Advice first, so it is the first thing read in an emergency.
    return " ".join((*advice, opener, SMS_PATH_SENTENCES[path], _reply_line(ref)))


def _named(job: EnrichedJob) -> str:
    words = _plain_fault(_verified_fault_text(job))
    return f'"{words}" (ref {job.request_id})' if words else f"a repair (ref {job.request_id})"


def tenant_sms(job: EnrichedJob | ExtractionResult) -> str:
    """The tenant's SMS for one job: its ref, its fault, its path sentence and the reply line.

    An ExtractionResult covers reports that never became a job (flagged for human, out of scope).

    Raises:
        ValueError: an OK ExtractionResult, whose SMS must come from its EnrichedJob.
    """
    # Invariant 10, master §5.2: reads this job only, so no other job, position or decided_by.
    if isinstance(job, ExtractionResult):
        if job.status is ExtractionStatus.OK:
            raise ValueError(f"{job.request_id}: extracted report; build the SMS from its EnrichedJob")
        path: SmsPath = "flagged" if job.status is ExtractionStatus.FLAGGED_FOR_HUMAN else "out_of_scope"
        return _sms(job.request_id, None, path, ())
    path = _ranked_path(to_rank_input(job).safety_level, job.tier)
    return _sms(job.request_id, _verified_fault_text(job), path, _advice(job.taxonomy_match))


def tenant_sms_report(jobs: Sequence[EnrichedJob]) -> str:
    """One SMS for a whole report: each job's fault and ref, and the refs to reply with.

    Raises:
        ValueError: no jobs, or jobs from more than one report.
    """
    parents = {j.parent_report_id for j in jobs}
    if len(parents) != 1:
        raise ValueError(f"tenant_sms_report needs the jobs of exactly one report, got {sorted(parents)}")
    if len(jobs) == 1:
        return tenant_sms(jobs[0])
    # §5.2 C5: N repairs with each ref; N counts this tenant's own repairs only.
    # "received", not "logged": FR7e forbids "logged" (master §10, B3).
    named = [_named(j) for j in jobs]
    # Advice first, each against the fault it is for, never against the whole report.
    advice = [f"For {name}: {text}" for name, j in zip(named, jobs) for text in _advice(j.taxonomy_match)]
    refs = ", ".join(j.request_id for j in jobs)
    return " ".join((
        *advice,
        f"Housing repairs: we've received {len(jobs)} repairs from your report: {'; '.join(named)}.",
        "We'll update you on each one separately.",
        f"Reply HELP with the ref of any repair that changes or gets worse: {refs}.",
    ))


def render_tenant_sms(trace: ReasoningTrace) -> str:
    """The tenant's SMS for a ranked job, from its trace; same wording as tenant_sms."""
    # Invariant 10: position, queue_length, decided_by, distance and flags are never read.
    path = _ranked_path(trace.safety_level, trace.tier)
    return _sms(trace.job_id, trace.fault_description, path, _advice(trace.taxonomy_match))


# Source: FS17 (https://dhlgcd.nt.gov.au/media/documents/fact-sheets/repairs-and-maintenance-fs17.pdf)
UNKNOWN_REF_REPLY = (
    "We couldn't find that reference. Please check the number or call the maintenance call centre on 1800 104 076."
)

_WHY_STATUS: dict[SmsPath, str] = {
    "safety_active": "is booked as a safety repair.",
    "safety_conditional": "is booked as a safety repair.",
    "urgent": "is booked as an urgent repair.",
    "routine": "is booked as a routine repair.",
    "review": "is with a coordinator, who is deciding what kind of repair it is.",
    "flagged": "is with a coordinator, who is deciding what kind of repair it is.",
    "out_of_scope": "is with a coordinator, who is deciding what kind of repair it is.",
}
# Same lines for every job: no fault-specific guessing about impact, no timeframe.
WHY_IMPACT_LINE = "We know this is hard to live with. A coordinator can see how long it has been waiting."
_NT_TIME = ZoneInfo("Australia/Darwin")


def _why_help_line(ref: str) -> str:
    return (
        "If anyone in the house is unwell, elderly or very young, or this is affecting anyone's health or "
        f"safety, reply HELP {ref} and a coordinator will look at it again."
    )


def _why_subject(ref: str, fault: str | None) -> str:
    words = _plain_fault(fault)
    return f'Your "{words}" repair ({ref})' if words else f"Your repair ({ref})"


def tenant_why(job: EnrichedJob | ExtractionResult) -> str:
    """The answer to "WHY <ref>": this job's fault, ref, category word and received date.

    Raises:
        ValueError: an OK ExtractionResult, whose answer must come from its EnrichedJob.
    """
    # Invariant 10: reads this job only, so no position, other job, count or decided_by.
    # Never explains how jobs are ranked and never states a timeframe.
    if isinstance(job, ExtractionResult):
        if job.status is ExtractionStatus.OK:
            raise ValueError(f"{job.request_id}: extracted report; build the answer from its EnrichedJob")
        path: SmsPath = "flagged" if job.status is ExtractionStatus.FLAGGED_FOR_HUMAN else "out_of_scope"
        # No job, so no received date to state.
        ref = job.request_id
        return " ".join((f"{_why_subject(ref, None)} {_WHY_STATUS[path]}", WHY_IMPACT_LINE, _why_help_line(ref)))
    ref = job.request_id
    path = _ranked_path(to_rank_input(job).safety_level, job.tier)
    received = job.original_report_timestamp.astimezone(_NT_TIME)
    return " ".join((
        f"{_why_subject(ref, _verified_fault_text(job))} {_WHY_STATUS[path]}",
        f"Yours was received on {received.day} {received:%B %Y}.",
        WHY_IMPACT_LINE,
        _why_help_line(ref),
    ))


def _source_text(source: str, tier: str) -> str:
    if source == "nt.gov.au":
        return "nt.gov.au — " + ("on the repaired-first list" if tier == "dangerous" else "general repairs list")
    if source.startswith("RTA "):
        return f"NT Residential Tenancies Act {source.removeprefix('RTA ')} — emergency repair"
    raise ValueError(f"unknown tier source: {source!r}")


def _table(rows: list[tuple[str, str, str]]) -> str:
    width = max(len(key) for key, _, _ in rows)
    return "\n".join(f"{key.ljust(width)}  {value}  {note}".rstrip() for key, value, note in rows)


def render_coordinator(trace: ReasoningTrace) -> str:
    """The coordinator's full why-trace as an aligned table.

    Raises:
        ValueError: a tiered trace has no tier_entry whose sources it can cite.
    """
    rows = [("job_id", trace.job_id, "")]
    rows += [("taxonomy_match", m, "(reference list name)") for m in trace.taxonomy_match]
    if trace.tier is None:
        rows.append(("tier", "untiered", "(needs a coordinator tier call)"))
    else:
        if trace.tier_entry is None:
            raise ValueError(f"{trace.job_id}: tiered trace has no tier_entry to cite")
        sources = "; ".join(_source_text(src, trace.tier) for src in TIER_TABLE[trace.tier_entry].sources)
        rows += [
            ("tier", trace.tier, f"({sources})"),
            ("base_points", str(trace.base_points), "(from tier, not text)"),
            ("severity_bump", f"+{trace.severity_bump}", ""),
            *[("tally_reason", r, "") for r in trace.tally_reasons],
            ("urgency_tally", str(trace.urgency_tally), f"({trace.base_points} + {trace.severity_bump})"),
        ]
    rows.append(("safety_level", str(trace.safety_level), f"({trace.safety_reason})"))
    rows += [(f"span:{s.field}", f'"{s.text}"', "verified") for s in trace.evidence_spans]
    rows += [("flag", f, "") for f in trace.flags]
    rows += [
        ("original_timestamp", trace.original_timestamp.isoformat(), "FIFO input, never overwritten"),
        ("position", f"{trace.position} of {trace.queue_length}", ""),
        ("decided_by", trace.decided_by, ""),
        *([("tenant_contact", "tenant asked why", "")] if trace.tenant_asked_why else []),
        _distance_row(trace.distance_km, trace.nearest_office),
        *_second_reader_rows(trace.second_reader),
        *_reread_rows(trace.reread),
    ]
    rows += [("logistics", n, "display only") for n in trace.logistics_notes]
    return _table(rows)


def _second_reader_rows(reading: SecondReading | None) -> list[tuple[str, str, str]]:
    # Flags only: shown to the coordinator, never fed back into safety, tally or rank.
    if reading is None:
        return []
    if reading.status == "ran":
        lines = reading.flags or ("agrees",)
    elif reading.status == "unavailable":
        lines = (f"unavailable ({reading.detail})",)
    else:
        lines = (reading.status,)
    return [("second_reader", line, "flag only") for line in lines]


def _reread_rows(record: ReRead | None) -> list[tuple[str, str, str]]:
    if record is None:
        return []
    second = record.read_2 or "unavailable"
    return [("re_read", f"read 1: {record.read_1}; read 2: {second}; used: {record.used}", "safer reading kept")]


def _distance_row(km: float | None, office: str | None) -> tuple[str, str, str]:
    if km is None:
        return ("distance", "unknown (community not in location table)", "not in sort_key")
    return (
        "distance",
        f"~{km:g} km to nearest NT Housing office ({office}) — straight-line; actual dispatch point not known",
        "not in sort_key",
    )


def render_review_entry(job: EnrichedJob, asked_why: bool = False) -> str:
    """The coordinator's entry for a review-band job, which has no position."""
    return _table(
        [
            *([("tenant_contact", "tenant asked why", "")] if asked_why else []),
            ("job_id", job.request_id, ""),
            ("fault", _verified_fault_text(job) or "(no verified fault text)", ""),
            ("community", job.community, ""),
            _distance_row(job.distance_cost_km, job.nearest_office),
            *_second_reader_rows(job.second_reader),
            *_reread_rows(job.reread),
            ("original_timestamp", job.original_report_timestamp.isoformat(), "FIFO input, never overwritten"),
            ("review_band", REVIEW_BAND_REASON, ""),
        ]
    )
