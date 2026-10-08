"""The six stages, wired once. Both the web app (app/server.py) and the CLI (demo.py) call this.

    Stage 1  intake        Report                      (intake.create_report, report_files, geh_form)
    Stage 2  extraction    Report -> ExtractionResult   (the only model call)
    Stage 3  verification  Report, ExtractedFacts -> VerifiedFacts
    Stage 4  evaluation    VerifiedFacts -> Evaluation
    Stage 5  logistics     Report, VerifiedFacts, Evaluation -> EnrichedJob
    Stage 6  ranking       EnrichedJob[] -> Stage6Output (RankResult + ReasoningTrace per job)

Each stage's return type is the next stage's parameter type; tests/test_pipeline.py pins that.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel, ConfigDict

from .adapter import to_rank_input
from .distances import load_coordinates, load_distances, straight_line_km
from .evaluation import Evaluation, Unverified, evaluate
from .explain import ReasoningTrace, build_traces
from .extraction import LLMClient, extract
from .intake import ReportRepository, new_request_id
from .models import ChildJob, EnrichedJob, ExtractedFacts, ExtractionResult, RankResult, Report, VerifiedSpan
from .ranking import rank
from .trades import required_trades
from .verification import claim_spans, verify_spans

_DISTANCES = load_distances()
_COORDS = load_coordinates()
_SAFETY_LEVEL_NAMES = ("none", "conditional", "active")  # index = Evaluation.safety.level


class VerifiedFacts(BaseModel):
    """Stage 3 output: the model's facts plus which quotes are really in the report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    facts: ExtractedFacts
    unverified: Unverified
    spans: tuple[VerifiedSpan, ...]


class Stage6Output(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    result: RankResult
    traces: tuple[ReasoningTrace, ...]


# ---- Stage 2 ----------------------------------------------------------------

def stage2_extract(report: Report, client: LLMClient) -> ExtractionResult:
    return extract(report, client)


# ---- Stage 3 ----------------------------------------------------------------

def stage3_verify(report: Report, facts: ExtractedFacts) -> VerifiedFacts:
    unverified = verify_spans(report.raw_text, facts)
    # Spans that back no claim (e.g. impact_status quoting "ongoing") are left out entirely.
    spans = tuple(VerifiedSpan(field=s.field, text=s.text, verified=(s.field, s.text) not in unverified)
                  for s in claim_spans(facts))
    return VerifiedFacts(facts=facts, unverified=unverified, spans=spans)


# ---- Stage 4 ----------------------------------------------------------------

def stage4_evaluate(verified: VerifiedFacts) -> Evaluation:
    return evaluate(verified.facts, verified.unverified)


# ---- Stage 5 ----------------------------------------------------------------

def stage5_enrich(report: Report, verified: VerifiedFacts, evaluation: Evaluation, job_id: str,
                  extra_flags: tuple[str, ...] = ()) -> EnrichedJob:
    """Logistics and the job record Stage 6 reads. Writes fields; reorders nothing."""
    level = _SAFETY_LEVEL_NAMES[evaluation.safety.level]
    road_km = _DISTANCES.get(report.community)
    return EnrichedJob(
        request_id=job_id, parent_report_id=report.request_id,
        community=report.community,
        # Invariant 6: every job from a report keeps the report's intake timestamp (FIFO).
        original_report_timestamp=report.original_report_timestamp,
        fault_description=verified.facts.fault_description, taxonomy_match=verified.facts.taxonomy_match,
        spans=list(verified.spans),
        tier=evaluation.tier.tier, tier_entry=evaluation.tally.winner,
        base_points=evaluation.tally.base, severity_bump=evaluation.tally.bump,
        urgency_tally=evaluation.tally.tally, tally_reasons=evaluation.tally.reasons,
        safety_flag=(level == "active"), safety_level=level, safety_reason=evaluation.safety.reason,
        flags=(*evaluation.flags, *extra_flags),
        distance_cost_km=road_km,
        # Only when no road distance is listed, and kept in its own field so it is never mistaken for one.
        distance_estimate_km=None if road_km is not None else straight_line_km("Darwin", report.community, _COORDS),
        required_trades=required_trades(evaluation.tally.winner),
    )


def enrich_fault(report: Report, facts: ExtractedFacts, job_id: Optional[str] = None,
                 extra_flags: tuple[str, ...] = ()) -> EnrichedJob:
    """Stages 3, 4 and 5 for one fault."""
    verified = stage3_verify(report, facts)
    if not facts.fault_description:
        # D4: extract() routes these out of scope, so evaluation must never see one.
        raise ValueError(f"{report.request_id}: no fault named, should not reach evaluation")
    return stage5_enrich(report, verified, stage4_evaluate(verified), job_id or report.request_id, extra_flags)


def duplicate_flags(faults: Sequence[ExtractedFacts], ids: Sequence[str]) -> list[tuple[str, ...]]:
    """Per fault, a flag for each taxonomy entry another fault in the same report also matched."""
    flags = []
    for i, facts in enumerate(faults):
        mine = []
        for name in dict.fromkeys(facts.taxonomy_match):
            sharing = [ids[j] for j, f in enumerate(faults) if name in f.taxonomy_match]
            if len(sharing) > 1:
                also = ", ".join(job_id for job_id in sharing if job_id != ids[i])
                # Flag, never merge: two entries may really be two faults, and a merge could drop one.
                mine.append(f'Possible duplicate: {len(sharing)} entries for "{name}" in this report '
                            f"(also {also}). Check before dispatching.")
        flags.append(tuple(mine))
    return flags


def build_jobs(report: Report, faults: Sequence[ExtractedFacts], ids: Optional[Sequence[str]] = None,
               carried_flags: Optional[Sequence[tuple[str, ...]]] = None) -> list[EnrichedJob]:
    """Stages 3 to 5 for every fault of one report: one job per fault, none dropped or merged.

    One fault keeps the report's id. In a compound report every job gets its own id, so none is
    mistaken for the report itself; parent_report_id links them back. Pass ids (and flags carried
    from escalation) to rebuild jobs that already exist.
    """
    if ids is None:
        ids = [report.request_id] if len(faults) == 1 else [new_request_id() for _ in faults]
    carried = carried_flags or [()] * len(faults)
    return [enrich_fault(report, facts, job_id, (*dups, *extra))
            for facts, job_id, dups, extra in zip(faults, ids, duplicate_flags(faults, ids), carried, strict=True)]


def save_children(repo: ReportRepository, faults: Sequence[ExtractedFacts], jobs: Sequence[EnrichedJob]) -> None:
    """Store a compound report's child jobs, so a tenant replying with a child's id can escalate it."""
    if len(jobs) == 1:
        return  # the single job is the report itself; escalation finds it by request_id
    for job, facts in zip(jobs, faults, strict=True):
        repo.save_child(ChildJob.model_validate({"job_id": job.request_id, "parent_report_id": job.parent_report_id,
                                                 "facts": facts, "flags": ()}))


# ---- Stage 6 ----------------------------------------------------------------

def stage6_rank(jobs: Sequence[EnrichedJob]) -> Stage6Output:
    by_id = {job.request_id: job for job in jobs}
    result = rank([to_rank_input(job) for job in jobs])
    return Stage6Output(result=result, traces=tuple(build_traces(result, by_id)))
