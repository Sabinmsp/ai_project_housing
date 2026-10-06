"""Escalation loop: a tenant re-contacts about an existing request.

Matched by exact request_id (never fuzzy), the follow-up is appended in
Stage 1, and the whole history re-enters Stage 2 for a fresh read. Stages
3 to 6 then run as normal, so the tally can change, but
original_report_timestamp is untouched: escalation never resets queue
fairness.

A compound report's child job id resolves to its parent report. The parent is
re-read and only the escalated child is updated, matched by taxonomy entry
(never by position, since the model may list faults in any order).
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from .extraction import LLMClient, extract
from .intake import ReportRepository
from .models import ChildJob, ExtractionResult, ExtractionStatus, Report

UNMATCHED = "Escalation could not be matched to this job automatically — review the re-extracted report."


def rematch_child(child: ChildJob, siblings: Sequence[ChildJob], result: ExtractionResult) -> ChildJob:
    """The escalated child after its parent was re-read: new facts only on one unambiguous match.

    siblings is every child of the parent, including this one. The child takes the
    re-extracted fault only when the fault count is unchanged and exactly one fault shares
    one of its (non-empty) taxonomy entries. Otherwise it keeps its facts and is flagged.
    Existing flags are always carried forward.
    """
    faults = result.extraction.faults if result.status is ExtractionStatus.OK and result.extraction else ()
    flags = list(child.flags)
    known = {name for s in siblings for name in s.facts.taxonomy_match}
    for f in faults:
        if not known.intersection(f.taxonomy_match):
            flags.append(f'New fault in escalation: "{f.fault_description}" matches no existing job '
                         "from this report — review the re-extracted report.")
    mine = set(child.facts.taxonomy_match)
    hits = [f for f in faults if mine.intersection(f.taxonomy_match)]
    # An empty taxonomy has no hits, so it is never matched.
    if len(faults) == len(siblings) and len(hits) == 1:
        facts = hits[0]
    else:
        # Never guess: a wrong match would move another fault's hazard onto this job.
        facts = child.facts
        flags.append(UNMATCHED)
    return ChildJob.model_validate({"job_id": child.job_id, "parent_report_id": child.parent_report_id,
                                    "facts": facts, "flags": tuple(flags)})


def escalate(repo: ReportRepository, request_id: str, text: str,
             received_at: datetime, client: LLMClient) -> tuple[Report, ExtractionResult]:
    """Append the follow-up and re-extract the report.

    request_id is a report id or a compound report's child job id; a child id re-reads its
    parent and updates only that child in the repository. Raises
    intake.UnknownRequestError if request_id does not match exactly.
    """
    child = repo.get_child(request_id)
    if child is None:
        report = repo.append_followup(request_id, text, received_at)
        return report, extract(report, client)
    report = repo.append_followup(child.parent_report_id, text, received_at)
    result = extract(report, client)
    repo.save_child(rematch_child(child, repo.children(child.parent_report_id), result))
    return report, result
