"""Escalation loop: a tenant re-contacts about an existing request.

Matched by exact request_id (never fuzzy), the follow-up is appended in
Stage 1, and the whole history re-enters Stage 2 for a fresh read. Stages
3 to 6 then run as normal, so the tally can change, but
original_report_timestamp is untouched: escalation never resets queue
fairness.
"""
from __future__ import annotations

from datetime import datetime

from .extraction import LLMClient, extract
from .intake import ReportRepository
from .models import ExtractionResult, Report


def escalate(repo: ReportRepository, request_id: str, text: str,
             received_at: datetime, client: LLMClient) -> tuple[Report, ExtractionResult]:
    """Raises intake.UnknownRequestError if request_id does not match exactly."""
    report = repo.append_followup(request_id, text, received_at)
    return report, extract(report, client)
