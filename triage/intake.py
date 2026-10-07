"""Intake: the first stage. Wraps raw text in a Report and stores it.

Pipeline: intake -> extraction -> verification -> evaluation -> ranking -> explain.
Input: raw text plus who, where and when. Output: a Report with its request_id and
original_report_timestamp stamped. No interpretation happens here. Storage sits behind
ReportRepository, so ranking never touches SQL.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional, Protocol

from .models import ChildJob, Report, SourceTag


def new_request_id() -> str:
    """A new opaque job ID: "R-" plus 8 hex digits.

    The only ID source (invariant 7). It never encodes region, so the job_id tie-break
    in ranking can't favour one place over another.
    """
    return f"R-{uuid.uuid4().hex[:8].upper()}"


def create_report(
    *,
    tenant_id: str,
    raw_text: str,
    source_tag: SourceTag | str,
    community: str,
    original_report_timestamp: datetime,
    request_id: Optional[str] = None,
    region: Optional[str] = None,
    source_file: Optional[str] = None,
    source_item: Optional[int] = None,
    timestamp_source: Optional[str] = None,
) -> Report:
    """Build a Report.

    original_report_timestamp is the time the TENANT reported the fault (for a
    phone call, when they rang), not when the record was typed in. It is ranking's
    FIFO input and is never overwritten after this point (invariant 6).
    """
    return Report(
        request_id=request_id or new_request_id(),
        tenant_id=tenant_id,
        raw_text=raw_text,
        source_tag=SourceTag(source_tag),
        community=community,
        original_report_timestamp=original_report_timestamp,
        region=region,
        source_file=source_file,
        source_item=source_item,
        timestamp_source=timestamp_source,
    )


class ReportRepository(Protocol):
    """Report storage as the pipeline sees it; SQLiteReportRepository is the one implementation."""

    def save(self, report: Report) -> None: ...
    def get(self, request_id: str) -> Optional[Report]: ...
    def append_followup(self, request_id: str, text: str, received_at: datetime) -> Report: ...
    def all(self) -> list[Report]: ...
    def save_child(self, job: ChildJob) -> None: ...
    def get_child(self, job_id: str) -> Optional[ChildJob]: ...
    def children(self, parent_report_id: str) -> list[ChildJob]: ...


class DuplicateRequestError(Exception):
    """A report with this request_id is already stored."""


class UnknownRequestError(Exception):
    """No stored report has this request_id (matched exactly, never fuzzy)."""


class SQLiteReportRepository:
    """SQLite-backed store. Use ':memory:' for tests and the demo."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                request_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                source_tag TEXT NOT NULL,
                community TEXT NOT NULL,
                original_report_timestamp TEXT NOT NULL,
                region TEXT,
                source_file TEXT,
                source_item INTEGER,
                timestamp_source TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS followups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL REFERENCES reports(request_id),
                text TEXT NOT NULL,
                received_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS child_jobs (
                job_id TEXT PRIMARY KEY,
                parent_report_id TEXT NOT NULL REFERENCES reports(request_id),
                facts TEXT NOT NULL,
                flags TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def save(self, report: Report) -> None:
        try:
            self._conn.execute(
                "INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report.request_id,
                    report.tenant_id,
                    report.raw_text,
                    report.source_tag.value,
                    report.community,
                    report.original_report_timestamp.isoformat(),
                    report.region,
                    report.source_file,
                    report.source_item,
                    report.timestamp_source,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as e:
            raise DuplicateRequestError(report.request_id) from e

    def _row_to_report(self, row: tuple, followups: list[str]) -> Report:
        raw = row[2] if not followups else row[2] + "\n" + "\n".join(followups)
        return Report(
            request_id=row[0],
            tenant_id=row[1],
            raw_text=raw,
            source_tag=SourceTag(row[3]),
            community=row[4],
            original_report_timestamp=datetime.fromisoformat(row[5]),
            region=row[6],
            source_file=row[7],
            source_item=row[8],
            timestamp_source=row[9],
        )

    def _followups(self, request_id: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT text FROM followups WHERE request_id = ? ORDER BY id", (request_id,)
        )
        return [r[0] for r in cur.fetchall()]

    def get(self, request_id: str) -> Optional[Report]:
        row = self._conn.execute(
            "SELECT * FROM reports WHERE request_id = ?", (request_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_report(row, self._followups(request_id))

    def append_followup(self, request_id: str, text: str, received_at: datetime) -> Report:
        """Append a tenant's follow-up and return the updated report (used by escalation.escalate).

        Matched by exact request_id only, never fuzzy. The follow-up text is
        appended so extraction re-reads the whole history, while
        original_report_timestamp is untouched: escalation never resets
        queue fairness (invariant 6).

        Raises:
            UnknownRequestError: no report has this request_id.
        """
        if self.get(request_id) is None:
            raise UnknownRequestError(request_id)
        self._conn.execute(
            "INSERT INTO followups (request_id, text, received_at) VALUES (?, ?, ?)",
            (request_id, text, received_at.isoformat()),
        )
        self._conn.commit()
        report = self.get(request_id)
        assert report is not None
        return report

    def all(self) -> list[Report]:
        rows = self._conn.execute("SELECT * FROM reports").fetchall()
        return [self._row_to_report(r, self._followups(r[0])) for r in rows]

    def save_child(self, job: ChildJob) -> None:
        """Insert or update a compound report's child job.

        Raises:
            UnknownRequestError: the parent report is not stored.
        """
        if self.get(job.parent_report_id) is None:
            raise UnknownRequestError(job.parent_report_id)
        # Upsert, not INSERT OR REPLACE: a replace re-inserts the row and changes children() order.
        self._conn.execute(
            "INSERT INTO child_jobs VALUES (?, ?, ?, ?) ON CONFLICT(job_id) DO UPDATE SET "
            "parent_report_id = excluded.parent_report_id, facts = excluded.facts, flags = excluded.flags",
            (job.job_id, job.parent_report_id, job.facts.model_dump_json(), json.dumps(job.flags)),
        )
        self._conn.commit()

    def _row_to_child(self, row: tuple) -> ChildJob:
        return ChildJob.model_validate({"job_id": row[0], "parent_report_id": row[1],
                                        "facts": json.loads(row[2]), "flags": json.loads(row[3])})

    def get_child(self, job_id: str) -> Optional[ChildJob]:
        row = self._conn.execute("SELECT * FROM child_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else self._row_to_child(row)

    def children(self, parent_report_id: str) -> list[ChildJob]:
        rows = self._conn.execute(
            "SELECT * FROM child_jobs WHERE parent_report_id = ? ORDER BY rowid", (parent_report_id,)
        ).fetchall()
        return [self._row_to_child(r) for r in rows]


def utc_now() -> datetime:
    """The current time in UTC, timezone-aware."""
    return datetime.now(timezone.utc)
