"""Stage 1 input: read tenant reports from files in a folder.

Each file is one report. PDF (.pdf) and plain text (.txt) use the same layout:

    Tenant ID: T-03
    Community: Maningrida
    Source: tenant_direct
    Reported: 2026-09-29 15:00
    Message:
    roof leaking in kids room, water coming through the light fitting

"Request ID:" is optional. "Reported" is when the TENANT reported the fault;
a time with no timezone is read as NT time (ACST, UTC+09:30). A file with a
missing or bad field is skipped and listed, never guessed: guessing the
timestamp would break queue fairness.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError

from .intake import create_report
from .models import Report, SourceTag

NT_TIME = timezone(timedelta(hours=9, minutes=30))
SUPPORTED_SUFFIXES = (".pdf", ".txt")

_REQUIRED = [
    ("Tenant ID", "tenant_id"),
    ("Community", "community"),
    ("Source", "source_tag"),
    ("Reported", "original_report_timestamp"),
]
_HEADERS = {label.lower(): field for label, field in _REQUIRED}
_HEADERS["request id"] = "request_id"

_SOURCE_ALIASES = {
    "officer": SourceTag.OFFICER,
    "phone": SourceTag.OFFICER,
    "tenant_direct": SourceTag.TENANT_DIRECT,
    "tenant direct": SourceTag.TENANT_DIRECT,
    "web": SourceTag.TENANT_DIRECT,
    "form": SourceTag.TENANT_DIRECT,
    "email": SourceTag.TENANT_DIRECT,
}

# Tried after ISO 8601 (2026-09-29 15:00). Australian day-first order.
_DATE_FORMATS = ("%d/%m/%Y %H:%M", "%d/%m/%Y %I:%M %p", "%d/%m/%Y %I:%M%p")


class ReportFileError(ValueError):
    pass


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def _parse_source(value: str) -> SourceTag:
    try:
        return _SOURCE_ALIASES[value.strip().lower()]
    except KeyError:
        raise ReportFileError(
            f"Source must be officer or tenant_direct, got {value!r}") from None


def _parse_timestamp(value: str) -> datetime:
    value = value.strip()
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        for fmt in _DATE_FORMATS:
            try:
                ts = datetime.strptime(value.upper(), fmt)
                break
            except ValueError:
                continue
        else:
            raise ReportFileError(
                f"Reported must look like 2026-09-29 15:00 or 29/09/2026 15:00, got {value!r}"
            ) from None
    return ts if ts.tzinfo else ts.replace(tzinfo=NT_TIME)


def parse_report_text(text: str) -> Report:
    """Turn one file's text into a Report. Raises ReportFileError if unusable."""
    fields: dict[str, str] = {}
    message: list[str] = []
    in_message = False
    for line in text.splitlines():
        if in_message:
            message.append(line)
            continue
        key, sep, value = line.partition(":")
        key = key.strip().lower()
        if not sep:
            continue
        if key == "message":
            in_message = True
            message.append(value)
        elif key in _HEADERS:
            fields[_HEADERS[key]] = value.strip()

    # PDF text wraps long lines; join them back into one message.
    raw_text = " ".join(" ".join(message).split())

    missing = [label for label, field in _REQUIRED if not fields.get(field)]
    if not raw_text:
        missing.append("Message")
    if missing:
        raise ReportFileError(f"missing {', '.join(missing)}")

    try:
        return create_report(
            tenant_id=fields["tenant_id"],
            raw_text=raw_text,
            source_tag=_parse_source(fields["source_tag"]),
            community=fields["community"],
            original_report_timestamp=_parse_timestamp(fields["original_report_timestamp"]),
            request_id=fields.get("request_id") or None,
        )
    except ValidationError as e:
        raise ReportFileError(str(e)) from e


def load_reports(folder: Path) -> tuple[list[Report], list[tuple[Path, str]]]:
    """Read every .pdf and .txt file in folder, in name order.

    Returns (reports, skipped). One bad file never stops the others.
    """
    reports: list[Report] = []
    skipped: list[tuple[Path, str]] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            reports.append(parse_report_text(read_text(path)))
        except ReportFileError as e:
            skipped.append((path, str(e)))
        except Exception as e:  # unreadable or corrupt file
            skipped.append((path, f"could not read file: {e.__class__.__name__}: {e}"))
    return reports, skipped
