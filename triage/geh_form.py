"""Intake input: NT Government Employee Housing repair request form (GEHSF03).

Input: one form PDF. Output: one Report per issue row, for intake to store.
One PDF holds one property and a table of issues. Each issue becomes its own
Report, because each is its own job. The first page carrying the form title
holds the filled-in values (page 3 of GEHSF03):

  * the header (region, address, tenant) is read from pypdf's layout text,
    where each value sits beside its printed label;
  * the issues table is read from the plain text, where every filled cell is
    on its own line: item #, issue, tenant's priority, location, date
    previously reported (or NIL), method, comments.

What goes into Report.raw_text: the issue, location and comments, in the
tenant's words. What does NOT: the tenant's own Immediate/Urgent/Routine
choice (tiers come from the published fault list in evaluation, and the model
must never see priority labels), and the tenant's name, phone, email and
address (the model has no need for them).

original_report_timestamp is the date the issue was previously reported to
DIPL when the form gives one (escalation never resets queue fairness),
otherwise the time the form arrived in the folder (the file's modified time).
Report.timestamp_source records which.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .intake import create_report
from .models import Report, SourceTag

NT_TIME = timezone(timedelta(hours=9, minutes=30))
FORM_MARKER = "GEH Repairs and maintenance request form"
PRIORITY_WORDS = {"immediate", "urgent", "routine"}

_DATE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


class FormParseError(ValueError):
    """The PDF is not a GEH form, or a required part of it is missing or malformed."""


@dataclass(frozen=True)
class FormItem:
    """One row of the form's issues table, before it becomes a Report."""

    number: int
    issue: str
    location: str
    previously_reported: Optional[str]  # dd/mm/yyyy, or None for NIL
    method: Optional[str]
    comments: Optional[str]


def is_geh_form(text: str) -> bool:
    """True if the text contains the GEH form's title."""
    return FORM_MARKER in text


def _nil(value: str) -> Optional[str]:
    return None if value.strip().upper() in ("", "NIL") else value.strip()


def parse_items(lines: list[str]) -> list[FormItem]:
    """Parse the issues table: one filled cell per line, rows numbered 1, 2, 3...

    Raises:
        FormParseError: a row is out of order or incomplete, or no row is filled in.
    """
    lines = [l.strip() for l in lines if l.strip()]
    items: list[FormItem] = []
    i, n = 0, 1
    while i < len(lines):
        if lines[i] != str(n):
            raise FormParseError(f"expected item {n}, found {lines[i]!r}")
        i += 1
        issue: list[str] = []
        while i < len(lines) and lines[i].lower() not in PRIORITY_WORDS:
            issue.append(lines[i])
            i += 1
        i += 1  # the tenant's own priority choice: deliberately not kept
        location: list[str] = []
        while i < len(lines) and not (_DATE.match(lines[i]) or lines[i].upper() == "NIL"):
            location.append(lines[i])
            i += 1
        if i + 1 >= len(lines) or not issue:
            raise FormParseError(f"item {n} is incomplete")
        date, method = lines[i], lines[i + 1]
        i += 2
        comments: list[str] = []
        while i < len(lines) and lines[i] != str(n + 1):
            comments.append(lines[i])
            i += 1
        items.append(FormItem(
            number=n,
            issue=" ".join(issue),
            location=" ".join(location),
            previously_reported=_nil(date),
            method=_nil(method),
            comments=_nil(" ".join(comments)),
        ))
        n += 1
    if not items:
        raise FormParseError("no issues filled in")
    return items


def _layout_value(layout: str, label: str) -> Optional[str]:
    m = re.search(rf"^\s*{re.escape(label)}\s{{2,}}(.+?)(\s{{2,}}.*)?$", layout, re.M)
    return m.group(1).strip() if m else None


def community_from_address(address: str) -> str:
    """'Lot 42, 18 Wattlebird Court, Humpty Doo NT 0836' -> 'Humpty Doo'.

    First comma-separated part that is not a lot, unit or street number, with
    the state and postcode removed.
    """
    for part in (p.strip() for p in address.split(",")):
        if not part or re.match(r"(?i)(lot|unit|flat|apartment|apt)\b|\d", part):
            continue
        return re.sub(r"\s+NT\s*\d{4}$", "", part).strip()
    raise FormParseError(f"no community in address {address!r}")


def tenant_id_for(email: str) -> str:
    """Stable pseudonymous id: the same tenant gets the same id on every form,
    and no name, phone or email is stored in the Report."""
    return "T-" + hashlib.sha256(email.strip().lower().encode()).hexdigest()[:8].upper()


def raw_text_for(item: FormItem) -> str:
    """Issue, location and comments only: the tenant's own words, no priority or personal details."""
    parts = [item.issue, f"Location: {item.location}"]
    if item.comments:
        parts.append(f"Comments: {item.comments}")
    return "\n".join(parts)


def parse_geh_form(path: Path) -> list[Report]:
    """One Report per issue on the form at path.

    Raises:
        FormParseError: not a GEH form, a header field is missing, or an item is invalid.
    """
    from pypdf import PdfReader

    pages = PdfReader(path).pages
    form = next((p for p in pages if is_geh_form(p.extract_text() or "")), None)
    if form is None:
        raise FormParseError("not a GEH repair request form")
    plain = form.extract_text().splitlines()
    layout = form.extract_text(extraction_mode="layout")

    region = _layout_value(layout, "Region")
    address = _layout_value(layout, "Property address")
    email_match = _EMAIL.search(layout)
    if not (region and address and email_match):
        raise FormParseError("missing Region, Property address or Email address")

    email_line = next(k for k, l in enumerate(plain) if email_match.group(0) in l)
    items = parse_items(plain[email_line + 1:])

    community = community_from_address(address)
    tenant_id = tenant_id_for(email_match.group(0))
    received = datetime.fromtimestamp(path.stat().st_mtime, tz=NT_TIME)

    reports: list[Report] = []
    for item in items:
        if item.previously_reported:
            ts = datetime.strptime(item.previously_reported, "%d/%m/%Y").replace(tzinfo=NT_TIME)
            source = f"date previously reported to DIPL ({item.method or 'method not given'})"
        else:
            ts, source = received, "form received (file time)"
        try:
            reports.append(create_report(
                tenant_id=tenant_id,
                raw_text=raw_text_for(item),
                source_tag=SourceTag.TENANT_DIRECT,
                community=community,
                original_report_timestamp=ts,
                region=region,
                source_file=path.name,
                source_item=item.number,
                timestamp_source=source,
            ))
        except ValidationError as e:
            raise FormParseError(f"item {item.number}: {e}") from e
    return reports
