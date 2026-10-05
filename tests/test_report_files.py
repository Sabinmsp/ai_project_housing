from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from triage.models import SourceTag
from triage.report_files import ReportFileError, load_reports, parse_report_text

ACST = timezone(timedelta(hours=9, minutes=30))
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

GOOD = """Tenant ID: T-03
Community: Maningrida
Source: tenant_direct
Reported: 2026-09-29 15:00
Message:
roof leaking in kids room,
water coming through the light fitting
"""


def test_all_fields_read():
    r = parse_report_text(GOOD)
    assert r.tenant_id == "T-03"
    assert r.community == "Maningrida"
    assert r.source_tag is SourceTag.TENANT_DIRECT
    assert r.raw_text == "roof leaking in kids room, water coming through the light fitting"
    assert r.request_id.startswith("R-")


def test_time_without_timezone_is_nt_time():
    r = parse_report_text(GOOD)
    assert r.original_report_timestamp == datetime(2026, 9, 29, 15, 0, tzinfo=ACST)


@pytest.mark.parametrize("value", ["29/09/2026 15:00", "29/09/2026 3:00 pm"])
def test_australian_date_format(value):
    r = parse_report_text(GOOD.replace("2026-09-29 15:00", value))
    assert r.original_report_timestamp == datetime(2026, 9, 29, 15, 0, tzinfo=ACST)


def test_phone_source_alias_is_officer():
    r = parse_report_text(GOOD.replace("tenant_direct", "phone"))
    assert r.source_tag is SourceTag.OFFICER


def test_request_id_kept_when_given():
    r = parse_report_text("Request ID: R-ABC12345\n" + GOOD)
    assert r.request_id == "R-ABC12345"


@pytest.mark.parametrize("line", ["Tenant ID: T-03\n", "Reported: 2026-09-29 15:00\n"])
def test_missing_field_is_named(line):
    with pytest.raises(ReportFileError, match=line.split(":")[0]):
        parse_report_text(GOOD.replace(line, ""))


def test_empty_message_rejected():
    with pytest.raises(ReportFileError, match="Message"):
        parse_report_text(GOOD.split("Message:")[0] + "Message:\n")


def test_bad_source_and_bad_date_rejected():
    with pytest.raises(ReportFileError, match="Source"):
        parse_report_text(GOOD.replace("tenant_direct", "carrier pigeon"))
    with pytest.raises(ReportFileError, match="Reported"):
        parse_report_text(GOOD.replace("2026-09-29 15:00", "last Tuesday"))


def test_bad_file_skipped_good_file_kept(tmp_path):
    (tmp_path / "good.txt").write_text(GOOD)
    (tmp_path / "bad.txt").write_text("hello, my toilet is broken")
    (tmp_path / "notes.md").write_text("ignored: not a report file type")
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")
    reports, skipped = load_reports(tmp_path)
    assert [r.tenant_id for r in reports] == ["T-03"]
    assert sorted(p.name for p, _ in skipped) == ["bad.txt", "broken.pdf"]


def test_sample_pdfs_in_reports_folder_all_load():
    reports, skipped = load_reports(REPORTS_DIR)
    assert skipped == []
    assert len(reports) >= 1
