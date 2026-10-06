from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from triage.intake import (
    DuplicateRequestError,
    SQLiteReportRepository,
    UnknownRequestError,
    create_report,
)
from triage.models import Report, SourceTag

T0 = datetime(2026, 9, 1, 8, 0, tzinfo=timezone(timedelta(hours=9, minutes=30)))


def make(**kw):
    base = dict(tenant_id="T-1", raw_text="toilet blocked", source_tag="tenant_direct",
                community="Wadeye", original_report_timestamp=T0)
    base.update(kw)
    return create_report(**base)


def test_report_gets_request_id_and_source_tag():
    r = make(source_tag="officer")
    assert r.request_id.startswith("R-")
    assert r.source_tag is SourceTag.OFFICER


def test_invalid_source_tag_rejected():
    with pytest.raises(ValueError):
        make(source_tag="email")


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError):
        make(original_report_timestamp=datetime(2026, 9, 1, 8, 0))


def test_report_carries_only_intake_fields():
    """Stage 1 stamps; it never interprets. No tier, score or rank can exist here."""
    assert set(Report.model_fields) == {
        "request_id", "tenant_id", "raw_text", "source_tag", "community",
        "original_report_timestamp",
        # provenance only: where the report came from
        "region", "source_file", "source_item", "timestamp_source",
    }
    with pytest.raises(ValidationError):
        Report(**make().model_dump(), urgency_tally=4)


def test_request_ids_are_unique():
    assert len({make().request_id for _ in range(500)}) == 500


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        make(raw_text="")


def test_report_is_immutable():
    r = make()
    with pytest.raises(ValidationError):
        r.original_report_timestamp = T0 + timedelta(days=1)


def test_save_and_get_round_trip():
    repo = SQLiteReportRepository()
    r = make()
    repo.save(r)
    assert repo.get(r.request_id) == r


def test_duplicate_request_id_rejected():
    repo = SQLiteReportRepository()
    r = make(request_id="R-1")
    repo.save(r)
    with pytest.raises(DuplicateRequestError):
        repo.save(make(request_id="R-1"))


def test_escalation_preserves_original_timestamp_and_appends_text():
    repo = SQLiteReportRepository()
    r = make(request_id="R-1")
    repo.save(r)
    later = T0 + timedelta(days=3)
    updated = repo.append_followup("R-1", "now water on the floor", later)
    assert updated.original_report_timestamp == T0
    assert "toilet blocked" in updated.raw_text
    assert "now water on the floor" in updated.raw_text


def test_escalation_matches_exact_id_only():
    repo = SQLiteReportRepository()
    repo.save(make(request_id="R-ABC"))
    with pytest.raises(UnknownRequestError):
        repo.append_followup("r-abc", "worse", T0)
    with pytest.raises(UnknownRequestError):
        repo.append_followup("R-AB", "worse", T0)
