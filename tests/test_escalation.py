from datetime import datetime, timedelta, timezone

import pytest

from triage.escalation import escalate
from triage.extraction import OfflineExtractor, extract
from triage.intake import SQLiteReportRepository, UnknownRequestError, create_report

T0 = datetime(2026, 9, 10, tzinfo=timezone(timedelta(hours=9, minutes=30)))


def saved_report(repo, text):
    r = create_report(tenant_id="T-1", raw_text=text, source_tag="tenant_direct",
                      community="Wurrumiyanga", original_report_timestamp=T0)
    repo.save(r)
    return r


def test_escalation_re_enters_stage_2_and_keeps_timestamp():
    repo, client = SQLiteReportRepository(), OfflineExtractor()
    r = saved_report(repo, "roof leaking in the bedroom")
    assert extract(r, client).extraction.faults[0].mechanism_type is None

    later = T0 + timedelta(days=20)
    updated, res = escalate(repo, r.request_id,
                            "now water coming through the light fitting", later, client)

    assert res.extraction.faults[0].mechanism_type == "active"  # re-read over the whole history
    assert updated.request_id == r.request_id
    assert updated.original_report_timestamp == T0  # fairness never resets


def test_escalation_never_fuzzy_matches():
    repo = SQLiteReportRepository()
    r = saved_report(repo, "toilet blocked")
    with pytest.raises(UnknownRequestError):
        escalate(repo, r.request_id.lower(), "worse", T0, OfflineExtractor())
