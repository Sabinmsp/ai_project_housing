from datetime import datetime, timezone

import pytest

import demo
from triage import evaluation
from triage.extraction import OfflineExtractor
from triage.intake import create_report


def test_stub_takes_base_and_bump_from_evaluation_not_reason_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "NO_ALTERNATIVE", "reworded +1 reason")
    report = create_report(tenant_id="T", raw_text="toilet blocked", source_tag="tenant_direct",
                           community="Darwin", original_report_timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc))
    job = demo._standin_stages_3_to_5(report, OfflineExtractor.read(report.raw_text))
    assert job.tally_reasons == ("reworded +1 reason",)  # the patch reached evaluation
    assert (job.base_points, job.severity_bump) == (3, 1)
