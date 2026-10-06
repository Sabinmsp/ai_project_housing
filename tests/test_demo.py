from datetime import datetime, timezone

import pytest

import demo
from triage import evaluation
from triage.adapter import to_rank_input
from triage.evaluation import Evaluation, evaluate
from triage.extraction import OfflineExtractor
from triage.intake import create_report
from triage.models import EnrichedJob, ExtractedFacts, ReportExtraction
from triage.ranking import rank


def test_stub_takes_base_and_bump_from_evaluation_not_reason_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "NO_ALTERNATIVE", "reworded +1 reason")
    report = create_report(tenant_id="T", raw_text="toilet blocked", source_tag="tenant_direct",
                           community="Darwin", original_report_timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc))
    (job,) = demo._standin_jobs(report, OfflineExtractor.read(report.raw_text))
    assert job.tally_reasons == ("reworded +1 reason",)  # the patch reached evaluation
    assert (job.base_points, job.severity_bump) == (3, 1)


T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def stub(raw_text: str, facts: dict[str, object]) -> tuple[EnrichedJob, Evaluation]:
    """Run the stub on hand-built facts; also return what evaluate() says for the same input."""
    report = create_report(tenant_id="T", raw_text=raw_text, source_tag="tenant_direct",
                           community="Darwin", original_report_timestamp=T0)
    f = ExtractedFacts.model_validate({
        "fault_description": None, "taxonomy_match": [], "alternative_mentioned": False,
        "coping_mentioned": False, "hazard_status": "none", "mechanism_type": None,
        "harm_claimed": False, "fault_or_sign": "fault", "claim_mismatch": None,
        "worsening_mentioned": False, "quoted_spans": [], **facts,
    })
    unverified = frozenset((s.field, s.text) for s in f.quoted_spans if s.text not in raw_text)
    return demo._standin_stages_3_to_5(report, f), evaluate(f, unverified)


ROOF_COLLAPSE = "ceiling has come down, water coming through the light fitting"
ROOF_FACTS: dict[str, object] = {
    "fault_description": "ceiling has come down",
    "hazard_status": "described", "mechanism_type": "active",
    "quoted_spans": [{"field": "fault_description", "text": "ceiling has come down"},
                     {"field": "hazard", "text": "water coming through the light fitting"}],
}


def test_untiered_active_hazard_is_ranked_not_held() -> None:
    job, _ = stub(ROOF_COLLAPSE, ROOF_FACTS)
    assert (job.tier, job.urgency_tally, job.safety_level) == (None, None, "active")
    assert not job.in_review_band
    result = rank([to_rank_input(job)])
    assert result.review_band == () and [e.job_id for e in result.ranked] == [job.request_id]


def test_stub_copies_evaluate_flags_unchanged() -> None:
    # Ambiguous match plus a quote the report doesn't contain: two flags from evaluate().
    job, ev = stub("drain blocked, smells", {
        "fault_description": "drain blocked",
        "taxonomy_match": ["blocked drain", "sewage leak"],
        "quoted_spans": [{"field": "fault_description", "text": "drain blocked"},
                         {"field": "taxonomy_match", "text": "sewage pouring out"}],
    })
    assert len(ev.flags) == 2
    assert job.flags == ev.flags


def test_stub_copies_evaluate_safety_reason() -> None:
    job, ev = stub(ROOF_COLLAPSE, ROOF_FACTS)
    assert job.safety_reason == ev.safety.reason
    assert "'water coming through the light fitting'" in job.safety_reason


def test_compound_report_raises_instead_of_dropping_a_fault() -> None:
    report = create_report(tenant_id="T", raw_text="toilet blocked and I can smell gas", source_tag="tenant_direct",
                           community="Darwin", original_report_timestamp=T0)
    toilet = OfflineExtractor.read("toilet blocked").faults[0]
    gas = OfflineExtractor.read("I can smell gas").faults[0]
    with pytest.raises(NotImplementedError, match="compound reports split into jobs in step 3.5"):
        demo._standin_jobs(report, ReportExtraction(faults=[toilet, gas]))


def test_single_fault_report_gives_one_job() -> None:
    report = create_report(tenant_id="T", raw_text="toilet blocked", source_tag="tenant_direct",
                           community="Darwin", original_report_timestamp=T0)
    (job,) = demo._standin_jobs(report, OfflineExtractor.read(report.raw_text))
    assert job.urgency_tally == 4
