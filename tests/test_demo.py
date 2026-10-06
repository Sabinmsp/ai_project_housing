import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import demo
from triage import evaluation, extraction
from triage.adapter import to_rank_input
from triage.evaluation import Evaluation, evaluate
from triage.extraction import OfflineExtractor
from triage.intake import create_report
from triage.models import EnrichedJob, ExtractedFacts, Report, ReportExtraction
from triage.ranking import rank
from triage.report_files import parse_report_text, read_text
from triage.verification import verify_spans


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
    unverified = verify_spans(raw_text, f)
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


def test_single_fault_report_gives_one_job() -> None:
    report = create_report(tenant_id="T", raw_text="toilet blocked", source_tag="tenant_direct",
                           community="Darwin", original_report_timestamp=T0)
    (job,) = demo._standin_jobs(report, OfflineExtractor.read(report.raw_text))
    assert job.urgency_tally == 4


def test_stub_ignores_a_span_that_backs_no_claim() -> None:
    # impact_status "ongoing" needs no quote; a span quoting "ongoing" is noise, not a fabrication.
    job, ev = stub("toilet blocked", {
        "fault_description": "toilet blocked", "taxonomy_match": ["blocked or broken toilet"],
        "quoted_spans": [{"field": "fault_description", "text": "toilet blocked"},
                         {"field": "taxonomy_match", "text": "toilet blocked"},
                         {"field": "impact_status", "text": "ongoing"}],
    })
    assert job.flags == ()
    assert [(s.field, s.text, s.verified) for s in job.spans] == [
        ("fault_description", "toilet blocked", True),
        ("taxonomy_match", "toilet blocked", True),
    ]  # the "ongoing" span is absent, not listed as unverified



# --- offline by default: no API client without --live --------------------------------


def _forbid_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("API client constructed without --live")
    monkeypatch.setattr(extraction, "OpenAICompatibleClient", refuse)
    monkeypatch.setattr(demo, "OpenAICompatibleClient", refuse)


def test_demo_without_live_never_builds_the_api_client(monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    _forbid_api_client(monkeypatch)
    demo.main([])
    out = capsys.readouterr().out
    assert out.startswith("MODE: offline")
    assert "extractor: offline" in out


def test_demo_live_uses_the_api_client(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeLive(OfflineExtractor):  # answers like the offline reader; no network
        name = "llm:fake"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(demo, "OpenAICompatibleClient", FakeLive)
    demo.main(["--live"])
    out = capsys.readouterr().out
    assert out.startswith("MODE: live LLM (llm:fake)")
    assert "extractor: llm:fake" in out


def test_demo_live_without_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("TRIAGE_API_KEY", "")
    _forbid_api_client(monkeypatch)
    with pytest.raises(SystemExit, match="--live needs"):
        demo.main(["--live"])



# --- step 3.5: one job per fault ------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
GAS_AND_WIRE = "wire sparking near the sink and I can smell gas"


def fault(description: str, names: list[str], hazard: str | None = None) -> ExtractedFacts:
    """One fault record whose quotes are all words from GAS_AND_WIRE."""
    spans = [{"field": "fault_description", "text": description}]
    spans += [{"field": "taxonomy_match", "text": description}] if names else []
    spans += [{"field": "hazard", "text": hazard}] if hazard else []
    return ExtractedFacts.model_validate({
        "fault_description": description, "taxonomy_match": names, "alternative_mentioned": False,
        "coping_mentioned": False, "impact_status": "ongoing",
        "hazard_status": "described" if hazard else "none", "mechanism_type": "active" if hazard else None,
        "harm_claimed": False, "fault_or_sign": "fault", "claim_mismatch": None,
        "worsening_mentioned": False, "quoted_spans": spans,
    })


WIRE = fault("wire sparking", ["exposed electrical wires"], hazard="wire sparking near the sink")
GAS = fault("smell gas", ["gas leak"], hazard="I can smell gas")


def compound_report() -> Report:
    return create_report(tenant_id="T-9", raw_text=GAS_AND_WIRE, source_tag="tenant_direct",
                         community="Wadeye", original_report_timestamp=T0)


def test_compound_report_gives_one_job_per_fault() -> None:
    report = compound_report()
    jobs = demo._standin_jobs(report, ReportExtraction(faults=(WIRE, GAS)))
    assert len(jobs) == 2
    assert len({j.request_id for j in jobs}) == 2
    assert report.request_id not in {j.request_id for j in jobs}  # no child passes for the report
    assert {j.parent_report_id for j in jobs} == {report.request_id}
    assert [j.original_report_timestamp for j in jobs] == [report.original_report_timestamp] * 2
    assert [j.community for j in jobs] == [report.community] * 2
    assert [j.taxonomy_match for j in jobs] == [["exposed electrical wires"], ["gas leak"]]


SEPARATE = [
    fault("wire sparking", ["exposed electrical wires"]),
    fault("smell gas", ["gas leak"]),
    fault("near the sink", []),
    fault("sparking", []),
]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_n_faults_give_n_jobs(n: int) -> None:
    report = compound_report()
    jobs = demo._standin_jobs(report, ReportExtraction(faults=tuple(SEPARATE[:n])))
    assert [j.fault_description for j in jobs] == [f.fault_description for f in SEPARATE[:n]]
    assert len({j.request_id for j in jobs}) == n
    assert all(j.parent_report_id == report.request_id for j in jobs)
    if n == 1:
        assert jobs[0].request_id == report.request_id  # single fault keeps the report's id
    else:
        assert report.request_id not in {j.request_id for j in jobs}


DUPLICATE = 'Possible duplicate: this report has two entries for "gas leak". Check before dispatching both.'


def test_shared_taxonomy_entry_flags_both_jobs_and_keeps_both() -> None:
    first = fault("smell gas", ["gas leak"])
    second = fault("I can smell gas", ["gas leak"])
    jobs = demo._standin_jobs(compound_report(), ReportExtraction(faults=(first, second)))
    assert len(jobs) == 2
    assert all(DUPLICATE in j.flags for j in jobs)


def test_distinct_taxonomy_entries_are_not_flagged_as_duplicates() -> None:
    jobs = demo._standin_jobs(compound_report(), ReportExtraction(faults=(WIRE, GAS)))
    assert not any("Possible duplicate" in flag for j in jobs for flag in j.flags)


def test_empty_taxonomy_never_flags_duplicates() -> None:
    jobs = demo._standin_jobs(compound_report(), ReportExtraction(faults=(fault("near the sink", []), fault("sparking", []))))
    assert not any("Possible duplicate" in flag for j in jobs for flag in j.flags)


@pytest.mark.parametrize("path", sorted((Path(__file__).parent.parent / "reports").glob("*.pdf")), ids=lambda p: p.name)
def test_single_fault_reports_match_baseline_apart_from_parent_id(path: Path) -> None:
    baseline = json.loads((FIXTURES / "standin_baseline.json").read_text())[path.name]
    report = parse_report_text(read_text(path))
    (job,) = demo._standin_jobs(report, OfflineExtractor.read(report.raw_text))
    assert job.request_id == job.parent_report_id == report.request_id
    dump = job.model_dump(mode="json")
    del dump["request_id"], dump["parent_report_id"]
    assert dump == baseline


@pytest.mark.parametrize("order", [(WIRE, GAS), (GAS, WIRE)], ids=["wire-first", "gas-first"])
def test_children_rank_on_their_own_merits_tie_broken_by_job_id(order: tuple[ExtractedFacts, ...]) -> None:
    jobs = demo._standin_jobs(compound_report(), ReportExtraction(faults=order))
    assert [(j.safety_level, j.urgency_tally) for j in jobs] == [("active", 4), ("active", 4)]
    result = rank([to_rank_input(j) for j in jobs])
    # Same safety, tally and timestamp: job_id decides, never the order the faults were listed.
    assert [e.job_id for e in result.ranked] == sorted(j.request_id for j in jobs)
    assert result.ranked[1].decided_by == "identical; order arbitrary but fixed"
