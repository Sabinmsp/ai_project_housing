import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import openai
import pytest

import demo
from triage import evaluation, extraction, recording
from triage.adapter import to_rank_input
from triage.evaluation import Evaluation, evaluate
from triage.extraction import OfflineExtractor
from triage.escalation import escalate
from triage.intake import SQLiteReportRepository, create_report
from triage.explain import ReasoningTrace, build_traces, render_tenant_sms
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


def test_child_ids_in_the_sms_can_be_escalated() -> None:
    repo, report = SQLiteReportRepository(), compound_report()
    repo.save(report)
    extraction = ReportExtraction(faults=(WIRE, GAS))
    jobs = demo._standin_jobs(report, extraction)
    demo._save_children(repo, extraction, jobs)
    later = datetime(2026, 9, 5, tzinfo=timezone.utc)
    updated, _ = escalate(repo, jobs[1].request_id, "still smell gas", later, OfflineExtractor())
    assert updated.request_id == report.request_id
    assert updated.original_report_timestamp == report.original_report_timestamp
    assert repo.get_child(jobs[0].request_id).facts == WIRE


def duplicate(n: int, *also: str) -> str:
    return f'Possible duplicate: {n} entries for "gas leak" in this report (also {", ".join(also)}). Check before dispatching.'


def test_shared_taxonomy_entry_flags_both_jobs_and_keeps_both() -> None:
    first = fault("smell gas", ["gas leak"])
    second = fault("I can smell gas", ["gas leak"])
    jobs = demo._standin_jobs(compound_report(), ReportExtraction(faults=(first, second)))
    assert len(jobs) == 2
    a, b = (j.request_id for j in jobs)
    assert [j.flags for j in jobs] == [(duplicate(2, b),), (duplicate(2, a),)]


THREE_GAS = (fault("smell gas", ["gas leak"]), fault("I can smell gas", ["gas leak"]), fault("gas", ["gas leak"]))


def test_three_shared_entries_give_the_count_and_name_every_sibling() -> None:
    jobs = demo._standin_jobs(compound_report(), ReportExtraction(faults=THREE_GAS))
    a, b, c = (j.request_id for j in jobs)
    assert [j.flags for j in jobs] == [(duplicate(3, b, c),), (duplicate(3, a, c),), (duplicate(3, a, b),)]


def test_duplicate_flag_never_reaches_the_tenant_sms() -> None:
    jobs = demo._standin_jobs(compound_report(), ReportExtraction(faults=THREE_GAS))
    traces = build_traces(rank([to_rank_input(j) for j in jobs]), {j.request_id: j for j in jobs})
    for trace in traces:
        assert any(f.startswith("Possible duplicate") for f in trace.flags)  # the coordinator sees it
        without = ReasoningTrace.model_validate({**trace.model_dump(), "flags": ()})
        assert render_tenant_sms(trace) == render_tenant_sms(without)
        others = {j.request_id for j in jobs} - {trace.job_id}
        assert not any(job_id in render_tenant_sms(trace) for job_id in others)


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



# --- run modes: offline / live / recorded / record ------------------------------------

def write_reports(folder: Path, *messages: str) -> Path:
    folder.mkdir()
    for i, message in enumerate(messages, start=1):
        (folder / f"R{i}.txt").write_text(
            f"Tenant ID: T-{i}\nCommunity: Darwin\nSource: tenant_direct\n"
            f"Reported: 2026-09-2{i} 15:00\nMessage: {message}\n", encoding="utf-8")
    return folder


def write_recording(store: Path, raw_text: str, response: str, prompt: str | None = None) -> None:
    store.mkdir(exist_ok=True)
    record = {"model": "gpt-4o", "recorded_at": "2026-10-06",
              "prompt_hash": prompt or recording.prompt_hash(), "response": response}
    recording.recording_path(raw_text, store).write_text(json.dumps(record), encoding="utf-8")


def forbid_offline_double(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("regex double used outside --offline")
    monkeypatch.setattr(OfflineExtractor, "complete_json", refuse)


class FakeLive(OfflineExtractor):
    """Stands in for the API client: answers like the offline reader, no network."""
    name = "llm:fake"
    model = "gpt-4o-fake"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "recorded"
    monkeypatch.setattr(demo, "RECORDED_DIR", path)
    return path


def test_no_key_replays_recording_and_flags_the_miss(tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    folder = write_reports(tmp_path / "reports", "toilet blocked", "stove's not working")
    write_recording(store, "toilet blocked", OfflineExtractor.read("toilet blocked").model_dump_json())
    forbid_offline_double(monkeypatch)
    demo.main([str(folder)])
    out = capsys.readouterr().out
    assert out.startswith("MODE: recorded gpt-4o responses (recorded 2026-10-06) — no API calls.")
    assert out.count("status=ok") == 1  # the recorded report
    assert out.count("status=flagged_for_human") == 1  # the miss; the run carried on
    assert recording.MISS in out


def test_recording_from_an_older_prompt_is_a_miss(tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    folder = write_reports(tmp_path / "reports", "toilet blocked")
    write_recording(store, "toilet blocked", OfflineExtractor.read("toilet blocked").model_dump_json(), prompt="old")
    forbid_offline_double(monkeypatch)
    demo.main([str(folder)])
    out = capsys.readouterr().out
    assert "status=flagged_for_human" in out and "recorded under an older prompt" in out
    assert "status=ok" not in out


def test_invalid_recorded_response_is_flagged_like_a_live_one(tmp_path: Path, store: Path,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    folder = write_reports(tmp_path / "reports", "toilet blocked")
    write_recording(store, "toilet blocked", '{"faults": [{"fault_description": null}]}')
    demo.main([str(folder)])
    out = capsys.readouterr().out
    assert "status=flagged_for_human  attempts=2" in out
    assert "ValidationError" in out


def test_key_present_goes_live_and_never_reads_recordings(tmp_path: Path, store: Path,
                                                          monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    folder = write_reports(tmp_path / "reports", "toilet blocked")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(demo, "OpenAICompatibleClient", FakeLive)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("recordings read in live mode")
    monkeypatch.setattr(demo, "RecordedClient", refuse)
    demo.main([str(folder)])
    out = capsys.readouterr().out
    assert out.startswith("MODE: live gpt-4o-fake — paid API calls.")
    assert "extractor: llm:fake" in out and "status=ok" in out


def test_offline_flag_with_key_never_builds_the_api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    folder = write_reports(tmp_path / "reports", "toilet blocked")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    demo.main(["--offline", str(folder)])  # the autouse guard raises if the API client is built
    out = capsys.readouterr().out
    assert out.startswith("MODE: offline") and "extractor: offline" in out


def test_record_writes_one_file_per_report(tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messages = ("toilet blocked", "stove's not working")
    folder = write_reports(tmp_path / "reports", *messages)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(demo, "OpenAICompatibleClient", FakeLive)
    demo.main(["--record", str(folder)])
    assert sorted(p.name for p in store.iterdir()) == sorted(recording.recording_path(m, store).name for m in messages)
    for message in messages:
        record = json.loads(recording.recording_path(message, store).read_text(encoding="utf-8"))
        assert record["prompt_hash"] == recording.prompt_hash("gpt-4o-fake")
        assert record["model"] == "gpt-4o-fake"
        assert ReportExtraction.model_validate_json(record["response"]) == OfflineExtractor.read(message)


def test_record_without_key_exits(store: Path) -> None:
    with pytest.raises(SystemExit, match="--record needs"):
        demo.main(["--record"])


def test_stage2_marks_a_span_that_backs_no_claim(tmp_path: Path, store: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    folder = write_reports(tmp_path / "reports", "toilet blocked")
    answer = json.loads(OfflineExtractor.read("toilet blocked").model_dump_json())
    answer["faults"][0]["quoted_spans"].append({"field": "impact_status", "text": "ongoing"})
    write_recording(store, "toilet blocked", json.dumps(answer))
    demo.main([str(folder)])
    out = capsys.readouterr().out
    assert "quote [impact_status]: (ignored — backs no claim)" in out
    assert "'ongoing'" not in out


def test_offline_never_reads_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = write_reports(tmp_path / "reports", "toilet blocked")
    calls: list[object] = []
    monkeypatch.setattr(demo, "_load_dotenv", lambda *args, **kwargs: calls.append(args))
    demo.main(["--offline", str(folder)])
    assert calls == []


KEY = "sk-test-not-a-real-key"


class RateLimitedOnStove(FakeLive):
    """Answers like FakeLive, except the provider rate-limits the stove report every time."""

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        if "stove" in extraction.report_text_from_prompt(user):
            response = httpx.Response(429, request=httpx.Request("POST", "https://api.example"))
            raise openai.RateLimitError(f"Rate limit reached for key {KEY}", response=response, body=None)
        return super().complete_json(system, user, schema)


def test_provider_error_flags_that_report_and_the_run_continues(tmp_path: Path, store: Path,
                                                                monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    folder = write_reports(tmp_path / "reports", "toilet blocked", "stove's not working", "dripping tap")
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setattr(demo, "OpenAICompatibleClient", RateLimitedOnStove)
    demo.main([str(folder)])
    out = capsys.readouterr().out
    assert out.count("status=ok") == 2
    assert out.count("status=flagged_for_human") == 1
    assert "RateLimitError: Rate limit reached for key <redacted>" in out
    assert KEY not in out


@pytest.mark.parametrize("content", ["not json{", "{}", '["a list"]', '{"model": "gpt-4o", "recorded_at": 1}'])
def test_unreadable_recording_is_a_flagged_miss_and_the_run_continues(
        tmp_path: Path, store: Path, capsys: pytest.CaptureFixture[str], content: str) -> None:
    folder = write_reports(tmp_path / "reports", "toilet blocked", "stove's not working")
    write_recording(store, "stove's not working", OfflineExtractor.read("stove's not working").model_dump_json())
    recording.recording_path("toilet blocked", store).write_text(content, encoding="utf-8")
    (store / "stray.json").write_text(content, encoding="utf-8")
    demo.main([str(folder)])
    out = capsys.readouterr().out
    assert out.startswith("MODE: recorded gpt-4o responses (recorded 2026-10-06)")
    assert out.count("status=ok") == 1
    assert out.count("status=flagged_for_human") == 1
    assert "recording unreadable" in out


def test_demo_run_stores_compound_children_for_escalation(tmp_path: Path, store: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    folder = write_reports(tmp_path / "reports", GAS_AND_WIRE)
    write_recording(store, GAS_AND_WIRE, ReportExtraction(faults=(WIRE, GAS)).model_dump_json())
    seen: list[tuple[SQLiteReportRepository, list[EnrichedJob]]] = []
    real = demo._save_children
    monkeypatch.setattr(demo, "_save_children", lambda repo, ext, jobs: (seen.append((repo, jobs)), real(repo, ext, jobs)))
    demo.main([str(folder)])
    ((repo, jobs),) = seen
    assert [repo.get_child(j.request_id).facts for j in jobs] == [WIRE, GAS]
