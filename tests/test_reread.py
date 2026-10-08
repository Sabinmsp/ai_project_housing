import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import demo
from triage import reread, second_reader
from triage.evaluation import evaluate
from triage.explain import build_traces, render_coordinator, tenant_sms, tenant_why
from triage.adapter import to_rank_input
from triage.extraction import OfflineExtractor
from triage.intake import create_report
from triage.models import ExtractedFacts, ExtractionResult, ExtractionStatus, ReportExtraction, ReRead, SecondReading
from triage.ranking import rank
from triage.reread import READING_ORDER, combine, reading, should_reread
from triage.verification import verify_spans

TEXT = "wire hanging near the sink, kids keep touching it, the other room is fine"
WIRE = "wire hanging near the sink"
HAZARD_QUOTE = {"active": "kids keep touching it", "conditional": "near the sink", "unclear": "wire hanging"}


def fault(hazard: str = "none", taxonomy: tuple[str, ...] = ("exposed electrical wires",), words: str = WIRE,
          **extra: Any) -> ExtractedFacts:
    """One fault with the given hazard reading (active / conditional / unclear / none)."""
    status = {"active": "described", "conditional": "described"}.get(hazard, hazard)
    spans = [{"field": "fault_description", "text": words}]
    if taxonomy:
        spans.append({"field": "taxonomy_match", "text": words})
    if hazard != "none":
        spans.append({"field": "hazard", "text": HAZARD_QUOTE[hazard]})
    return ExtractedFacts.model_validate({
        "fault_description": words, "taxonomy_match": list(taxonomy), "alternative_mentioned": False,
        "coping_mentioned": False, "hazard_status": status,
        "mechanism_type": hazard if hazard in ("active", "conditional") else None, "harm_claimed": False,
        "fault_or_sign": "fault", "claim_mismatch": None, "worsening_mentioned": False, "quoted_spans": spans, **extra,
    })


def ok(*faults: ExtractedFacts) -> ExtractionResult:
    return ExtractionResult(request_id="R-1", status=ExtractionStatus.OK, extraction=ReportExtraction(faults=faults),
                            attempts=1, errors=[], extractor="llm:fake")


def flagged(*fields: str) -> SecondReading:
    return SecondReading(status="ran", flags=tuple(f"disagrees on {f} — check (x)" for f in fields), detail=None)


# --- when it runs ------------------------------------------------------------------------

@pytest.mark.parametrize(("client", "second", "expected"), [
    ("llm:gpt-4o", flagged("hazard"), True),
    ("llm:gpt-4o", flagged("mechanism"), True),
    ("llm:gpt-4o", SecondReading(status="ran", flags=("low confidence on hazard — check (x)",), detail=None), True),
    ("llm:gpt-4o", SecondReading(status="ran", flags=(), detail=None), False),  # agrees
    ("llm:gpt-4o", flagged("alternative", "fault_or_sign"), False),  # not a hazard field
    ("llm:gpt-4o", second_reader.not_run("not run (no key)"), False),
    ("llm:gpt-4o", None, False),
    ("offline", flagged("hazard"), False),
    ("recorded:gpt-4o", flagged("hazard"), False),
    ("recording:llm:gpt-4o", flagged("hazard"), False),  # a re-read would overwrite the recording
])
def test_rereads_only_in_live_mode_after_a_hazard_or_mechanism_flag(client: str, second: SecondReading | None,
                                                                    expected: bool) -> None:
    assert should_reread(client, second) is expected


def test_jev_never_sets_a_value() -> None:
    # combine() takes the two extraction readings only; the second reader can't reach it.
    assert list(inspect.signature(combine).parameters) == ["read_1", "reread", "single"]


# --- combining ---------------------------------------------------------------------------

def test_reads_agree_unchanged_and_no_inconsistency_flag() -> None:
    facts, record, flags = combine(fault("unclear"), ok(fault("unclear")), single=True)
    assert facts == fault("unclear")
    assert record == ReRead(read_1="unclear", read_2="unclear", unavailable_reason=None, used="read 1") and flags == ()


def test_read_2_higher_raises_safety_and_flags() -> None:
    read_1 = fault("none", alternative_mentioned=False)
    facts, record, flags = combine(read_1, ok(fault("active")), single=True)
    assert reading(facts) == "active"
    assert {k: v for k, v in facts.model_dump().items() if k not in ("hazard_status", "mechanism_type", "quoted_spans")} \
        == {k: v for k, v in read_1.model_dump().items() if k not in ("hazard_status", "mechanism_type", "quoted_spans")}
    assert record == ReRead(read_1="none", read_2="active", unavailable_reason=None, used="read 2")
    assert flags == ("Readings inconsistent — read 1: none, read 2: active; the safer reading is used. Check.",)


def test_read_2_lower_keeps_read_1_and_flags() -> None:
    facts, record, flags = combine(fault("conditional"), ok(fault("none")), single=True)
    assert facts == fault("conditional") and record.used == "read 1"
    assert flags == ("Readings inconsistent — read 1: conditional, read 2: none; the safer reading is used. Check.",)


def test_reread_error_keeps_read_1_and_flags_unavailable() -> None:
    failed = ExtractionResult(request_id="R-1", status=ExtractionStatus.FLAGGED_FOR_HUMAN, extraction=None, attempts=2,
                              errors=["attempt 1: ValidationError: x", "attempt 2: ValidationError: y"], extractor="llm:fake")
    facts, record, flags = combine(fault("none"), failed, single=True)
    assert facts == fault("none") and record.read_2 is None and record.used == "read 1"
    assert str(record.unavailable_reason).startswith("attempt 1: ValidationError")
    assert len(flags) == 1 and flags[0].startswith("Re-read unavailable — attempt 1: ValidationError") \
        and flags[0].endswith("read 1 kept. Check.")


def test_matched_by_taxonomy_never_position() -> None:
    gas = fault("active", taxonomy=("gas leak",), words="kids keep touching it")
    wire_2 = fault("conditional")
    facts, record, _ = combine(fault("none"), ok(gas, wire_2), single=False)  # the wire is second in read 2
    assert record.read_2 == "conditional" and reading(facts) == "conditional"


def test_no_matching_fault_keeps_read_1_and_flags() -> None:
    # Compound report: taxonomy matching.
    facts, record, flags = combine(fault("none"), ok(fault("active", taxonomy=("gas leak",))), single=False)
    assert facts == fault("none") and record.read_2 is None and flags[0].startswith("Re-read unavailable — no re-read fault")


def test_compound_fault_with_empty_taxonomy_is_unavailable() -> None:
    _, record, flags = combine(fault("none", taxonomy=()), ok(fault("active", taxonomy=())), single=False)
    assert record == ReRead(read_1="none", read_2=None, unavailable_reason="no re-read fault matches taxonomy []",
                            used="read 1")
    assert flags == ("Re-read unavailable — no re-read fault matches taxonomy []; read 1 kept. Check.",)


def test_single_fault_with_empty_taxonomy_is_compared_and_used() -> None:
    # One fault each read: matched directly, e.g. an unlisted fault with no taxonomy.
    facts, record, flags = combine(fault("none", taxonomy=()), ok(fault("active", taxonomy=())), single=True)
    assert reading(facts) == "active"
    assert record == ReRead(read_1="none", read_2="active", unavailable_reason=None, used="read 2")
    assert flags == ("Readings inconsistent — read 1: none, read 2: active; the safer reading is used. Check.",)


def test_single_fault_matches_directly_even_if_taxonomy_differs() -> None:
    _, record, _ = combine(fault("none"), ok(fault("conditional", taxonomy=("gas leak",))), single=True)
    assert record.read_2 == "conditional" and record.used == "read 2"


# --- can only raise safety ---------------------------------------------------------------

def evaluated(facts: ExtractedFacts) -> tuple[int, Any]:
    ev = evaluate(facts, verify_spans(TEXT, facts))
    return ev.safety.level, ev.tally


readings = st.sampled_from(READING_ORDER)


@given(readings, readings, st.booleans())
def test_reread_never_lowers_safety_and_never_changes_the_tally(first: str, second: str, alternative: bool) -> None:
    read_1 = fault(first) if not alternative else ExtractedFacts.model_validate(
        {**fault(first).model_dump(), "alternative_mentioned": True,
         "quoted_spans": [*fault(first).model_dump()["quoted_spans"],
                          {"field": "alternative_mentioned", "text": "the other room is fine"}]})
    facts, _, _ = combine(read_1, ok(fault(second)), single=True)
    level_1, tally_1 = evaluated(read_1)
    level_2, _ = evaluated(fault(second))
    level, tally = evaluated(facts)
    assert level == max(level_1, level_2)  # the safer reading, by its evaluated level
    assert tally == tally_1  # tally inputs stay from read 1


# --- coordinator and tenant ----------------------------------------------------------------

def built_job() -> Any:
    report = create_report(tenant_id="T", raw_text=TEXT, source_tag="tenant_direct", community="Darwin",
                           original_report_timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc))
    (job,) = demo._build_jobs(report, ReportExtraction(faults=(fault("none"),)))
    return job


def test_trace_shows_both_readings() -> None:
    job = reread.attach(built_job(), ReRead(read_1="none", read_2="active", unavailable_reason=None, used="read 2"),
                        ("Readings inconsistent — read 1: none, read 2: active; the safer reading is used. Check.",))
    (tr,) = build_traces(rank([to_rank_input(job)]), {job.request_id: job})
    view = render_coordinator(tr)
    assert "read 1: none; read 2: active; used: read 2" in view and "Readings inconsistent" in view


def test_unavailable_display_has_no_safer_reading_wording() -> None:
    record = ReRead(read_1="none", read_2=None, unavailable_reason="attempt 1: ValidationError: x", used="read 1")
    job = reread.attach(built_job(), record, ("Re-read unavailable — attempt 1: ValidationError: x; read 1 kept. Check.",))
    (tr,) = build_traces(rank([to_rank_input(job)]), {job.request_id: job})
    (row,) = [line for line in render_coordinator(tr).splitlines() if line.startswith("re_read")]
    assert row.split(None, 1)[1].rstrip() == "read 1: none; read 2: unavailable (attempt 1: ValidationError: x); used: read 1"
    assert "safer reading kept" not in row


def test_reread_record_needs_a_reason_exactly_when_unavailable() -> None:
    with pytest.raises(ValueError, match="unavailable_reason"):
        ReRead(read_1="none", read_2=None, unavailable_reason=None, used="read 1")
    with pytest.raises(ValueError, match="unavailable_reason"):
        ReRead(read_1="none", read_2="active", unavailable_reason="x", used="read 2")
    with pytest.raises(ValueError, match="can't be used"):
        ReRead(read_1="none", read_2=None, unavailable_reason="x", used="read 2")


def test_tenant_text_unaffected_by_the_reread_record() -> None:
    job = built_job()
    marked = reread.attach(job, ReRead(read_1="none", read_2="active", unavailable_reason=None, used="read 2"),
                           ("Readings inconsistent — read 1: none, read 2: active; the safer reading is used. Check.",))
    for render in (tenant_sms, lambda j: tenant_why(j, pinned=False), lambda j: tenant_why(j, pinned=True)):
        assert render(marked) == render(job)
        assert "re-read" not in render(marked).lower() and "reading" not in render(marked).lower()


# --- demo wiring (fake live extractor, fake Jev) -------------------------------------------

class SequenceLive(OfflineExtractor):
    """A live extractor returning one canned extraction per call, counting calls."""

    name = "llm:fake"
    model = "gpt-4o-fake"
    replies: list[ReportExtraction] = []
    calls = 0

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        type(self).calls += 1
        reply = type(self).replies[min(type(self).calls, len(type(self).replies)) - 1]
        return reply.model_dump_json()


def jev_says(monkeypatch: pytest.MonkeyPatch, second: SecondReading) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.setattr(second_reader, "read", lambda client, state, facts: second)


def run_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
             *replies: ReportExtraction) -> str:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    SequenceLive.replies, SequenceLive.calls = list(replies), 0
    monkeypatch.setattr(demo, "OpenAICompatibleClient", SequenceLive)
    folder = tmp_path / "r"
    folder.mkdir()
    (folder / "R1.txt").write_text(f"Tenant ID: T-1\nCommunity: Darwin\nSource: tenant_direct\n"
                                   f"Reported: 2026-09-21 15:00\nMessage: {TEXT}\n", encoding="utf-8")
    demo.main([str(folder)])
    return capsys.readouterr().out


def test_demo_no_flag_no_reread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    jev_says(monkeypatch, SecondReading(status="ran", flags=(), detail=None))
    out = run_live(tmp_path, monkeypatch, capsys, ReportExtraction(faults=(fault("none"),)),
                   ReportExtraction(faults=(fault("active"),)))
    assert SequenceLive.calls == 1 and "re_read" not in out


def test_demo_flag_and_higher_reread_raises_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    jev_says(monkeypatch, flagged("hazard"))
    out = run_live(tmp_path, monkeypatch, capsys, ReportExtraction(faults=(fault("none"),)),
                   ReportExtraction(faults=(fault("active"),)))
    ranked = out.split("RANKED QUEUE")[1]
    assert SequenceLive.calls == 2
    assert "read 1: none; read 2: active; used: read 2" in ranked and "Readings inconsistent" in ranked
    assert any(line.startswith("safety_level") and line.split()[1] == "2" for line in ranked.splitlines())


def test_demo_flag_and_lower_reread_keeps_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    jev_says(monkeypatch, flagged("mechanism"))
    out = run_live(tmp_path, monkeypatch, capsys, ReportExtraction(faults=(fault("active"),)),
                   ReportExtraction(faults=(fault("none"),)))
    ranked = out.split("RANKED QUEUE")[1]
    assert SequenceLive.calls == 2 and "used: read 1" in ranked
    assert any(line.startswith("safety_level") and line.split()[1] == "2" for line in ranked.splitlines())


def test_demo_offline_never_rereads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    jev_says(monkeypatch, flagged("hazard"))
    calls = []
    monkeypatch.setattr(reread, "combine", lambda *a: calls.append(a))
    folder = tmp_path / "r"
    folder.mkdir()
    (folder / "R1.txt").write_text(f"Tenant ID: T-1\nCommunity: Darwin\nSource: tenant_direct\n"
                                   f"Reported: 2026-09-21 15:00\nMessage: {TEXT}\n", encoding="utf-8")
    demo.main([str(folder), "--offline"])
    assert calls == [] and "re_read" not in capsys.readouterr().out
