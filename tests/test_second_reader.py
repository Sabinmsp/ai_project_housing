import io
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import demo
from tests.test_extraction import BANNED_WORDS
from triage import second_reader
from triage.adapter import to_rank_input
from triage.explain import build_traces, render_coordinator, render_review_entry, tenant_sms, tenant_sms_report, tenant_why
from triage.extraction import OfflineExtractor
from triage.intake import create_report
from triage.models import EnrichedJob, ExtractedFacts, ReportExtraction, SecondReading
from triage.ranking import rank
from triage.second_reader import LOW_CONFIDENCE, QUESTIONS, TIMEOUT_S, attach, compare, read, state_for

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def facts(**overrides: Any) -> ExtractedFacts:
    base: dict[str, Any] = {
        "fault_description": "water touching the wire", "taxonomy_match": [], "alternative_mentioned": False,
        "coping_mentioned": False, "hazard_status": "described", "mechanism_type": "active", "harm_claimed": False,
        "fault_or_sign": "fault", "claim_mismatch": None, "worsening_mentioned": False,
        "quoted_spans": [{"field": "fault_description", "text": "water touching the wire"},
                         {"field": "hazard", "text": "water touching the wire"}],
    }
    return ExtractedFacts.model_validate({**base, **overrides})


AGREE = {"hazard": "described", "mechanism": "happening now", "alternative": "no", "fault_or_sign": "fault"}


def response(choices: dict[str, str] = AGREE, confidence: float | dict[str, float] = 0.9) -> dict[str, Any]:
    conf = confidence if isinstance(confidence, dict) else dict.fromkeys(choices, confidence)
    return {
        "model": "jev-1.13.0",
        "answers": {f: {"type": "choice", "choice": c, "probabilities": {c: conf[f]}, "confidence": conf[f]}
                    for f, c in choices.items()},
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


class FakeJev:
    """Answers from a canned response body, recording what it was asked."""

    def __init__(self, body: dict[str, Any] | Exception) -> None:
        self.body = body
        self.states: list[object] = []

    def ask(self, state: object, questions: dict[str, Any]) -> dict[str, Any]:
        self.states.append(state)
        assert questions is QUESTIONS
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


# --- compare -------------------------------------------------------------------------------

def test_agreement_gives_no_flag() -> None:
    reading = read(FakeJev(response()), "water touching the wire", facts())
    assert reading == SecondReading(status="ran", flags=(), detail=None)


@pytest.mark.parametrize(("field", "jev_choice", "expected"), [
    ("hazard", "none", "described"),
    ("mechanism", "could happen", "happening now"),
    ("alternative", "yes", "no"),
    ("fault_or_sign", "sensed cue only", "fault"),
])
def test_each_field_disagreement_gives_its_own_flag(field: str, jev_choice: str, expected: str) -> None:
    reading = read(FakeJev(response({**AGREE, field: jev_choice})), "text", facts())
    assert reading.flags == (
        f"disagrees on {field} — check (extraction: {expected}; second reader: {jev_choice}, confidence 0.90)",
    )


@pytest.mark.parametrize("field", list(QUESTIONS))
def test_low_confidence_gives_a_flag_even_when_agreeing(field: str) -> None:
    conf = {**dict.fromkeys(AGREE, 0.9), field: 0.69}
    reading = read(FakeJev(response(AGREE, conf)), "text", facts())
    assert reading.flags == (f"low confidence on {field} — check (second reader: {AGREE[field]}, confidence 0.69)",)


def test_confidence_at_the_cutoff_is_not_low() -> None:
    assert LOW_CONFIDENCE == 0.7
    assert read(FakeJev(response(AGREE, 0.7)), "text", facts()).flags == ()


def test_disagreement_with_low_confidence_is_one_line() -> None:
    reading = read(FakeJev(response({**AGREE, "hazard": "unclear"}, {**dict.fromkeys(AGREE, 0.9), "hazard": 0.4})),
                   "text", facts())
    assert reading.flags[0] == ("disagrees on hazard — check (extraction: described; second reader: unclear, "
                                "confidence 0.40)")
    assert [f for f in reading.flags if "hazard" in f] == [reading.flags[0]]


FAN = "the ceiling fan wobbles and makes a noise"


@pytest.mark.parametrize("jev_mechanism", ["happening now", "could happen", "no hazard"])
def test_fan_like_case_shows_one_hazard_line_and_no_mechanism_line(jev_mechanism: str) -> None:
    fan = facts(fault_description=FAN, hazard_status="none", mechanism_type=None,
                quoted_spans=[{"field": "fault_description", "text": FAN}])
    jev = {"hazard": "unclear", "mechanism": jev_mechanism, "alternative": "no", "fault_or_sign": "fault"}
    reading = read(FakeJev(response(jev, {**dict.fromkeys(AGREE, 0.9), "hazard": 0.48, "mechanism": 0.3})), FAN, fan)
    assert reading.flags == (
        "disagrees on hazard — check (extraction: none; second reader: unclear, confidence 0.48)",
    )


@pytest.mark.parametrize("jev_hazard", ["none", "unclear"])
def test_mechanism_not_reported_when_second_reader_sees_no_pathway(jev_hazard: str) -> None:
    reading = read(FakeJev(response({**AGREE, "hazard": jev_hazard, "mechanism": "could happen"})), "t", facts())
    assert not any("mechanism" in f for f in reading.flags)
    assert [f.split(" — ")[0] for f in reading.flags] == ["disagrees on hazard"]


@pytest.mark.parametrize(("extracted", "jev"), [
    ({"hazard_status": "none", "mechanism_type": None}, {"hazard": "none", "mechanism": "no hazard"}),
    ({"mechanism_type": "conditional"}, {"mechanism": "could happen"}),
    ({"alternative_mentioned": True, "quoted_spans": [
        {"field": "fault_description", "text": "water touching the wire"},
        {"field": "hazard", "text": "water touching the wire"},
        {"field": "alternative_mentioned", "text": "other shower"}]}, {"alternative": "yes"}),
    ({"fault_or_sign": "sign", "quoted_spans": [
        {"field": "fault_description", "text": "water touching the wire"},
        {"field": "hazard", "text": "water touching the wire"},
        {"field": "fault_or_sign", "text": "smells"}]}, {"fault_or_sign": "sensed cue only"}),
])
def test_matching_options_agree(extracted: dict[str, Any], jev: dict[str, str]) -> None:
    assert read(FakeJev(response({**AGREE, **jev})), "text", facts(**extracted)).flags == ()


def test_unclear_hazard_has_no_mechanism_to_compare() -> None:
    unclear = facts(hazard_status="unclear", mechanism_type=None)
    for mechanism in ("happening now", "could happen", "no hazard"):
        assert read(FakeJev(response({**AGREE, "hazard": "unclear", "mechanism": mechanism})), "t", unclear).flags == ()


# --- failures never block ------------------------------------------------------------------

@pytest.mark.parametrize("body", [
    TimeoutError("timed out"),
    {**response(), "answers": {k: v for k, v in response()["answers"].items() if k != "hazard"}},  # missing answer
    response({**AGREE, "hazard": "maybe"}),  # not one of the options
])
def test_error_or_bad_response_is_unavailable(body: Any) -> None:
    reading = read(FakeJev(body), "text", facts())
    assert reading.status == "unavailable" and reading.flags == () and reading.detail


def test_extra_fields_are_ignored() -> None:
    body = response()
    body = {**body, "id": "resp_1", "created": 1760000000, "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {f: {**a, "id": f"a_{f}", "created": 1, "probabilities": {a["choice"]: 0.9}}
                        for f, a in body["answers"].items()}}
    assert read(FakeJev(body), "text", facts()) == SecondReading(status="ran", flags=(), detail=None)


@pytest.mark.parametrize("missing", ["type", "choice", "confidence"])
def test_missing_needed_field_is_unavailable(missing: str) -> None:
    body = response()
    body["answers"]["hazard"].pop(missing)
    assert read(FakeJev(body), "text", facts()).status == "unavailable"


def test_probabilities_are_optional() -> None:
    body = response()
    for a in body["answers"].values():
        a.pop("probabilities")
    assert read(FakeJev(body), "text", facts()).status == "ran"


def test_http_error_is_unavailable_with_the_key_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "ts-secret-123"

    def fail(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 401, f"bad key {key}", {}, io.BytesIO())  # type: ignore[arg-type]
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    reading = read(second_reader.TypeSafeClient(key), "text", facts())
    assert reading.status == "unavailable" and key not in str(reading.detail) and "<redacted>" in str(reading.detail)


def test_request_follows_the_documented_api(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Body(io.BytesIO):
        def __enter__(self) -> "Body":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def capture(request: urllib.request.Request, timeout: float) -> Body:
        seen.update(url=request.full_url, method=request.get_method(), timeout=timeout,
                    auth=request.get_header("Authorization"), body=json.loads(request.data))  # type: ignore[arg-type]
        return Body(json.dumps(response()).encode())
    monkeypatch.setattr(urllib.request, "urlopen", capture)
    assert read(second_reader.TypeSafeClient("k"), "toilet blocked", facts()).status == "ran"
    assert (seen["url"], seen["method"], seen["timeout"], seen["auth"]) == (
        "https://api.typesafe.ai/v1/systemone", "POST", TIMEOUT_S, "Bearer k")
    assert seen["body"] == {"model": "jev-latest", "state": "toilet blocked", "questions": QUESTIONS}
    assert TIMEOUT_S == 30


def test_no_key_means_no_client() -> None:
    assert second_reader.client_from_env() is None  # conftest sets TYPESAFE_API_KEY=""


# --- what Jev sees (invariants 1 and 3) ----------------------------------------------------

def test_every_question_is_a_choice() -> None:
    # Only choice answers carry a confidence in the API, so every field can be checked against 0.7.
    assert {q["type"] for q in QUESTIONS.values()} == {"choice"}
    assert set(QUESTIONS) == {"hazard", "mechanism", "alternative", "fault_or_sign"}


@pytest.mark.parametrize("word", BANNED_WORDS)
def test_questions_have_no_banned_word(word: str) -> None:
    assert word.lower() not in json.dumps(QUESTIONS).lower()


def test_state_is_the_report_or_report_plus_this_fault_only() -> None:
    assert state_for("toilet blocked", "toilet blocked", compound=False) == "toilet blocked"
    assert state_for("wire sparking and gas smell", "wire sparking", compound=True) == {
        "report": "wire sparking and gas smell", "fault": "wire sparking"}


# --- flags only: never changes safety, tally or rank ---------------------------------------

REPORTS = ("toilet blocked", "water's pooling next to the switchboard", "stove's not working",
           "I can smell gas in the kitchen", "wire sparking near the sink and I can smell gas", "the fan wobbles")


def offline_jobs() -> list[tuple[EnrichedJob, ExtractedFacts]]:
    pairs = []
    for i, text in enumerate(REPORTS):
        report = create_report(tenant_id="T", raw_text=text, source_tag="tenant_direct", community="Wadeye",
                               original_report_timestamp=T0)
        extraction = OfflineExtractor.read(text)
        pairs += list(zip(demo._build_jobs(report, extraction), extraction.faults))
    return pairs


answers = st.fixed_dictionaries({f: st.sampled_from(list(q["criteria"])) for f, q in QUESTIONS.items()})  # type: ignore[arg-type]


@given(st.lists(st.tuples(answers, st.floats(0, 1)), min_size=len(REPORTS) * 2, max_size=len(REPORTS) * 2))
def test_second_reader_never_changes_safety_tally_or_rank(drawn: list[tuple[dict[str, str], float]]) -> None:
    pairs = offline_jobs()
    plain = [job for job, _ in pairs]
    read_jobs = [attach(job, read(FakeJev(response(ch, conf)), "text", f))
                 for (job, f), (ch, conf) in zip(pairs, drawn)]
    assert [to_rank_input(j) for j in read_jobs] == [to_rank_input(j) for j in plain]
    assert [(j.safety_level, j.urgency_tally, j.tier, j.flags) for j in read_jobs] == \
           [(j.safety_level, j.urgency_tally, j.tier, j.flags) for j in plain]
    assert rank([to_rank_input(j) for j in read_jobs]) == rank([to_rank_input(j) for j in plain])


# --- coordinator view and tenant text ------------------------------------------------------

def coordinator(reading: SecondReading) -> str:
    job = attach(offline_jobs()[0][0], reading)
    (tr,) = build_traces(rank([to_rank_input(job)]), {job.request_id: job})
    return render_coordinator(tr)


@pytest.mark.parametrize(("reading", "line"), [
    (SecondReading(status="ran", flags=(), detail=None), "agrees"),
    (SecondReading(status="ran", flags=("disagrees on hazard — check (x)",), detail=None), "disagrees on hazard — check (x)"),
    (SecondReading(status="ran", flags=("low confidence on hazard — check (x)",), detail=None),
     "low confidence on hazard — check (x)"),
    (second_reader.not_run("not run (no key)"), "not run (no key)"),
    (second_reader.not_run("not run (offline)"), "not run (offline)"),
    (second_reader.not_run("not run (recorded mode)"), "not run (recorded mode)"),
    (SecondReading(status="unavailable", flags=(), detail="TimeoutError: timed out"), "unavailable (TimeoutError: timed out)"),
])
def test_coordinator_view_shows_the_second_reader_line(reading: SecondReading, line: str) -> None:
    rows = [r for r in coordinator(reading).splitlines() if r.startswith("second_reader")]
    assert len(rows) == 1 and line in rows[0]


def test_review_entry_shows_the_second_reader_line() -> None:
    job = next(j for j, _ in offline_jobs() if j.in_review_band)
    assert "second_reader" in render_review_entry(attach(job, second_reader.not_run("not run (no key)")))


TENANT_BANNED = ("reader", "jev", "confidence", "typesafe", "disagree")


def test_tenant_text_never_mentions_the_second_reader() -> None:
    noisy = SecondReading(status="ran", flags=("disagrees on hazard — check (extraction: described; second reader: "
                                               "none, confidence 0.40)",), detail=None)
    plain_jobs = [j for j, _ in offline_jobs()]
    for job in plain_jobs:
        read_job = attach(job, noisy)
        for render in (tenant_sms, lambda j: tenant_why(j, pinned=False, classified_by_coordinator=False), lambda j: tenant_why(j, pinned=True, classified_by_coordinator=True)):
            text = render(read_job)
            assert text == render(job)
            assert not any(word in text.lower() for word in TENANT_BANNED), text
    report = create_report(tenant_id="T", raw_text=COMPOUND_TEXT, source_tag="tenant_direct", community="Wadeye",
                           original_report_timestamp=T0)
    siblings = demo._build_jobs(report, compound_extraction())
    read_siblings = [attach(j, noisy) for j in siblings]
    assert len(siblings) == 2 and tenant_sms_report(read_siblings) == tenant_sms_report(siblings)
    assert not any(word in tenant_sms_report(read_siblings).lower() for word in TENANT_BANNED)


# --- demo wiring ---------------------------------------------------------------------------

def write_report(folder: Path, message: str) -> Path:
    folder.mkdir()
    (folder / "R1.txt").write_text(f"Tenant ID: T-1\nCommunity: Darwin\nSource: tenant_direct\n"
                                   f"Reported: 2026-09-21 15:00\nMessage: {message}\n", encoding="utf-8")
    return folder


class FakeLive(OfflineExtractor):
    name = "llm:fake"
    model = "gpt-4o-fake"


COMPOUND_TEXT = "wire sparking near the sink and I can smell gas"


def compound_extraction() -> ReportExtraction:
    def one(words: str, name: str) -> ExtractedFacts:
        return facts(fault_description=words, taxonomy_match=[name], quoted_spans=[
            {"field": "fault_description", "text": words}, {"field": "taxonomy_match", "text": words},
            {"field": "hazard", "text": words}])
    return ReportExtraction(faults=(one("wire sparking", "exposed electrical wires"), one("smell gas", "gas leak")))


class FakeCompoundLive(FakeLive):
    """A live extractor that reads two faults in COMPOUND_TEXT."""

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        return compound_extraction().model_dump_json()


def calls_to_jev(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any] | Exception) -> FakeJev:
    fake = FakeJev(body)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test-key")
    monkeypatch.setattr(second_reader, "TypeSafeClient", lambda key: fake)
    return fake


def test_offline_never_calls_jev_even_with_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    fake = calls_to_jev(monkeypatch, response())
    demo.main([str(write_report(tmp_path / "r", "toilet blocked")), "--offline"])
    out = capsys.readouterr().out
    assert fake.states == [] and "SECOND READER: not run (offline)" in out and "not run (offline)  flag only" in out


def test_recorded_mode_never_calls_jev_even_with_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(demo, "RECORDED_DIR", tmp_path / "recorded")
    fake = calls_to_jev(monkeypatch, response())
    demo.main([str(write_report(tmp_path / "r", "toilet blocked"))])
    out = capsys.readouterr().out
    assert out.startswith("MODE: recorded") and fake.states == [] and "SECOND READER: not run (recorded mode)" in out


def test_live_without_jev_key_is_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(demo, "OpenAICompatibleClient", FakeLive)
    demo.main([str(write_report(tmp_path / "r", "toilet blocked"))])
    assert "not run (no key)  flag only" in capsys.readouterr().out


def test_live_asks_per_fault_with_the_fault_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(demo, "OpenAICompatibleClient", FakeCompoundLive)
    fake = calls_to_jev(monkeypatch, response())
    demo.main([str(write_report(tmp_path / "r", COMPOUND_TEXT))])
    assert fake.states == [{"report": COMPOUND_TEXT, "fault": "wire sparking"},
                           {"report": COMPOUND_TEXT, "fault": "smell gas"}]
    assert "second_reader" in capsys.readouterr().out


def test_jev_down_still_ranks_every_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(demo, "OpenAICompatibleClient", FakeLive)
    calls_to_jev(monkeypatch, TimeoutError("timed out"))
    demo.main([str(write_report(tmp_path / "r", "toilet blocked"))])
    out = capsys.readouterr().out
    ranked = out.split("RANKED QUEUE")[1]
    assert "#1" in ranked and "unavailable (TimeoutError: timed out)" in ranked


def test_probe_script_exits_1_without_a_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import importlib
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    probe = importlib.import_module("probe_jev")
    monkeypatch.setattr(probe, "_load_dotenv", lambda *a, **k: None)  # never read the real .env
    assert probe.main() == 1
    assert "nothing sent" in capsys.readouterr().err
