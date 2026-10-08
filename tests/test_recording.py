import pytest

from triage import extraction, recording


def test_prompt_hash_is_stable() -> None:
    assert recording.prompt_hash() == recording.prompt_hash()


@pytest.mark.parametrize("names", [
    (*extraction.FAULT_NAMES, "new fault"),           # added
    extraction.FAULT_NAMES[:-1],                      # removed
    tuple(reversed(extraction.FAULT_NAMES)),          # same names, different order
])
def test_changing_fault_names_changes_prompt_hash(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> None:
    before = recording.prompt_hash()
    monkeypatch.setattr(extraction, "FAULT_NAMES", names)
    assert recording.prompt_hash() != before


def test_changing_the_message_template_changes_prompt_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    before = recording.prompt_hash()
    original = extraction.build_user_prompt
    monkeypatch.setattr(extraction, "build_user_prompt", lambda raw_text: "SOURCE: x\n" + original(raw_text))
    assert recording.prompt_hash() != before


def test_changing_the_system_prompt_changes_prompt_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    before = recording.prompt_hash()
    # recording imports SYSTEM_PROMPT by name, so that is the binding prompt_hash reads.
    monkeypatch.setattr(recording, "SYSTEM_PROMPT", recording.SYSTEM_PROMPT + " ")
    assert recording.prompt_hash() != before


def test_changing_the_response_schema_changes_prompt_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    before = recording.prompt_hash()
    original = recording.response_schema
    monkeypatch.setattr(recording, "response_schema", lambda: {**original(), "title": "Other"})
    assert recording.prompt_hash() != before


def test_changing_triage_model_changes_prompt_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    before = recording.prompt_hash()
    monkeypatch.setenv("TRIAGE_MODEL", "gpt-4o-mini")
    assert recording.prompt_hash() != before
    assert recording.prompt_hash() == recording.prompt_hash("gpt-4o-mini")


def test_empty_triage_model_hashes_as_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    before = recording.prompt_hash()
    monkeypatch.setenv("TRIAGE_MODEL", "")
    assert recording.prompt_hash() == before == recording.prompt_hash(extraction.DEFAULT_MODEL)


RECORDED_MODEL = "anthropic/claude-sonnet-5.5"  # the model data/recorded/ was saved with


def test_the_saved_answers_still_match_the_current_recording_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Coordinator-only repair types must not reach FAULT_NAMES: if they did, the prompt (and so
    # this key) would change and none of the saved answers would replay.
    import json
    saved = {json.loads(p.read_text())["prompt_hash"] for p in recording.RECORDED_DIR.glob("*.json")}
    assert saved == {recording.prompt_hash(RECORDED_MODEL)}


# Reports with no saved answer, by (file, form row). Only this one: its answer was cut off
# mid-string, so it was deleted rather than replayed. It needs re-recording with RECORDED_MODEL.
KNOWN_MISSING = {("05_Top-End_Wurrumiyanga_roof-leak.pdf", 2)}


def test_every_recorded_report_still_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    from triage.extraction import extract
    from triage.models import ExtractionStatus
    from triage.report_files import load_reports
    monkeypatch.setenv("TRIAGE_MODEL", RECORDED_MODEL)
    client = recording.RecordedClient(recording.RECORDED_DIR)
    root = Path(recording.__file__).resolve().parent.parent
    reports = [r for folder in ("reports", "pdf") for r in load_reports(root / folder)[0]]
    assert len(reports) >= 20
    missing = set()
    for report in reports:
        result = extract(report, client)
        if any("RecordingMissing" in e for e in result.errors):
            missing.add((report.source_file, report.source_item))
        else:
            # Found under today's key and valid: read, or read as naming no fault.
            assert result.status in (ExtractionStatus.OK, ExtractionStatus.NO_FAULT_NAMED), (report.source_file, result.errors)
    assert missing == KNOWN_MISSING
