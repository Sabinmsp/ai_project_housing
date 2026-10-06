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
