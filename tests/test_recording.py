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
