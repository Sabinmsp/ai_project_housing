"""Recorded model responses for extraction: replay saved real answers with no API key, or record new ones.

Input: a report's text (via the user prompt). Output: the raw JSON the model gave for it,
read from or written to data/recorded/. Both clients sit behind extract(), so a recorded
answer goes through the same _parse and validation as a live one.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

from triage import extraction
from triage.extraction import (SYSTEM_PROMPT, LLMClient, configured_model, report_text_from_prompt,
                               response_schema)

RECORDED_DIR = Path(__file__).resolve().parent.parent / "data" / "recorded"
MISS = "Needs the live model: set OPENAI_API_KEY"
_FIELDS = ("model", "recorded_at", "prompt_hash", "response")


class RecordingMissing(ValueError):
    """No usable recording for this report. A ValueError, so extract() flags it for a human."""


def prompt_hash(model: str | None = None) -> str:
    """Fingerprint of the model and everything it sees besides the report itself."""
    parts = [
        # A different model gives a different answer to the same prompt.
        model or configured_model(),
        SYSTEM_PROMPT,
        extraction.build_user_prompt("REPORT_TEXT"),  # the message template, with a fixed placeholder
        # Read at call time and kept in order: a renamed or reordered fault list is a different question.
        json.dumps(list(extraction.FAULT_NAMES)),
        json.dumps(response_schema(), sort_keys=True),
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def recording_path(raw_text: str, folder: Path) -> Path:
    """One file per exact report text, named by its SHA-256."""
    return folder / f"{hashlib.sha256(raw_text.encode('utf-8')).hexdigest()}.json"


def _read(path: Path) -> dict | None:
    """The recording at path, or None if it is not valid JSON with every field as a string."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or not all(isinstance(record.get(k), str) for k in _FIELDS):
        return None
    return record


class RecordedClient:
    """Replays a saved response for the exact report text; a miss raises, never guesses."""

    def __init__(self, folder: Path = RECORDED_DIR) -> None:
        self.folder = folder
        self.model = configured_model()
        self.current = prompt_hash(self.model)
        # A stray or corrupt file is skipped here and flagged when its report is looked up.
        records = (_read(p) for p in sorted(self.folder.glob("*.json")))
        usable = [r for r in records if r and r["prompt_hash"] == self.current]
        self.recorded_on = max((r["recorded_at"] for r in usable), default=None)
        self.name = f"recorded:{self.model}"

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """The recorded answer for this report text.

        Raises:
            RecordingMissing: no file, an unreadable file, or one from another prompt_hash.
        """
        path = recording_path(report_text_from_prompt(user), self.folder)
        if not path.exists():
            raise RecordingMissing(MISS)
        record = _read(path)
        if record is None:
            raise RecordingMissing(f"{MISS} (recording unreadable: {path.name})")
        # A response recorded under another prompt answers a different question.
        if record["prompt_hash"] != self.current:
            raise RecordingMissing(f"{MISS} (recorded under an older prompt)")
        return record["response"]


class RecordingClient:
    """A live client that also saves every response it returns, one file per report text."""

    def __init__(self, live: LLMClient, folder: Path = RECORDED_DIR) -> None:
        self.live = live
        self.folder = folder
        self.model = getattr(live, "model", live.name)
        self.name = f"recording:{live.name}"

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """Call the live client, save its raw answer, and return it unchanged."""
        raw = self.live.complete_json(system, user, schema)
        self.folder.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "recorded_at": date.today().isoformat(),
                  "prompt_hash": prompt_hash(self.model), "response": raw}
        # On a retry this overwrites the first attempt, so the file holds the last answer given.
        recording_path(report_text_from_prompt(user), self.folder).write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return raw
