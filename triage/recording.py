"""Recorded model responses: replay saved real answers without an API key, or record new ones.

Both clients sit behind extract(), so a recorded response goes through the same _parse and
validation as a live one; nothing recorded is trusted blindly.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

from triage import extraction
from triage.extraction import SYSTEM_PROMPT, LLMClient, report_text_from_prompt, response_schema

RECORDED_DIR = Path(__file__).resolve().parent.parent / "data" / "recorded"
MISS = "Needs the live model: set OPENAI_API_KEY"


class RecordingMissing(ValueError):
    """No usable recording for this report. A ValueError, so extract() flags it for a human."""


def prompt_hash() -> str:
    """Fingerprint of everything the model sees besides the report itself."""
    parts = [
        SYSTEM_PROMPT,
        extraction.build_user_prompt("REPORT_TEXT"),  # the message template, with a fixed placeholder
        # Read at call time and kept in order: a renamed or reordered fault list is a different question.
        json.dumps(list(extraction.FAULT_NAMES)),
        json.dumps(response_schema(), sort_keys=True),
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def recording_path(raw_text: str, folder: Path) -> Path:
    return folder / f"{hashlib.sha256(raw_text.encode('utf-8')).hexdigest()}.json"


class RecordedClient:
    """Replays a saved response for the exact report text; a miss raises, never guesses."""

    def __init__(self, folder: Path = RECORDED_DIR) -> None:
        self.folder = folder
        self.current = prompt_hash()
        usable = [r for r in self._all() if r["prompt_hash"] == self.current]
        self.model = usable[0]["model"] if usable else "gpt-4o"
        self.recorded_on = max((r["recorded_at"] for r in usable), default=None)
        self.name = f"recorded:{self.model}"

    def _all(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.folder.glob("*.json"))]

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        path = recording_path(report_text_from_prompt(user), self.folder)
        if not path.exists():
            raise RecordingMissing(MISS)
        record = json.loads(path.read_text(encoding="utf-8"))
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
        raw = self.live.complete_json(system, user, schema)
        self.folder.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "recorded_at": date.today().isoformat(),
                  "prompt_hash": prompt_hash(), "response": raw}
        # On a retry this overwrites the first attempt, so the file holds the last answer given.
        recording_path(report_text_from_prompt(user), self.folder).write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return raw
