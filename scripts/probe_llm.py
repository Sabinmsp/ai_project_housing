"""Probe the real LLM extractor on fixed samples and save every response to scratch/probe/.

Exercises extraction and verification exactly as extract() would, outside the pipeline.
Input: SAMPLES below. Output: a table on stdout and one JSON file per run.
Manual tool only: it calls the paid API, so no test or CI runs it.
Usage: python scripts/probe_llm.py [--runs N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo import _load_dotenv  # noqa: E402  same .env loader as the demo
from triage.extraction import (  # noqa: E402
    SYSTEM_PROMPT,
    OpenAICompatibleClient,
    _parse,
    build_user_prompt,
    response_schema,
)
from triage.tiers import FAULT_NAMES  # noqa: E402
from triage.verification import verify_spans  # noqa: E402

# None of these come from reports/.
SAMPLES = (
    "dunny won't go down",
    "I can smell gas in the kitchen",
    "water's pooling next to the switchboard",
    "wire's hanging near the sink, hasn't touched water yet",
    "wire sparking near the sink and I can smell gas",
    "stove's not working",
    "toilet blocked",
)
OUT_DIR = ROOT / "scratch" / "probe"


def _api_key() -> str | None:
    """TRIAGE_API_KEY or OPENAI_API_KEY; an empty value counts as no key."""
    return os.environ.get("TRIAGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or None


def _redact(text: str, key: str) -> str:
    """text with the API key replaced by <redacted>."""
    # Provider errors can echo part of the key; never let it reach the screen or scratch/.
    return text.replace(key, "<redacted>") if key else text


def probe_once(client: OpenAICompatibleClient, sample: str, key: str) -> dict:
    """One extraction exactly as extract() makes it, kept raw so failures can be inspected."""
    record: dict = {"sample": sample, "model": client.model, "raw": None, "parsed": None,
                    "error": None, "unverified": []}
    try:
        raw = client.complete_json(SYSTEM_PROMPT, build_user_prompt(sample), response_schema())
    except Exception as e:  # network or provider error: record it and carry on with the next run
        record["error"] = _redact(f"{e.__class__.__name__}: {e}", key)
        return record
    try:
        record["raw"] = json.loads(raw)
    except json.JSONDecodeError:
        record["raw"] = raw
    try:
        extraction = _parse(raw, FAULT_NAMES)
    except ValueError as e:  # pydantic's ValidationError is a ValueError
        record["error"] = _redact(f"{e.__class__.__name__}: {e}", key)
        return record
    record["parsed"] = extraction.model_dump(mode="json")
    # The pipeline's span verification, per fault.
    record["unverified"] = [
        [i, field, text] for i, f in enumerate(extraction.faults) for field, text in sorted(verify_spans(sample, f))
    ]
    return record


def table_rows(index: int, run: int, record: dict) -> list[str]:
    """One summary row per extracted fault, or one ERROR / no-fault row."""
    head = f"{index:<2} {run:<3}"
    if record["error"]:
        return [f"{head} {'-':<2} ERROR {' '.join(record['error'].split())[:110]}"]  # full text is in the file
    faults = record["parsed"]["faults"]
    if not faults:
        return [f"{head} 0  (no fault named)"]
    rows = []
    for i, f in enumerate(faults):
        unverified = sum(1 for fault_index, _, _ in record["unverified"] if fault_index == i)
        alt_cope = f"{'alt' if f['alternative_mentioned'] else '-'}/{'cope' if f['coping_mentioned'] else '-'}"
        rows.append(
            f"{head if i == 0 else ' ' * len(head)} {len(faults) if i == 0 else '':<2} "
            f"{'; '.join(f['taxonomy_match']) or '(none)':<38} {f['fault_or_sign']:<5} {f['hazard_status']:<9} "
            f"{f['mechanism_type'] or '-':<11} {str(f['harm_claimed']):<5} {alt_cope:<8} {unverified}"
        )
    return rows


def main() -> int:
    """Run every sample --runs times; returns 1 when there is no key or no openai package."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3, help="runs per sample (default 3)")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    key = _api_key()
    if key is None:
        print("No API key: set OPENAI_API_KEY (or TRIAGE_API_KEY) in .env or the environment. Nothing was sent.")
        return 1
    try:
        client = OpenAICompatibleClient()
    except ImportError:
        print("The openai package is not installed: pip install openai")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"model {client.model}; samples:")
    for index, sample in enumerate(SAMPLES):
        print(f"  {index}: {sample}")
    print(f"\n{'#':<2} {'run':<3} {'n':<2} {'taxonomy_match':<38} {'f/s':<5} {'hazard':<9} "
          f"{'mechanism':<11} {'harm':<5} {'alt/cope':<8} unverified")
    for index, sample in enumerate(SAMPLES):
        for run in range(1, args.runs + 1):
            record = probe_once(client, sample, key)
            (OUT_DIR / f"{index}_{run}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            print("\n".join(table_rows(index, run, record)))
    print(f"\nSaved to {OUT_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
