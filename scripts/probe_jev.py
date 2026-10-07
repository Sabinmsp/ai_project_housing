"""Probe the second reader (Jev) once on the extraction probe's samples; print each answer.

Input: SAMPLES from probe_llm.py. Output: one line per sample and question on stdout.
Manual tool only: it calls the paid TypeSafe API, so no test or CI runs it.
Usage: python scripts/probe_jev.py   (needs TYPESAFE_API_KEY in the environment or .env)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo import _load_dotenv  # noqa: E402  same .env loader as the demo
from probe_llm import SAMPLES  # noqa: E402  the same seven samples as the extraction probe
from triage.second_reader import LOW_CONFIDENCE, QUESTIONS, _validate, client_from_env  # noqa: E402


def main() -> int:
    """Ask Jev about every sample once; 1 without a key."""
    _load_dotenv()
    client = client_from_env()
    if client is None:
        print("TYPESAFE_API_KEY not set (environment or .env); nothing sent.", file=sys.stderr)
        return 1
    print(f"Jev probe: {len(SAMPLES)} samples, paid API calls. Low confidence: < {LOW_CONFIDENCE} (uncalibrated).")
    for sample in SAMPLES:
        print(f"\n{sample!r}")
        try:
            response = _validate(client.ask(sample, QUESTIONS))
        except Exception as e:  # report and carry on with the next sample
            print(f"  ERROR {client.redact(f'{e.__class__.__name__}: {e}')[:200]}")
            continue
        for field, answer in response.answers.items():
            low = "  LOW" if answer.confidence < LOW_CONFIDENCE else ""
            print(f"  {field:14} {answer.choice:16} confidence {answer.confidence:.2f}{low}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
