import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import HealthCheck, settings

# Property tests build queues of up to 40 jobs; on a slow machine Hypothesis's
# input-speed health check can fail a run that would otherwise pass.
settings.register_profile("triage", suppress_health_check=[HealthCheck.too_slow], deadline=None)
settings.load_profile("triage")


import openai
import pytest

import demo


@pytest.fixture(autouse=True)
def no_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reads .env or builds the real API client; a test that needs "live" swaps in a fake."""
    monkeypatch.setattr(demo, "_load_dotenv", lambda *args, **kwargs: None)
    for name in ("OPENAI_API_KEY", "TRIAGE_API_KEY"):
        monkeypatch.setenv(name, "")
    # A developer's shell settings must not change the recording hash or the client a test sees.
    for name in ("TRIAGE_MODEL", "TRIAGE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("real API client constructed in a test")
    # The SDK constructor itself: every path to a paid call (demo, extraction, probe script) goes through it.
    monkeypatch.setattr(openai, "OpenAI", refuse)
