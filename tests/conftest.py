import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Before app.server is imported: its default database path is read once, at import. A test that
# builds Workspace() without a db_path must never write the real data/fairfix.db.
os.environ["FAIRFIX_DB"] = ":memory:"

from hypothesis import HealthCheck, settings

# Property tests build queues of up to 40 jobs; on a slow machine Hypothesis's
# input-speed health check can fail a run that would otherwise pass.
settings.register_profile("triage", suppress_health_check=[HealthCheck.too_slow], deadline=None)
settings.load_profile("triage")


import urllib.request

import openai
import pytest

import app.server
import demo


@pytest.fixture(autouse=True)
def no_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reads .env or builds the real API client; a test that needs "live" swaps in a fake."""
    # Neither loader may read the real .env: it holds a key and TRIAGE_LIVE=1, and setdefault
    # into os.environ would outlive the test.
    monkeypatch.setattr(demo, "_load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.server, "_load_dotenv", lambda *args, **kwargs: None)
    for name in ("OPENAI_API_KEY", "TRIAGE_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.setenv(name, "")
    # A developer's shell settings must not change the recording hash, the mode or the client a test sees.
    for name in ("TRIAGE_MODEL", "TRIAGE_BASE_URL", "TRIAGE_LIVE", "TRIAGE_OFFLINE_FALLBACK"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("real API client constructed in a test")
    # The SDK constructor itself: every path to a paid call (demo, extraction, probe script) goes through it.
    monkeypatch.setattr(openai, "OpenAI", refuse)
    # The second reader's only network call goes through urlopen; no test may reach Jev.
    # BaseException: the second reader turns any Exception into "unavailable", which would
    # hide a real call behind a passing test.
    def refuse_network(*args: object, **kwargs: object) -> None:
        raise NetworkCallInTest("real network call in a test")
    monkeypatch.setattr(urllib.request, "urlopen", refuse_network)
    # Every other path (httpx, requests, a raw socket) ends in a socket connect or a DNS lookup.
    # Unix sockets stay allowed; nothing in the suite needs them, but they never leave the machine.
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded(real):
        def connect(self: socket.socket, address: object) -> object:
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise NetworkCallInTest(f"real network call in a test: connect to {address!r}")
            return real(self, address)
        return connect
    monkeypatch.setattr(socket.socket, "connect", guarded(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guarded(real_connect_ex))
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)


class NetworkCallInTest(BaseException):
    """Raised by the autouse guard; not an Exception, so no handler can swallow it."""
