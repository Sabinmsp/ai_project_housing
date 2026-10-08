"""The autouse guard in conftest.py: no test reads .env, reaches the network or the real database."""
import os
import socket
import urllib.request

import pytest

import app.server as server
from tests.conftest import NetworkCallInTest


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_a_socket_connection_in_a_test_is_refused(method):
    # A raw socket and a numeric address: no DNS lookup, so only the connect guard can stop it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(NetworkCallInTest):
        getattr(s, method)(("127.0.0.1", 9))


def test_a_dns_lookup_and_urlopen_in_a_test_are_refused():
    with pytest.raises(NetworkCallInTest):
        socket.getaddrinfo("api.openai.com", 443)
    with pytest.raises(NetworkCallInTest):
        urllib.request.urlopen("https://api.typesafe.ai")


def test_the_default_workspace_never_reads_env_never_goes_live_and_never_uses_the_real_db():
    # The repo's .env holds a key and TRIAGE_LIVE=1. A Workspace built with no arguments, as
    # ws() does, must still be offline-safe in a test.
    w = server.Workspace()
    assert w.client.live is None
    assert w.db_path == ":memory:"
    assert "TRIAGE_LIVE" not in os.environ and not os.environ.get("OPENAI_API_KEY")
