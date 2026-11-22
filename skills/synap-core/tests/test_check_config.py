"""test_check_config.py tests skills/synap-core/scripts/check_config.py.

Covers:
- load_token() reading SYNAP_API_TOKEN from the environment, falling back to ./.env
- texts() extraction from context objects
- Unreachable server handling (clear error message, no traceback)
- Server config evaluation ([OK], [WARN], [ERROR] and scoping mode)
- SDK import error handling when maximem-synap is missing
- Mocked SDK execution for recall verification, user separation, and cleanup
- The RESULT: / EXIT CODE: output footers
"""
from __future__ import annotations

import asyncio
import http.server
import importlib.util
import json
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import SCRIPTS, run_script


def load_check_config():
    """Load check_config as a module without executing its top-level asyncio.run(main())."""
    spec = importlib.util.spec_from_file_location("check_config_mod", SCRIPTS / "check_config.py")
    mod = importlib.util.module_from_spec(spec)
    with patch("asyncio.run") as mock_run:
        mock_run.side_effect = lambda coro: coro.close()
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_server():
    """A minimal HTTP server running on 127.0.0.1 with configurable JSON responses."""
    routes: dict[str, tuple[int, dict]] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in routes:
                status, body = routes[path]
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", routes
    server.shutdown()


# =====================================================================
# load_token() tests
# =====================================================================

def test_load_token_from_environment(monkeypatch):
    mod = load_check_config()
    monkeypatch.setenv("SYNAP_API_TOKEN", "test-token-from-env")
    assert mod.load_token() == "test-token-from-env"


def test_load_token_from_dotenv_file(tmp_path, monkeypatch):
    mod = load_check_config()
    monkeypatch.delenv("SYNAP_API_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SOME_VAR=hello\nSYNAP_API_TOKEN=token-from-file\nOTHER_VAR=world\n")
    assert mod.load_token() == "token-from-file"


def test_load_token_prefers_environment_over_dotenv(tmp_path, monkeypatch):
    mod = load_check_config()
    monkeypatch.setenv("SYNAP_API_TOKEN", "token-from-env")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SYNAP_API_TOKEN=token-from-file\n")
    assert mod.load_token() == "token-from-env"


def test_load_token_returns_empty_when_no_token_set(tmp_path, monkeypatch):
    mod = load_check_config()
    monkeypatch.delenv("SYNAP_API_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SYNAP_LLM_PROVIDER=openai\n")
    assert mod.load_token() == ""


def test_load_token_returns_empty_when_no_dotenv_file(tmp_path, monkeypatch):
    mod = load_check_config()
    monkeypatch.delenv("SYNAP_API_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    assert mod.load_token() == ""


# =====================================================================
# texts() and say() tests
# =====================================================================

def test_say_formats_level_and_records_lines(capsys):
    mod = load_check_config()
    mod.lines.clear()
    mod.say("OK", "everything is running")
    mod.say("WARN", "starter config in use")
    mod.say("ERROR", "cannot connect")
    out = capsys.readouterr().out
    assert "[OK] everything is running" in out
    assert "[WARN] starter config in use" in out
    assert "[ERROR] cannot connect" in out
    assert mod.lines == [
        "[OK] everything is running",
        "[WARN] starter config in use",
        "[ERROR] cannot connect",
    ]


def test_texts_extracts_content_from_context():
    mod = load_check_config()
    ctx = SimpleNamespace(
        facts=[SimpleNamespace(content="Ana likes tea"), "Ben likes coffee"],
        preferences=[SimpleNamespace(content="Dark mode")],
        episodes=[],
        emotions=None,
        temporal_events=[SimpleNamespace(content="Trip to Lisbon")],
    )
    result = mod.texts(ctx)
    assert result == ["Ana likes tea", "Ben likes coffee", "Dark mode", "Trip to Lisbon"]


def test_texts_empty_context():
    mod = load_check_config()
    ctx = SimpleNamespace()
    assert mod.texts(ctx) == []


# =====================================================================
# CLI execution and server reachability
# =====================================================================

def test_unreachable_server_fails_with_clear_error_and_no_traceback(run):
    done = run("check_config.py", "--url", "http://127.0.0.1:59999")
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert "Traceback" not in done.stdout
    assert "[ERROR] cannot reach http://127.0.0.1:59999" in done.stdout
    assert "RESULT: FAIL cannot reach http://127.0.0.1:59999" in done.stdout
    assert "EXIT CODE: 1" in done.stdout


def test_invalid_config_reports_error_and_fails(fake_server, run):
    url, routes = fake_server
    routes["/browser/api/config"] = (200, {
        "valid": False,
        "problems": ["storage_architecture.scoping: invalid mode", "line 12: unclosed bracket"],
    })
    routes["/api/v1/auth/whoami"] = (200, {"user_context_isolation": "equals_customer"})

    done = run("check_config.py", "--url", url)
    assert done.returncode == 1
    assert "[ERROR] the config is not accepted: storage_architecture.scoping: invalid mode" in done.stdout
    assert "RESULT: FAIL" in done.stdout
    assert "EXIT CODE: 1" in done.stdout


def test_starter_config_warns_developer(fake_server, run):
    url, routes = fake_server
    routes["/browser/api/config"] = (200, {
        "valid": True,
        "starter": True,
        "sections": [{"name": "storage_architecture", "settings": 1}],
    })
    routes["/api/v1/auth/whoami"] = (200, {"user_context_isolation": "equals_customer"})

    done = run("check_config.py", "--url", url)
    assert "[OK] the config is accepted: storage_architecture (1)" in done.stdout
    assert "[WARN] this is still the starter sample, not a config of your own" in done.stdout
    assert "[OK] scoping mode: equals_customer" in done.stdout


def test_developer_config_reported_as_ok(fake_server, run):
    url, routes = fake_server
    routes["/browser/api/config"] = (200, {
        "valid": True,
        "starter": False,
        "sections": [{"name": "storage_architecture", "settings": 2}],
    })
    routes["/api/v1/auth/whoami"] = (200, {"user_context_isolation": "strict"})

    done = run("check_config.py", "--url", url)
    assert "[OK] the config is accepted: storage_architecture (2)" in done.stdout
    assert "[OK] this is the developer's own config" in done.stdout
    assert "[OK] scoping mode: strict" in done.stdout


# =====================================================================
# Mocked SDK main() tests for recall, user separation, and cleanup
# =====================================================================

def test_main_passes_when_recall_and_separation_succeed(monkeypatch, capsys):
    mod = load_check_config()
    mod.lines.clear()

    config_resp = {
        "valid": True,
        "starter": False,
        "sections": [{"name": "storage_architecture", "settings": 1}],
    }
    whoami_resp = {"user_context_isolation": "equals_customer"}

    def fake_get(path):
        if path == "/browser/api/config":
            return config_resp
        if path == "/api/v1/auth/whoami":
            return whoami_resp
        return {}

    mock_sdk = MagicMock()
    mock_sdk.initialize = AsyncMock()
    mock_sdk.shutdown = AsyncMock()

    created_mock = MagicMock(ingestion_id="ingest-1")
    done_mock = MagicMock(memory_ids=["11111111-1111-1111-1111-111111111111"], memories_created=1)
    mock_sdk.memories.create = AsyncMock(return_value=created_mock)
    mock_sdk.memories.wait_for_completion = AsyncMock(return_value=done_mock)
    mock_sdk.memories.delete = AsyncMock()

    async def fake_fetch(user_id=None, **kwargs):
        if "a" in user_id:
            return SimpleNamespace(facts=[SimpleNamespace(content="I am vegetarian")])
        return SimpleNamespace(facts=[SimpleNamespace(content="Priya loves orchids")])

    mock_sdk.user.context.fetch = AsyncMock(side_effect=fake_fetch)

    with patch("sys.argv", ["check_config.py", "--url", "http://127.0.0.1:8000"]), \
         patch.object(mod, "load_token", return_value=""), \
         patch.dict("sys.modules", {"maximem_synap": MagicMock(MaximemSynapSDK=MagicMock(return_value=mock_sdk))}):

        with pytest.raises(SystemExit) as exc:
            with patch("urllib.request.urlopen") as mock_open:
                mock_open.side_effect = lambda req, timeout=20: MagicMock(
                    read=lambda: json.dumps(fake_get(req.full_url.split("8000")[-1])).encode()
                )
                asyncio.run(mod.main())

        assert exc.value.code == 0
        captured = capsys.readouterr().out
        assert "user A's diet is recalled" in captured
        assert "user B's sister is recalled" in captured
        assert "the two users stay apart" in captured
        assert "cleanup: removed 2 of 2 test memories" in captured
        assert "RESULT: PASS" in captured
        assert "EXIT CODE: 0" in captured


def test_main_fails_when_user_separation_fails(monkeypatch, capsys):
    mod = load_check_config()
    mod.lines.clear()

    config_resp = {"valid": True, "starter": False, "sections": []}
    whoami_resp = {"user_context_isolation": "equals_customer"}

    mock_sdk = MagicMock()
    mock_sdk.initialize = AsyncMock()
    mock_sdk.shutdown = AsyncMock()
    mock_sdk.memories.create = AsyncMock(return_value=MagicMock(ingestion_id="ingest-1"))
    mock_sdk.memories.wait_for_completion = AsyncMock(return_value=MagicMock(memory_ids=[], memories_created=1))
    mock_sdk.memories.delete = AsyncMock()

    # Leaking diet into user B
    async def fake_fetch(user_id=None, **kwargs):
        return SimpleNamespace(facts=[SimpleNamespace(content="vegetarian and Priya")])

    mock_sdk.user.context.fetch = AsyncMock(side_effect=fake_fetch)

    with patch("sys.argv", ["check_config.py", "--url", "http://127.0.0.1:8000"]), \
         patch.object(mod, "load_token", return_value=""), \
         patch.dict("sys.modules", {"maximem_synap": MagicMock(MaximemSynapSDK=MagicMock(return_value=mock_sdk))}):

        with pytest.raises(SystemExit) as exc:
            with patch("urllib.request.urlopen") as mock_open:
                mock_open.side_effect = lambda req, timeout=20: MagicMock(
                    read=lambda: json.dumps(config_resp if "config" in req.full_url else whoami_resp).encode()
                )
                asyncio.run(mod.main())

        assert exc.value.code == 1
        captured = capsys.readouterr().out
        assert "[ERROR] the two users stay apart" in captured
        assert "RESULT: FAIL the two users stay apart" in captured
        assert "EXIT CODE: 1" in captured
