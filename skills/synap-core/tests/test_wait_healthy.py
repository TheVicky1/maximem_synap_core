"""test_wait_healthy.py tests skills/synap-core/scripts/wait_healthy.py.

Covers:
- parse_ps(): JSON array format (old Compose), line-delimited JSON format (new Compose), empty/malformed input
- health(): 200 OK, HTTP errors, connection failures
- main():
  - Exit 0: server responds 200 OK within timeout -> RESULT: PASS, EXIT CODE: 0
  - Exit 1: container restarting in a loop or failed -> RESULT: FAIL service <name> failed, EXIT CODE: 1
  - Exit 1: first-start service failure includes helpful guidance
  - Exit 2: timeout expires before healthy -> RESULT: FAIL timed out, EXIT CODE: 2
"""
from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from conftest import SCRIPTS


def load_wait_healthy():
    spec = importlib.util.spec_from_file_location("wait_healthy_mod", SCRIPTS / "wait_healthy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def wait_mod():
    return load_wait_healthy()


# =====================================================================
# parse_ps() tests
# =====================================================================

def test_parse_ps_empty_input(wait_mod):
    assert wait_mod.parse_ps("") == []
    assert wait_mod.parse_ps("   \n  ") == []


def test_parse_ps_json_array(wait_mod):
    text = '[{"Service": "synap", "State": "running"}, {"Service": "postgres", "State": "running"}]'
    rows = wait_mod.parse_ps(text)
    assert len(rows) == 2
    assert rows[0]["Service"] == "synap"
    assert rows[1]["Service"] == "postgres"


def test_parse_ps_line_delimited_json(wait_mod):
    text = '{"Service": "synap", "State": "running"}\n{"Service": "redis", "State": "running"}\n'
    rows = wait_mod.parse_ps(text)
    assert len(rows) == 2
    assert rows[0]["Service"] == "synap"
    assert rows[1]["Service"] == "redis"


def test_parse_ps_malformed_json_does_not_crash(wait_mod):
    assert wait_mod.parse_ps("[broken json") == []
    assert wait_mod.parse_ps("{broken json\n") == []


# =====================================================================
# health() tests
# =====================================================================

def test_health_200_ok(wait_mod):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"status": "ok"}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        status, body = wait_mod.health("http://127.0.0.1:8000")
    assert status == 200
    assert body == '{"status": "ok"}'


def test_health_connection_error_returns_0(wait_mod):
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        status, body = wait_mod.health("http://127.0.0.1:8000")
    assert status == 0
    assert body == ""


# =====================================================================
# main() execution: exit 0 (healthy), exit 1 (failed/restarting), exit 2 (timeout)
# =====================================================================

def test_main_exits_0_when_healthy(wait_mod, capsys):
    with patch.object(wait_mod, "health", return_value=(200, '{"edition": "core", "status": "ok"}')), \
         patch("sys.argv", ["wait_healthy.py", "--timeout", "10"]):
        with pytest.raises(SystemExit) as exc:
            wait_mod.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "[OK] healthy after" in out
        assert "RESULT: PASS" in out
        assert "EXIT CODE: 0" in out


def test_main_exits_1_when_service_is_restarting(wait_mod, capsys):
    restarting_ps = '[{"Service": "synap", "State": "restarting", "ExitCode": 1, "Health": ""}]'

    with patch.object(wait_mod, "health", return_value=(0, "")), \
         patch.object(wait_mod, "run", side_effect=lambda cmd, timeout=60: restarting_ps if "ps" in cmd else "FATAL ERROR in server startup"), \
         patch("sys.argv", ["wait_healthy.py", "--timeout", "10"]):
        with pytest.raises(SystemExit) as exc:
            wait_mod.main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "[ERROR] service 'synap' is restarting in a loop:" in out
        assert "FATAL ERROR in server startup" in out
        assert "RESULT: FAIL service synap failed" in out
        assert "EXIT CODE: 1" in out


def test_main_exits_1_when_first_start_fails_and_warns(wait_mod, capsys):
    failed_ps = '[{"Service": "first-start", "State": "exited", "ExitCode": 1, "Health": ""}]'

    with patch.object(wait_mod, "health", return_value=(0, "")), \
         patch.object(wait_mod, "run", side_effect=lambda cmd, timeout=60: failed_ps if "ps" in cmd else "download failed 403"), \
         patch("sys.argv", ["wait_healthy.py", "--timeout", "10"]):
        with pytest.raises(SystemExit) as exc:
            wait_mod.main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "[ERROR] service 'first-start' exited with a failure:" in out
        assert "[WARN] first-start stops when a model line in .env is still a placeholder" in out
        assert "RESULT: FAIL service first-start failed" in out
        assert "EXIT CODE: 1" in out


def test_main_exits_2_on_timeout(wait_mod, capsys):
    # Simulate time advancing past timeout immediately
    times = [100.0, 100.0, 200.0]  # start, check, expired

    def fake_time():
        return times.pop(0) if times else 300.0

    running_ps = '[{"Service": "synap", "State": "running", "ExitCode": 0, "Health": "starting"}]'

    with patch("time.time", side_effect=fake_time), \
         patch("time.sleep", return_value=None), \
         patch.object(wait_mod, "run", return_value=running_ps), \
         patch.object(wait_mod, "health", return_value=(0, "")), \
         patch("sys.argv", ["wait_healthy.py", "--timeout", "10"]):
        with pytest.raises(SystemExit) as exc:
            wait_mod.main()
        assert exc.value.code == 2
        out = capsys.readouterr().out
        assert "[ERROR] not healthy after 10s" in out
        assert "RESULT: FAIL timed out waiting for the server" in out
        assert "EXIT CODE: 2" in out
