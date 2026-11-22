"""test_preflight.py tests skills/synap-core/scripts/preflight.py.

Covers:
- check_os(): system/architecture logging, Windows WSL2 warning, Linux SYNAP_USER check
- check_docker(): missing docker, missing docker compose, unreachable daemon, low vs adequate memory
- check_disk(): low disk (<8 GiB), warning (8-12 GiB), sufficient disk (>=12 GiB)
- check_port(): free port vs port already in use
- check_existing(): clean install vs existing containers/volumes
- check_env(): missing .env vs existing .env with filled keys
- main(): overall pass/fail results and exit codes
"""
from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from conftest import SCRIPTS


def load_preflight():
    """Load preflight as a module without running main()."""
    spec = importlib.util.spec_from_file_location("preflight_mod", SCRIPTS / "preflight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def preflight(monkeypatch):
    monkeypatch.setenv("SYNAP_PORT", "8000")
    mod = load_preflight()
    mod.PORT = 8000
    mod.results.clear()
    return mod


# =====================================================================
# check_os()
# =====================================================================

def test_check_os_windows_warns(preflight):
    with patch("platform.system", return_value="Windows"), \
         patch("platform.machine", return_value="AMD64"), \
         patch("platform.release", return_value="10.0"):
        preflight.check_os()

    assert any("[WARN] Windows: run everything inside WSL2" in r for r in preflight.results)


def test_check_os_linux_without_synap_user_warns(preflight, monkeypatch):
    monkeypatch.delenv("SYNAP_USER", raising=False)
    with patch("platform.system", return_value="Linux"), \
         patch("platform.machine", return_value="x86_64"), \
         patch("platform.release", return_value="6.5.0"):
        preflight.check_os()

    assert any("[WARN] Linux: SYNAP_USER is not set in this shell" in r for r in preflight.results)


def test_check_os_linux_with_synap_user_passes(preflight, monkeypatch):
    monkeypatch.setenv("SYNAP_USER", "1000:1000")
    with patch("platform.system", return_value="Linux"), \
         patch("platform.machine", return_value="x86_64"), \
         patch("platform.release", return_value="6.5.0"):
        preflight.check_os()

    assert any("[OK] SYNAP_USER is set for this shell" in r for r in preflight.results)


# =====================================================================
# check_docker()
# =====================================================================

def test_check_docker_missing_fails(preflight):
    with patch.object(preflight, "run", return_value=(127, "")):
        ok = preflight.check_docker()
    assert ok is False
    assert any("[ERROR] Docker is not installed or not on PATH" in r for r in preflight.results)


def test_check_docker_compose_missing_fails(preflight):
    def fake_run(cmd, timeout=20):
        if cmd == ["docker", "--version"]:
            return (0, "Docker version 27.0.0")
        return (1, "docker: 'compose' is not a docker command")

    with patch.object(preflight, "run", side_effect=fake_run):
        ok = preflight.check_docker()
    assert ok is False
    assert any("[ERROR] Docker Compose v2 is missing" in r for r in preflight.results)


def test_check_docker_daemon_unreachable_fails(preflight):
    def fake_run(cmd, timeout=20):
        if cmd == ["docker", "--version"]:
            return (0, "Docker version 27.0.0")
        if cmd == ["docker", "compose", "version"]:
            return (0, "Docker Compose version v2.28.0")
        return (1, "Cannot connect to the Docker daemon")

    with patch.object(preflight, "run", side_effect=fake_run):
        ok = preflight.check_docker()
    assert ok is False
    assert any("[ERROR] The Docker daemon is not reachable" in r for r in preflight.results)


def test_check_docker_low_memory_warns(preflight):
    # 4 GiB = 4 * 2**30 bytes
    mem_bytes = str(4 * 2**30)

    def fake_run(cmd, timeout=20):
        if cmd == ["docker", "--version"]:
            return (0, "Docker version 27.0.0")
        if cmd == ["docker", "compose", "version"]:
            return (0, "Docker Compose version v2.28.0")
        return (0, f"27.0.0|{mem_bytes}|Ubuntu 22.04")

    with patch.object(preflight, "run", side_effect=fake_run):
        ok = preflight.check_docker()
    assert ok is True
    assert any("[WARN] Docker has 4.0 GiB of memory" in r for r in preflight.results)


def test_check_docker_adequate_memory_passes(preflight):
    # 8 GiB
    mem_bytes = str(8 * 2**30)

    def fake_run(cmd, timeout=20):
        if cmd == ["docker", "--version"]:
            return (0, "Docker version 27.0.0")
        if cmd == ["docker", "compose", "version"]:
            return (0, "Docker Compose version v2.28.0")
        return (0, f"27.0.0|{mem_bytes}|Ubuntu 22.04")

    with patch.object(preflight, "run", side_effect=fake_run):
        ok = preflight.check_docker()
    assert ok is True
    assert any("[OK] Docker daemon 27.0.0 reachable, 8.0 GiB of memory available" in r for r in preflight.results)


# =====================================================================
# check_disk()
# =====================================================================

def test_check_disk_insufficient_space_fails(preflight):
    usage_mock = MagicMock(free=5 * 2**30)  # 5 GiB free
    with patch("shutil.disk_usage", return_value=usage_mock):
        preflight.check_disk("/some/path", image_path=False)
    assert any("[ERROR] Only 5.0 GiB free" in r for r in preflight.results)


def test_check_disk_marginal_space_warns(preflight):
    usage_mock = MagicMock(free=10 * 2**30)  # 10 GiB free
    with patch("shutil.disk_usage", return_value=usage_mock):
        preflight.check_disk("/some/path", image_path=False)
    assert any("[WARN] 10.0 GiB free" in r for r in preflight.results)


def test_check_disk_sufficient_space_passes(preflight):
    usage_mock = MagicMock(free=50 * 2**30)  # 50 GiB free
    with patch("shutil.disk_usage", return_value=usage_mock):
        preflight.check_disk("/some/path", image_path=False)
    assert any("[OK] 50 GiB free" in r for r in preflight.results)


# =====================================================================
# check_port()
# =====================================================================

def test_check_port_free(preflight):
    socket_mock = MagicMock()
    socket_mock.__enter__.return_value.connect_ex.return_value = 111  # not busy
    with patch("socket.socket", return_value=socket_mock):
        preflight.check_port()
    assert any("[OK] port 8000 is free" in r for r in preflight.results)


def test_check_port_busy_warns(preflight):
    socket_mock = MagicMock()
    socket_mock.__enter__.return_value.connect_ex.return_value = 0  # busy
    with patch("socket.socket", return_value=socket_mock), \
         patch.object(preflight, "run", return_value=(0, "existing-synap-server")):
        preflight.check_port()
    assert any("[WARN] port 8000 is already in use by existing-synap-server" in r for r in preflight.results)


# =====================================================================
# check_existing()
# =====================================================================

def test_check_existing_clean_install(preflight):
    with patch.object(preflight, "run", return_value=(0, "")):
        preflight.check_existing()
    assert any("[OK] no earlier Synap Core containers or volumes" in r for r in preflight.results)


def test_check_existing_earlier_containers_and_volumes_warns(preflight):
    def fake_run(cmd, timeout=20):
        if "ps" in cmd:
            return (0, "synap-core-postgres|Up 2 hours\nsynap-core-synap|Up 2 hours")
        return (0, "synap-core_postgres_data\nsynap-core_neo4j_data")

    with patch.object(preflight, "run", side_effect=fake_run):
        preflight.check_existing()
    assert any("[WARN] an earlier Synap Core stack exists: 2 containers named synap-core-*" in r for r in preflight.results)
    assert any("[WARN] Docker volumes from an earlier Synap Core exist" in r for r in preflight.results)


# =====================================================================
# check_env()
# =====================================================================

def test_check_env_missing_is_ok(preflight, tmp_path):
    preflight.check_env(str(tmp_path))
    assert any("[OK] no .env in" in r for r in preflight.results)


def test_check_env_existing_reports_filled_keys(preflight, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SYNAP_LLM_PROVIDER=openai\nSYNAP_LLM_MODEL=gpt-4o-mini\n# comment\nEMPTY_VAR=\n")
    preflight.check_env(str(tmp_path))
    assert any("[OK] .env exists with these lines filled in (names only): SYNAP_LLM_PROVIDER, SYNAP_LLM_MODEL" in r for r in preflight.results)


# =====================================================================
# main() execution: PASS vs FAIL
# =====================================================================

def test_main_exits_1_when_docker_fails(preflight, capsys):
    with patch.object(preflight, "check_os"), \
         patch.object(preflight, "check_docker", return_value=False), \
         patch.object(preflight, "check_tools"), \
         patch.object(preflight, "check_network"), \
         patch.object(preflight, "check_env"), \
         patch("sys.argv", ["preflight.py"]):
        preflight.say("ERROR", "Docker is not installed or not on PATH.")
        with pytest.raises(SystemExit) as exc:
            preflight.main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "RESULT: FAIL Docker is not installed or not on PATH" in out
        assert "EXIT CODE: 1" in out


def test_main_exits_0_when_no_errors(preflight, capsys):
    with patch.object(preflight, "check_os"), \
         patch.object(preflight, "check_docker", return_value=True), \
         patch.object(preflight, "check_disk"), \
         patch.object(preflight, "check_port"), \
         patch.object(preflight, "check_existing"), \
         patch.object(preflight, "check_tools"), \
         patch.object(preflight, "check_network"), \
         patch.object(preflight, "check_env"), \
         patch("sys.argv", ["preflight.py"]):
        preflight.main()
        out = capsys.readouterr().out
        assert "RESULT: PASS" in out
        assert "EXIT CODE: 0" in out
