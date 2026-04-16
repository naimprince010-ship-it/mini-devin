"""Dev-server helper behavior for the host process sandbox."""

from mini_devin.sandbox.process_execution_sandbox import (
    ProcessExecutionSandbox,
    _command_with_port,
    _dev_server_port_candidates,
    _strip_trailing_background,
)


def test_dev_server_port_candidates_prioritize_explicit_port():
    ports = _dev_server_port_candidates("npm run dev -- --port 3000")
    assert ports[0] == 3000
    assert 5001 in ports
    assert 5002 in ports
    assert 5173 in ports
    assert 8000 in ports
    assert len(ports) == len(set(ports))


def test_command_with_port_rewrites_npm_dev():
    cmd = _command_with_port("npm run dev", 5173)
    assert cmd.startswith("PORT=5173 ")
    assert "-- --port 5173" in cmd


def test_command_with_port_rewrites_python_app_and_strips_background():
    cmd = _command_with_port("python app.py &", 5001)
    assert "&" not in cmd
    assert "PORT=5001" in cmd
    assert "FLASK_RUN_PORT=5001" in cmd
    assert cmd.endswith("python app.py")


def test_strip_trailing_background_only_removes_terminal_ampersand():
    assert _strip_trailing_background("npm run dev &") == "npm run dev"
    assert _strip_trailing_background("echo 'a & b'") == "echo 'a & b'"


def test_run_dev_server_skips_busy_port_and_tries_fallback(monkeypatch, tmp_path):
    sb = ProcessExecutionSandbox(tmp_path)

    busy = {3000}
    opened: list[int] = []

    def fake_port_open(port: int, *, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
        del host, timeout
        return int(port) in busy

    class FakeProc:
        def __init__(self, pid: int):
            self.pid = pid

        def poll(self):
            return None

        def terminate(self):
            return None

    def fake_popen(args, cwd=None, env=None, stdin=None, stdout=None, stderr=None, start_new_session=None):
        del cwd, env, stdin, stderr, start_new_session
        cmd = str(args[-1])
        if "--port 3001" in cmd or "PORT=3001" in cmd:
            busy.add(3001)
            opened.append(3001)
        if stdout is not None:
            stdout.write(b"Server is running on port 3001\n")
            stdout.flush()
        return FakeProc(4321)

    monkeypatch.setattr("mini_devin.sandbox.process_execution_sandbox._port_open", fake_port_open)
    monkeypatch.setattr("mini_devin.sandbox.process_execution_sandbox.subprocess.Popen", fake_popen)

    result = sb.run_dev_server_in_workspace({}, ["sh", "-c", "node server.js"], timeout_sec=3)
    assert result["ok"] is True
    assert result["port"] == 3001
    assert "3000" not in str(result["stdout"])
