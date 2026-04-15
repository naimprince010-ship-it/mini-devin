"""Dev-server helper behavior for the host process sandbox."""

from mini_devin.sandbox.process_execution_sandbox import (
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
