"""Tests for live preview port state and agent-facing hints."""

from mini_devin.api.live_preview_state import (
    allowed_ports,
    live_preview_probe_hints,
    live_preview_set_port_warning,
)


def test_live_preview_probe_hints_warns_when_port_env_listens(monkeypatch):
    monkeypatch.setenv("PORT", "8080")
    h = live_preview_probe_hints(listening_ports=[5173, 8080])
    assert "what_live_preview_is_for" in h
    assert "platform_bind_warning" in h
    assert "8080" in h["platform_bind_warning"]


def test_live_preview_probe_hints_no_platform_warning_when_port_not_listening(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    h = live_preview_probe_hints(listening_ports=[5173])
    assert "platform_bind_warning" not in h


def test_live_preview_set_port_warning_matches_port_env(monkeypatch):
    monkeypatch.setenv("PORT", "3000")
    assert live_preview_set_port_warning(3000) is not None
    assert live_preview_set_port_warning(3001) is None


def test_allowed_ports_include_common_fallback_dev_ports(monkeypatch):
    monkeypatch.delenv("LIVE_PREVIEW_ALLOWED_PORTS", raising=False)
    ports = allowed_ports()
    assert 3001 in ports
    assert 3002 in ports
    assert 5001 in ports
    assert 5002 in ports
    assert 5173 in ports
    assert 8000 in ports
