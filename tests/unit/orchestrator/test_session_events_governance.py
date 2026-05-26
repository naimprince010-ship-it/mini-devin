from __future__ import annotations

import pytest

from mini_devin.orchestrator.session_events import (
    GOVERNANCE_TELEMETRY_SCHEMA_VERSION,
    build_governance_signal,
    normalize_governance_signals,
)


def test_build_governance_signal_budget_shape() -> None:
    row = build_governance_signal(
        "budget",
        status="within_limit",
        counters={"llm_total_tokens": 10, "token_budget": 100},
        detail="tracking",
    )

    assert row["schema"] == GOVERNANCE_TELEMETRY_SCHEMA_VERSION
    assert row["observe_only"] is True
    assert row["signal_type"] == "budget"
    assert row["status"] == "within_limit"
    assert row["counters"]["token_budget"] == 100


def test_build_governance_signal_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        build_governance_signal("unknown", status="stable")


def test_normalize_governance_signals_drops_invalid_rows() -> None:
    rows = normalize_governance_signals(
        [
            {"signal_type": "retry", "status": "active", "counters": {"consecutive_failures": 1}},
            {"signal_type": "bogus", "status": "active"},
            {"status": "missing_type"},
        ]
    )

    assert len(rows) == 1
    assert rows[0]["signal_type"] == "retry"
    assert rows[0]["status"] == "active"
