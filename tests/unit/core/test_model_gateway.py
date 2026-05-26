from __future__ import annotations

from pathlib import Path

import pytest

from mini_devin.core.model_gateway import (
    CacheEntry,
    FailureType,
    ModelBudget,
    ModelGateway,
    ModelWeight,
    ReasoningDepth,
    RoutingContext,
)


@pytest.fixture(autouse=True)
def _gateway_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLODDER_MODEL_GATEWAY", "1")
    monkeypatch.setenv("PLODDER_MODEL_PROMPT_CACHE", "1")
    monkeypatch.setenv("PLODDER_MODEL_BUDGET_ENFORCEMENT", "1")
    monkeypatch.setenv("PLODDER_WORKER_METRICS", "1")
    monkeypatch.setenv("PLODDER_OBSERVABILITY", "1")
    monkeypatch.setenv("PLODDER_TIMELINE_RECORDING", "1")


def test_routing_decision_prefers_heavy_model_for_high_reasoning(tmp_path: Path) -> None:
    gateway = ModelGateway(tmp_path)
    context = RoutingContext(task_id="task-1", session_id="sess-1", reasoning_depth=ReasoningDepth.HIGH)

    decision = gateway.select_route(
        base_model="llama-3.3-70b-versatile",
        fallback_candidates=["gpt-4o-mini", "gpt-4o"],
        messages=[{"role": "user", "content": "Need deep analysis"}],
        tools=[{"name": "terminal"}],
        context=context,
    )

    assert decision.model_weight == ModelWeight.HEAVY
    assert decision.selected_model
    assert len(decision.candidate_models) >= 2


def test_fallback_behavior_and_provider_failure_recovery(tmp_path: Path) -> None:
    gateway = ModelGateway(tmp_path)
    context = RoutingContext(task_id="task-2", session_id="sess-2", reasoning_depth=ReasoningDepth.MEDIUM)
    decision = gateway.select_route(
        base_model="openai/gpt-4o",
        fallback_candidates=["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"],
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        context=context,
    )

    fallback = gateway.fallback_after_failure(
        decision=decision,
        failed_model=decision.selected_model,
        failure_type=FailureType.AUTH,
    )

    assert fallback.next_model is not None
    assert fallback.reason.startswith("fallback:")
    assert not fallback.next_model.startswith("openai/")


def test_budget_enforcement_blocks_after_spend(tmp_path: Path) -> None:
    gateway = ModelGateway(tmp_path)
    context = RoutingContext(
        task_id="task-3",
        session_id="sess-3",
        budget=ModelBudget(max_cost_usd=0.001, max_tokens=10),
    )

    first_allowed = gateway.enforce_budget(context=context)
    gateway.account_usage(
        context=context,
        model="openai/gpt-4o-mini",
        usage={"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16},
    )
    second_allowed = gateway.enforce_budget(context=context)

    assert first_allowed is True
    assert second_allowed is False


def test_token_accounting_updates_budget_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLODDER_MODEL_PRICING_PER_1K_JSON", '{"openai/gpt-4o-mini": 0.2}')
    gateway = ModelGateway(tmp_path)
    context = RoutingContext(task_id="task-4", session_id="sess-4")

    record = gateway.account_usage(
        context=context,
        model="openai/gpt-4o-mini",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    tokens, cost = gateway.budget_snapshot("task-4")

    assert record.total_tokens == 150
    assert record.estimated_cost_usd == pytest.approx(0.03)
    assert tokens == 150
    assert cost == pytest.approx(0.03)


def test_cache_hits_and_misses(tmp_path: Path) -> None:
    gateway = ModelGateway(tmp_path)
    key = "cache-key-1"

    miss = gateway.cache_get(key)
    assert miss is None

    gateway.cache_put(
        CacheEntry(
            key=key,
            model="openai/gpt-4o-mini",
            response_content="cached-response",
            response_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason="stop",
            created_at="2099-01-01T00:00:00+00:00",
        )
    )

    hit = gateway.cache_get(key)
    assert hit is not None
    assert hit.response_content == "cached-response"


def test_retry_classification_by_failure_type() -> None:
    assert ModelGateway.classify_failure("RateLimitError: 429") == FailureType.RATE_LIMIT
    assert ModelGateway.classify_failure("AuthenticationError: 401") == FailureType.AUTH
    assert ModelGateway.classify_failure("Connection reset by peer") == FailureType.TRANSIENT
    assert ModelGateway.classify_failure("Operation timed out") == FailureType.TIMEOUT
    assert ModelGateway.classify_failure("Model Not Found: 404") == FailureType.NOT_FOUND
    assert ModelGateway.classify_failure("weird unknown") == FailureType.UNKNOWN

    assert ModelGateway.should_retry(FailureType.RATE_LIMIT) is True
    assert ModelGateway.should_retry(FailureType.AUTH) is False


def test_provider_failure_recovery_none_when_failed_model_not_in_route(tmp_path: Path) -> None:
    gateway = ModelGateway(tmp_path)
    context = RoutingContext(task_id="task-5", session_id="sess-5")
    decision = gateway.select_route(
        base_model="openai/gpt-4o",
        fallback_candidates=[],
        messages=[{"role": "user", "content": "x"}],
        tools=None,
        context=context,
    )

    fallback = gateway.fallback_after_failure(
        decision=decision,
        failed_model="openai/does-not-exist",
        failure_type=FailureType.PROVIDER,
    )

    assert fallback.next_model is None
    assert fallback.reason.startswith("failed_model_not_in_candidates:")
