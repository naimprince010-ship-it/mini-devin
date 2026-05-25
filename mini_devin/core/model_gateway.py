"""Feature-flagged model gateway and intelligence routing scaffold.

This module is additive and local-first. It introduces a centralized routing layer for
provider/model selection, fallback decisions, budget hooks, token/cost accounting,
prompt-cache scaffolding, and retry classification while preserving existing runtime behavior
when disabled.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mini_devin.contracts.protocols import DurableCheckpoint
from mini_devin.core.providers import get_litellm_model_name
from mini_devin.orchestration.checkpoint_store import JsonlCheckpointStore
from mini_devin.orchestration.observability import TimelineRecord, emit_worker_metric, record_timeline_event
from mini_devin.orchestration.runtime_contracts import FileTypedEventEmitter


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def model_gateway_enabled() -> bool:
    return _flag_enabled("PLODDER_MODEL_GATEWAY")


def prompt_cache_enabled() -> bool:
    return _flag_enabled("PLODDER_MODEL_PROMPT_CACHE", True)


def budget_enforcement_enabled() -> bool:
    return _flag_enabled("PLODDER_MODEL_BUDGET_ENFORCEMENT", False)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _workspace_root(workspace: str | Path | None = None) -> Path:
    root = Path(workspace or os.environ.get("PLODDER_WORKSPACE", os.getcwd())).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _plodder_dir(workspace: str | Path | None = None) -> Path:
    root = _workspace_root(workspace) / ".plodder"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_path(workspace: str | Path | None = None) -> Path:
    return _plodder_dir(workspace) / "model_gateway_cache.jsonl"


def _budget_path(workspace: str | Path | None = None) -> Path:
    return _plodder_dir(workspace) / "model_gateway_budgets.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, default=str, ensure_ascii=False, indent=2), encoding="utf-8")


class FailureType(str):
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    BUDGET = "budget"
    TIMEOUT = "timeout"
    PROVIDER = "provider"
    UNKNOWN = "unknown"


class ReasoningDepth(str):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelWeight(str):
    LIGHT = "light"
    HEAVY = "heavy"


@dataclass(frozen=True, slots=True)
class ModelBudget:
    max_cost_usd: float = 0.0
    max_tokens: int = 0

    def allows(self, *, spent_cost_usd: float, spent_tokens: int) -> bool:
        if self.max_cost_usd > 0 and spent_cost_usd >= self.max_cost_usd:
            return False
        if self.max_tokens > 0 and spent_tokens >= self.max_tokens:
            return False
        return True


@dataclass(frozen=True, slots=True)
class TokenAccountingRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    ts: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True, slots=True)
class RoutingContext:
    task_id: str = ""
    session_id: str = ""
    task_type: str = "general"
    reasoning_depth: ReasoningDepth = ReasoningDepth.MEDIUM
    prefer_heavy_model: bool = False
    force_model: str | None = None
    budget: ModelBudget | None = None


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    selected_model: str
    candidate_models: tuple[str, ...]
    reasoning_depth: ReasoningDepth
    model_weight: ModelWeight
    cache_key: str


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: str
    model: str
    response_content: str | None
    response_usage: dict[str, int]
    finish_reason: str
    created_at: str


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    next_model: str | None
    reason: str


class ModelGateway:
    """Centralized model routing and accounting scaffold."""

    def __init__(
        self,
        workspace: str | Path | None = None,
        *,
        checkpoint_store: JsonlCheckpointStore | None = None,
        typed_emitter: FileTypedEventEmitter | None = None,
    ) -> None:
        self.workspace = _workspace_root(workspace)
        self.checkpoint_store = checkpoint_store or JsonlCheckpointStore(self.workspace)
        self.typed_emitter = typed_emitter or FileTypedEventEmitter(self.workspace)
        self._cache_ttl_seconds = max(0, int((os.environ.get("PLODDER_MODEL_CACHE_TTL_SECONDS") or "300").strip() or "300"))

    @staticmethod
    def _default_light_models() -> list[str]:
        raw = (os.environ.get("PLODDER_LIGHT_MODELS") or "").strip()
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        return ["llama-3.1-8b-instant", "gpt-4o-mini", "gemini/gemini-2.0-flash"]

    @staticmethod
    def _default_heavy_models() -> list[str]:
        raw = (os.environ.get("PLODDER_HEAVY_MODELS") or "").strip()
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        return ["llama-3.3-70b-versatile", "gpt-4o", "claude-3-5-sonnet-20241022"]

    @staticmethod
    def classify_failure(error_message: str) -> str:
        msg = (error_message or "").lower()
        if "rate limit" in msg or "ratelimit" in msg or "429" in msg or "quota" in msg:
            return FailureType.RATE_LIMIT
        if "timeout" in msg or "timed out" in msg or "408" in msg:
            return FailureType.TIMEOUT
        if "authentication" in msg or "401" in msg or "unauthorized" in msg or "invalid api key" in msg:
            return FailureType.AUTH
        if "not found" in msg or "404" in msg or "model not found" in msg:
            return FailureType.NOT_FOUND
        if "budget" in msg:
            return FailureType.BUDGET
        if "connection" in msg or "503" in msg or "502" in msg or "500" in msg:
            return FailureType.TRANSIENT
        if "provider" in msg:
            return FailureType.PROVIDER
        return FailureType.UNKNOWN

    @staticmethod
    def should_retry(failure_type: str) -> bool:
        return failure_type in (FailureType.RATE_LIMIT, FailureType.TIMEOUT, FailureType.TRANSIENT, FailureType.PROVIDER)

    @staticmethod
    def _provider_of(model: str) -> str:
        lowered = (model or "").strip().lower()
        if "/" in lowered:
            return lowered.split("/", 1)[0]
        if lowered.startswith("gpt"):
            return "openai"
        if lowered.startswith("claude"):
            return "anthropic"
        if lowered.startswith("gemini"):
            return "google"
        if lowered.startswith("llama") or lowered.startswith("mixtral"):
            return "groq"
        return "unknown"

    def _record_gateway_event(self, event_type: str, *, task_id: str, session_id: str, payload: Mapping[str, Any]) -> None:
        record_timeline_event(
            self.workspace,
            TimelineRecord(
                event_type=event_type,
                source="model_gateway",
                session_id=session_id or None,
                task_id=task_id or None,
                unit_id=task_id or "gateway",
                status=event_type,
                payload=dict(payload),
            ),
        )
        self.typed_emitter.emit({"event_type": event_type, "task_id": task_id, "session_id": session_id, **dict(payload)})

    def _cache_key(self, model: str, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None) -> str:
        payload = {
            "model": model,
            "messages": list(messages),
            "tools": list(tools or []),
        }
        raw = json.dumps(payload, default=str, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache_entries(self) -> list[CacheEntry]:
        path = _cache_path(self.workspace)
        if not path.is_file():
            return []
        entries: list[CacheEntry] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            entries.append(
                CacheEntry(
                    key=str(row.get("key") or ""),
                    model=str(row.get("model") or ""),
                    response_content=row.get("response_content"),
                    response_usage=dict(row.get("response_usage") or {}),
                    finish_reason=str(row.get("finish_reason") or "stop"),
                    created_at=str(row.get("created_at") or ""),
                )
            )
        return entries

    def cache_get(self, key: str) -> CacheEntry | None:
        if not prompt_cache_enabled():
            return None
        for entry in reversed(self._load_cache_entries()):
            if entry.key != key:
                continue
            if self._cache_ttl_seconds > 0:
                try:
                    created = datetime.fromisoformat(entry.created_at.replace("Z", "+00:00"))
                    age = (_utcnow() - created).total_seconds()
                    if age > self._cache_ttl_seconds:
                        return None
                except ValueError:
                    return None
            return entry
        return None

    def cache_put(self, entry: CacheEntry) -> None:
        if not prompt_cache_enabled():
            return
        path = _cache_path(self.workspace)
        row = {
            "key": entry.key,
            "model": entry.model,
            "response_content": entry.response_content,
            "response_usage": dict(entry.response_usage),
            "finish_reason": entry.finish_reason,
            "created_at": entry.created_at,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    def _load_budget_state(self) -> dict[str, Any]:
        data = _read_json(_budget_path(self.workspace), default={})
        if not isinstance(data, dict):
            return {}
        return data

    def _save_budget_state(self, data: Mapping[str, Any]) -> None:
        _write_json(_budget_path(self.workspace), dict(data))

    def budget_snapshot(self, task_id: str) -> tuple[int, float]:
        state = self._load_budget_state()
        row = state.get(task_id) if isinstance(state, dict) else None
        if not isinstance(row, dict):
            return (0, 0.0)
        return (int(row.get("tokens") or 0), float(row.get("cost_usd") or 0.0))

    def enforce_budget(self, *, context: RoutingContext) -> bool:
        if not budget_enforcement_enabled() or context.budget is None or not context.task_id:
            return True
        spent_tokens, spent_cost = self.budget_snapshot(context.task_id)
        allowed = context.budget.allows(spent_cost_usd=spent_cost, spent_tokens=spent_tokens)
        if not allowed:
            self._record_gateway_event(
                "model_gateway.budget_blocked",
                task_id=context.task_id,
                session_id=context.session_id,
                payload={"spent_tokens": spent_tokens, "spent_cost_usd": spent_cost},
            )
            emit_worker_metric(
                self.workspace,
                "worker.model_gateway.budget_blocked",
                1.0,
                labels={"task_id": context.task_id},
            )
        return allowed

    def _pricing_per_1k(self) -> dict[str, float]:
        raw = (os.environ.get("PLODDER_MODEL_PRICING_PER_1K_JSON") or "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in data.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def account_usage(self, *, context: RoutingContext, model: str, usage: Mapping[str, int]) -> TokenAccountingRecord:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        pricing = self._pricing_per_1k()
        price = pricing.get(model, pricing.get(self._provider_of(model), 0.0))
        estimated_cost = (float(total_tokens) / 1000.0) * float(price)
        record = TokenAccountingRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )

        if context.task_id:
            state = self._load_budget_state()
            current = state.get(context.task_id) if isinstance(state.get(context.task_id), dict) else {}
            state[context.task_id] = {
                "tokens": int(current.get("tokens") or 0) + total_tokens,
                "cost_usd": float(current.get("cost_usd") or 0.0) + estimated_cost,
                "updated_at": _utcnow().isoformat(),
            }
            self._save_budget_state(state)

            self.checkpoint_store.save(
                DurableCheckpoint(
                    checkpoint_id=f"model-gateway:{context.session_id or 'session'}:{context.task_id}",
                    scope_id=context.task_id,
                    state={
                        "total_tokens": state[context.task_id]["tokens"],
                        "estimated_cost_usd": state[context.task_id]["cost_usd"],
                    },
                    metadata={"model": model},
                )
            )

            self._record_gateway_event(
                "model_gateway.accounted",
                task_id=context.task_id,
                session_id=context.session_id,
                payload={
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost,
                },
            )

        emit_worker_metric(
            self.workspace,
            "worker.model_gateway.tokens",
            float(total_tokens),
            labels={"task_id": context.task_id or "", "model": model},
        )
        return record

    def select_route(
        self,
        *,
        base_model: str,
        fallback_candidates: Sequence[str],
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        context: RoutingContext,
    ) -> RoutingDecision:
        if context.force_model:
            selected = context.force_model
        else:
            selected = base_model

        message_chars = sum(len(str(m.get("content") or "")) for m in messages if isinstance(m, Mapping))
        prefers_heavy = context.prefer_heavy_model or context.reasoning_depth == ReasoningDepth.HIGH or message_chars > 6000
        model_weight = ModelWeight.HEAVY if prefers_heavy else ModelWeight.LIGHT

        curated = self._default_heavy_models() if model_weight == ModelWeight.HEAVY else self._default_light_models()
        candidates: list[str] = []
        seen: set[str] = set()

        for model in [selected, *curated, *fallback_candidates]:
            litellm_name = get_litellm_model_name(model)
            if litellm_name in seen:
                continue
            seen.add(litellm_name)
            candidates.append(litellm_name)

        cache_key = self._cache_key(candidates[0], messages, tools)
        decision = RoutingDecision(
            selected_model=candidates[0],
            candidate_models=tuple(candidates),
            reasoning_depth=context.reasoning_depth,
            model_weight=model_weight,
            cache_key=cache_key,
        )

        self._record_gateway_event(
            "model_gateway.routed",
            task_id=context.task_id,
            session_id=context.session_id,
            payload={
                "selected_model": decision.selected_model,
                "candidate_models": list(decision.candidate_models),
                "reasoning_depth": decision.reasoning_depth,
                "model_weight": decision.model_weight,
                "tool_count": len(list(tools or [])),
            },
        )
        emit_worker_metric(
            self.workspace,
            "worker.model_gateway.route",
            1.0,
            labels={"task_id": context.task_id or "", "weight": decision.model_weight},
        )
        return decision

    def fallback_after_failure(
        self,
        *,
        decision: RoutingDecision,
        failed_model: str,
        failure_type: str,
    ) -> FallbackDecision:
        candidates = list(decision.candidate_models)
        if failed_model not in candidates:
            reason = f"failed_model_not_in_candidates:{failed_model}"
            return FallbackDecision(next_model=None, reason=reason)

        idx = candidates.index(failed_model)

        for next_idx in range(idx + 1, len(candidates)):
            candidate = candidates[next_idx]
            if failure_type in (FailureType.AUTH, FailureType.PROVIDER):
                if self._provider_of(candidate) == self._provider_of(failed_model):
                    continue
            return FallbackDecision(next_model=candidate, reason=f"fallback:{failure_type}")

        return FallbackDecision(next_model=None, reason=f"no_fallback:{failure_type}")


def routing_context_from_dict(payload: Mapping[str, Any] | None) -> RoutingContext:
    raw = dict(payload or {})
    depth_raw = str(raw.get("reasoning_depth") or "medium").lower().strip()
    depth = ReasoningDepth.MEDIUM
    if depth_raw in (ReasoningDepth.LOW, ReasoningDepth.MEDIUM, ReasoningDepth.HIGH):
        depth = depth_raw

    budget: ModelBudget | None = None
    budget_raw = raw.get("budget")
    if isinstance(budget_raw, Mapping):
        budget = ModelBudget(
            max_cost_usd=float(budget_raw.get("max_cost_usd") or 0.0),
            max_tokens=int(budget_raw.get("max_tokens") or 0),
        )

    return RoutingContext(
        task_id=str(raw.get("task_id") or ""),
        session_id=str(raw.get("session_id") or ""),
        task_type=str(raw.get("task_type") or "general"),
        reasoning_depth=depth,
        prefer_heavy_model=bool(raw.get("prefer_heavy_model") or False),
        force_model=str(raw.get("force_model")) if raw.get("force_model") else None,
        budget=budget,
    )
