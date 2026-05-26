"""Redis Streams adapter scaffold for future queue-backed orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RedisStreamsConfig:
    """Connection settings for a Redis Streams transport."""

    url: str
    stream_name: str = "plodder.events"
    consumer_group: str = "plodder-workers"
    consumer_name: str = ""
    maxlen: int = 10_000
    block_ms: int = 1_000


@runtime_checkable
class RedisStreamsAdapter(Protocol):
    """Typed adapter boundary for event publication and consumption."""

    config: RedisStreamsConfig

    def connect(self) -> None:
        ...

    def publish(self, event: Mapping[str, Any]) -> str:
        ...

    def create_consumer_group(self) -> None:
        ...

    def read(self) -> list[tuple[str, dict[str, Any]]]:
        ...

    def ack(self, message_id: str) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class RedisStreamsAdapterSkeleton:
    """No-op transport skeleton. Real Redis wiring comes in a later phase."""

    config: RedisStreamsConfig

    def connect(self) -> None:
        raise NotImplementedError("Redis Streams transport is not implemented yet")

    def publish(self, event: Mapping[str, Any]) -> str:
        raise NotImplementedError("Redis Streams transport is not implemented yet")

    def create_consumer_group(self) -> None:
        raise NotImplementedError("Redis Streams transport is not implemented yet")

    def read(self) -> list[tuple[str, dict[str, Any]]]:
        raise NotImplementedError("Redis Streams transport is not implemented yet")

    def ack(self, message_id: str) -> None:
        raise NotImplementedError("Redis Streams transport is not implemented yet")

    def close(self) -> None:
        raise NotImplementedError("Redis Streams transport is not implemented yet")
"""Redis Streams adapter skeleton for orchestrator events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class RedisStreamMessage:
    """Message envelope for a stream-backed event transport."""

    stream: str
    message_id: str | None = None
    fields: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class RedisStreamsAdapter(Protocol):
    """Transport contract for Redis Streams-backed fan-out and leases."""

    def publish(self, stream: str, fields: Mapping[str, Any]) -> RedisStreamMessage:
        ...

    def create_consumer_group(self, stream: str, group: str) -> None:
        ...

    def read_group(
        self,
        stream: str,
        group: str,
        consumer: str,
        *,
        count: int = 1,
        block_ms: int = 1000,
    ) -> list[RedisStreamMessage]:
        ...

    def ack(self, stream: str, group: str, message_id: str) -> None:
        ...

    def close(self) -> None:
        ...


class RedisStreamsSkeleton:
    """Placeholder adapter for the future Redis Streams transport."""

    def __init__(self, *, namespace: str = "plodder", client: Any | None = None) -> None:
        self.namespace = namespace
        self.client = client

    def publish(self, stream: str, fields: Mapping[str, Any]) -> RedisStreamMessage:
        raise NotImplementedError("Redis Streams publish is not wired yet.")

    def create_consumer_group(self, stream: str, group: str) -> None:
        raise NotImplementedError("Redis Streams consumer groups are not wired yet.")

    def read_group(
        self,
        stream: str,
        group: str,
        consumer: str,
        *,
        count: int = 1,
        block_ms: int = 1000,
    ) -> list[RedisStreamMessage]:
        raise NotImplementedError("Redis Streams reads are not wired yet.")

    def ack(self, stream: str, group: str, message_id: str) -> None:
        raise NotImplementedError("Redis Streams acknowledgements are not wired yet.")

    def close(self) -> None:
        return None


def create_redis_streams_adapter(*, namespace: str = "plodder", client: Any | None = None) -> RedisStreamsSkeleton:
    return RedisStreamsSkeleton(namespace=namespace, client=client)
