"""Shared JSON schema loading and validation for runtime events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class EventSchemaValidationError(ValueError):
    """Raised when an event payload does not satisfy its schema."""


@dataclass(frozen=True, slots=True)
class EventSchemaDocument:
    """Loaded event schema artifact."""

    event_type: str
    path: Path
    schema: dict[str, Any]


def _default_schema_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data-contracts" / "events"


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _type_matches(expected: Any, value: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(item, value) for item in expected)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return _is_integer(value)
    if expected == "number":
        return (_is_integer(value) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _parse_datetime(value: str) -> None:
    normalized = value.replace("Z", "+00:00")
    datetime.fromisoformat(normalized)


def _validate_schema(schema: Mapping[str, Any], value: Any, *, path: str = "$") -> None:
    if "const" in schema and value != schema["const"]:
        raise EventSchemaValidationError(f"{path}: expected constant {schema['const']!r}")

    if "enum" in schema and value not in schema["enum"]:
        raise EventSchemaValidationError(f"{path}: value {value!r} is not allowed")

    expected_type = schema.get("type")
    if expected_type is not None and not _type_matches(expected_type, value):
        raise EventSchemaValidationError(
            f"{path}: expected type {expected_type!r}, got {type(value).__name__}"
        )

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise EventSchemaValidationError(f"{path}: string is shorter than minLength")
        if schema.get("format") == "date-time":
            try:
                _parse_datetime(value)
            except ValueError as exc:
                raise EventSchemaValidationError(f"{path}: invalid date-time value") from exc

    if isinstance(value, Mapping):
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for key in required:
            if key not in value:
                raise EventSchemaValidationError(f"{path}: missing required property {key!r}")

        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                child_schema = properties[key]
                if isinstance(child_schema, Mapping):
                    _validate_schema(child_schema, item, path=child_path)
            elif additional is False:
                raise EventSchemaValidationError(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, Mapping):
                _validate_schema(additional, item, path=child_path)

    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        item_schema = schema["items"]
        for index, item in enumerate(value):
            _validate_schema(item_schema, item, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class EventSchemaRegistry:
    """In-memory index of loaded event schema documents."""

    documents: dict[str, EventSchemaDocument]

    @classmethod
    def load_default(cls) -> "EventSchemaRegistry":
        return cls.load_from_directory(_default_schema_dir())

    @classmethod
    def load_from_directory(cls, directory: str | Path) -> "EventSchemaRegistry":
        base = Path(directory)
        if not base.is_dir():
            raise FileNotFoundError(f"Event schema directory not found: {base}")

        documents: dict[str, EventSchemaDocument] = {}
        for path in sorted(base.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
            if not isinstance(schema, dict):
                raise EventSchemaValidationError(f"{path}: schema must be a JSON object")

            event_type = _infer_event_type(schema, path)
            documents[event_type] = EventSchemaDocument(event_type=event_type, path=path, schema=schema)
        return cls(documents=documents)

    def event_types(self) -> tuple[str, ...]:
        return tuple(sorted(self.documents))

    def get(self, event_type: str) -> EventSchemaDocument:
        try:
            return self.documents[event_type]
        except KeyError as exc:
            raise EventSchemaValidationError(f"unknown event type: {event_type}") from exc

    def validate(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        document = self.get(event_type)
        if not isinstance(payload, Mapping):
            raise EventSchemaValidationError("payload: expected mapping")
        payload_dict = dict(payload)
        _validate_schema(document.schema, payload_dict)
        return payload_dict

    def validate_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        event_type = str(payload.get("event_type") or "")
        if not event_type:
            raise EventSchemaValidationError("missing event_type")
        return self.validate(event_type, payload)


def _infer_event_type(schema: Mapping[str, Any], path: Path) -> str:
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        event_prop = properties.get("event_type")
        if isinstance(event_prop, Mapping):
            const_value = event_prop.get("const")
            if isinstance(const_value, str) and const_value.strip():
                return const_value
    return path.stem


def load_event_schema_registry(directory: str | Path | None = None) -> EventSchemaRegistry:
    """Load all known event schemas from the contract folder."""

    return EventSchemaRegistry.load_from_directory(directory or _default_schema_dir())


def validate_event_payload(payload: Mapping[str, Any], *, directory: str | Path | None = None) -> dict[str, Any]:
    """Validate a payload against the schema named by its ``event_type``."""

    return load_event_schema_registry(directory).validate_payload(payload)


