"""
Canonical function-tool contracts (read_file, write_file, run_command, search_repo).

The live LLM still calls ``editor`` / ``terminal`` with structured parameters; this module
documents the strict mapping and provides JSON-schema-shaped definitions for prompts/UI.
"""

from __future__ import annotations

import json
from typing import Any

# Logical names -> implementation mapping for documentation / export
CANONICAL_TOOL_MAP: dict[str, dict[str, Any]] = {
    "run_command": {
        "implements_tool": "terminal",
        "description": "Run a shell command in the workspace; primary way to inspect the environment.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_directory": {"type": "string", "default": "."},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            "required": ["command"],
        },
        "structured_feedback": {
            "stdout": "string",
            "exit_code": "integer",
            "recovery_hint": "string | null",
            "filesystem_delta": "object",
        },
    },
    "read_file": {
        "implements_tool": "editor",
        "arguments_template": {"action": "read_file", "path": "<path>"},
        "description": "Read file contents (optionally with line range).",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "implements_tool": "editor",
        "arguments_template": {"action": "write_file", "path": "<path>", "content": "<content>"},
        "description": "Create or overwrite a file.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    "search_repo": {
        "implements_tool": "editor",
        "arguments_template": {"action": "search", "pattern": "<pattern>", "path": "."},
        "description": "Search for a pattern across files in the repo.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "file_pattern": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
}


def canonical_tools_prompt_block() -> str:
    """Short system text: logical tools vs actual function names."""
    lines = [
        "## Canonical tools (use the **registered function names** `terminal` and `editor`)",
        "",
        "Logical capability | Registered tool | How to call",
        "---|---|---",
    ]
    for name, spec in CANONICAL_TOOL_MAP.items():
        impl = spec.get("implements_tool", "")
        tmpl = spec.get("arguments_template")
        if tmpl:
            lines.append(f"| `{name}` | `{impl}` | JSON args include {json_template(tmpl)} |")
        else:
            lines.append(f"| `{name}` | `{impl}` | See JSON schema in tool definition |")
    lines.append("")
    lines.append("Prefer **terminal** first to list, build, test, and gather signals; use **editor** for precise file IO.")
    return "\n".join(lines)


def json_template(tmpl: dict[str, Any]) -> str:
    return json.dumps(tmpl, ensure_ascii=False)


def export_registry_dict() -> dict[str, Any]:
    return {"version": 1, "tools": CANONICAL_TOOL_MAP}
