"""
Map entry file / language hint → Docker image + argv for ``/workspace`` layout.

``files`` keys are relative POSIX paths (as uploaded to the sandbox tar).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ToolchainSpec:
    """Resolved execution plan inside the container."""

    language_id: str
    image: str
    argv: list[str]


def _posix(rel: str) -> str:
    return rel.replace("\\", "/")


def infer_language_from_entry(entry: str, *, hint: str | None) -> str:
    if hint and hint.strip().lower() not in ("", "auto"):
        return hint.strip().lower()
    suf = PurePosixPath(entry).suffix.lower()
    if suf in (".py", ".pyw"):
        return "python"
    if suf in (".js", ".mjs", ".cjs"):
        return "javascript"
    if suf == ".ts":
        return "typescript"
    if suf == ".go":
        return "go"
    if suf == ".rs":
        return "rust"
    if suf in (".c",):
        return "c"
    if suf in (".cpp", ".cc", ".cxx", ".hpp"):
        return "cpp"
    if suf == ".sh":
        return "shell"
    return "python"


def build_toolchain_spec(
    entry: str,
    *,
    language_hint: str | None,
    python_image: str,
    node_image: str,
    go_image: str = "golang:1.22-alpine",
    rust_image: str = "rust:alpine",
    alpine_image: str = "alpine:3.19",
    cpp_image: str = "gcc:12-bookworm",
    typescript_image: str = "node:22-alpine",
) -> ToolchainSpec:
    rel = _posix(entry).lstrip("/")
    lang = infer_language_from_entry(rel, hint=language_hint)
    alias = {"js": "javascript", "ts": "typescript", "py": "python", "node": "javascript"}
    lang = alias.get(lang, lang)

    w = f"/workspace/{rel}"

    if lang == "python":
        return ToolchainSpec("python", python_image, ["python", w])
    if lang in ("javascript", "js", "node"):
        return ToolchainSpec("javascript", node_image, ["node", w])
    if lang in ("typescript", "ts"):
        # Node 22+ can execute .ts with strip types (no npm needed for simple scripts).
        return ToolchainSpec("typescript", typescript_image, ["node", "--experimental-strip-types", w])
    if lang == "go":
        return ToolchainSpec("go", go_image, ["go", "run", w])
    if lang == "rust":
        # Single-crate file: compile+run in one shot.
        return ToolchainSpec(
            "rust",
            rust_image,
            ["sh", "-c", f"rustc {w} -o /tmp/a.out && /tmp/a.out"],
        )
    if lang == "shell":
        return ToolchainSpec("shell", alpine_image, ["sh", w])
    if lang == "c":
        return ToolchainSpec(
            "c",
            cpp_image,
            ["sh", "-c", f"gcc -O2 -Wall -o /tmp/a.out {w} && /tmp/a.out"],
        )
    if lang == "cpp":
        return ToolchainSpec(
            "cpp",
            cpp_image,
            ["sh", "-c", f"g++ -std=c++17 -O2 -Wall -o /tmp/a.out {w} && /tmp/a.out"],
        )
    # default
    return ToolchainSpec("python", python_image, ["python", w])


def pick_default_entry(files: dict[str, str]) -> str | None:
    """Choose a reasonable main file from the snapshot keys."""
    if not files:
        return None
    keys = sorted(files.keys(), key=lambda k: (k.count("/"), k))
    for prefer in ("main.py", "app.py", "index.js", "index.mjs", "main.ts", "src/main.py"):
        for k in keys:
            if k.replace("\\", "/").endswith(prefer):
                return k.replace("\\", "/")
    # first python or js
    for k in keys:
        low = k.lower()
        if low.endswith((".py", ".js", ".mjs", ".ts", ".go", ".rs", ".sh", ".c", ".cpp", ".cc")):
            return k.replace("\\", "/")
    return keys[0].replace("\\", "/")


def image_for_shell_language(language: str | None, *, python_image: str, node_image: str, alpine_image: str) -> str:
    """Pick a base image that likely contains toolchain binaries (npm, pip, sh)."""
    L = (language or "auto").lower()
    if L in ("javascript", "typescript", "node", "js", "ts"):
        return node_image
    if L in ("python", "py", "pip"):
        return python_image
    return alpine_image
