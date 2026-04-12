"""
Scan a checked-out repository on disk and build a text digest for Project Memory.

Used to seed long-term project context so later sessions (with the same ``project_id``)
retrieve it via ``ProjectMemory.get_context_for_task`` / API injection.
"""

from __future__ import annotations

import fnmatch
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".eggs",
        ".next",
        "coverage",
        "htmlcov",
    }
)


def _skip_path_parts(parts: tuple[str, ...]) -> bool:
    for name in parts:
        if name in SKIP_DIR_NAMES:
            return True
        if name.endswith(".egg-info"):
            return True
    return False

SKIP_FILE_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".dylib",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".map",
)

# Prefer these paths first (relative, case-sensitive then case-insensitive check).
MANIFEST_RELATIVE = (
    "README.md",
    "Readme.md",
    "README.rst",
    "README.txt",
    "CONTRIBUTING.md",
    "LICENSE",
    "LICENSE.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "requirements.txt",
    "Pipfile",
    "setup.py",
    "setup.cfg",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Dockerfile",
    "Makefile",
    "turbo.json",
    "tsconfig.json",
    "vite.config.ts",
    "vite.config.js",
)


def _load_gitignore_patterns(root: Path) -> list[str]:
    gi = root / ".gitignore"
    if not gi.is_file():
        return []
    patterns: list[str] = []
    for line in gi.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        patterns.append(s)
    return patterns


def _ignored_path(rel_posix: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if "/" in pat or pat.startswith("*"):
            if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(rel_posix, pat.lstrip("/")):
                return True
        else:
            if rel_posix == pat or rel_posix.endswith("/" + pat):
                return True
    return False


def _collect_relative_paths(root: Path, *, max_paths: int) -> list[str]:
    patterns = _load_gitignore_patterns(root)
    out: list[str] = []
    root = root.resolve()

    for p in root.rglob("*"):
        if len(out) >= max_paths:
            break
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        if _skip_path_parts(p.parts):
            continue
        if p.is_file():
            if p.name.startswith(".") and p.name not in (".env.example",):
                continue
            if p.suffix.lower() in SKIP_FILE_SUFFIXES:
                continue
            if _ignored_path(rel, patterns):
                continue
            out.append(rel)
    out.sort()
    return out


def _read_text_capped(path: Path, max_chars: int) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(raw) > max_chars:
        return raw[:max_chars] + "\n…(truncated)\n"
    return raw


def build_repo_digest(
    root: Path,
    *,
    max_total_chars: int = 100_000,
    max_chars_per_file: int = 24_000,
    max_inventory_paths: int = 6_000,
) -> dict[str, Any]:
    """
    Build a markdown digest: key manifest files + full path inventory.

    Returns keys: markdown, paths_count, manifest_files_used, warnings.
    """
    root = root.resolve()
    warnings: list[str] = []
    if not root.is_dir():
        return {
            "markdown": "",
            "paths_count": 0,
            "manifest_files_used": [],
            "warnings": ["not_a_directory"],
        }

    sections: list[str] = []
    used_manifest: list[str] = []
    budget = max_total_chars

    header = (
        f"# Repository snapshot\n\n"
        f"- **Root**: `{root}`\n"
        f"- **Generated (UTC)**: {datetime.now(timezone.utc).isoformat()}\n\n"
    )
    sections.append(header)
    budget -= len(header)

    # Manifest: exact names at repo root only (fast, stable).
    manifest_body = "## Key files (repository root)\n\n"
    chunk_manifest = ""
    for name in MANIFEST_RELATIVE:
        fp = root / name
        if not fp.is_file():
            # case-insensitive fallback for README etc.
            lower = name.lower()
            found: Path | None = None
            for c in root.iterdir():
                if c.is_file() and c.name.lower() == lower:
                    found = c
                    break
            fp = found if found else fp
        if not fp.is_file():
            continue
        text = _read_text_capped(fp, max_chars_per_file)
        block = f"### `{fp.relative_to(root).as_posix()}`\n\n```\n{text}\n```\n\n"
        if len(chunk_manifest) + len(block) > budget * 0.65:
            warnings.append("manifest_truncated_budget")
            break
        chunk_manifest += block
        used_manifest.append(fp.relative_to(root).as_posix())

    manifest_body += chunk_manifest or "_No common manifest files found at repo root._\n\n"
    sections.append(manifest_body)
    budget -= len(manifest_body)

    paths = _collect_relative_paths(root, max_paths=max_inventory_paths)
    inv_header = f"## File inventory ({len(paths)} paths)\n\n```\n"
    inv_footer = "\n```\n"
    inv_lines = "\n".join(paths)
    inventory = inv_header + inv_lines + inv_footer
    if len(inventory) > max(8_000, budget - 500):
        cut = max(8_000, budget - 500) - len(inv_header) - len(inv_footer) - 80
        inventory = inv_header + inv_lines[:cut] + "\n…(inventory truncated)\n" + inv_footer
        warnings.append("inventory_truncated")

    sections.append(inventory)

    markdown = "".join(sections)
    if len(markdown) > max_total_chars:
        markdown = markdown[: max_total_chars - 40] + "\n\n…(digest hard truncated)\n"
        warnings.append("digest_hard_truncated")

    return {
        "markdown": markdown,
        "paths_count": len(paths),
        "manifest_files_used": used_manifest,
        "warnings": warnings,
    }


def is_repo_path_under_workspace(repo: Path, workspace_parent: Path) -> bool:
    """True if ``repo`` is the same as or inside ``workspace_parent`` (resolved)."""
    try:
        r = repo.resolve()
        w = workspace_parent.resolve()
        r.relative_to(w)
        return True
    except (ValueError, OSError):
        return False


def ingest_allowlist_roots() -> list[Path]:
    """
    Directories under which ``repo_path`` is accepted for ingest (server-side safety).

    - ``../agent-workspace`` next to the Plodder repo (session clones live here).
    - The Plodder checkout itself (for self-host / dev).
    - Optional ``REPO_INGEST_EXTRA_ROOT`` absolute path for custom layouts.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    ws = (repo_root.parent / "agent-workspace").resolve()
    roots: list[Path] = [ws, repo_root.resolve()]
    extra = (os.environ.get("REPO_INGEST_EXTRA_ROOT") or "").strip()
    if extra:
        roots.append(Path(extra).resolve())
    return roots


def is_path_allowed_for_repo_ingest(repo: Path) -> bool:
    if not repo.is_dir():
        return False
    try:
        t = repo.resolve()
    except OSError:
        return False
    for base in ingest_allowlist_roots():
        try:
            t.relative_to(base.resolve())
            return True
        except ValueError:
            continue
    return False
