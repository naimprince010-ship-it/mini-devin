"""
Scan a checked-out repository on disk and build a text digest for Project Memory.

Used to seed long-term project context so later sessions (with the same ``project_id``)
retrieve it via ``ProjectMemory.get_context_for_task`` / API injection.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
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

CODE_FILE_SUFFIXES = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
    }
)

CONFIG_FILE_NAMES = frozenset(
    {
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "Cargo.toml",
        "go.mod",
        "turbo.json",
        "tsconfig.json",
        "vite.config.ts",
        "vite.config.js",
    }
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


def _read_text_for_analysis(path: Path, max_chars: int = 80_000) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return raw[:max_chars]


def _path_priority(rel: str) -> tuple[int, str]:
    name = rel.rsplit("/", 1)[-1]
    if name in CONFIG_FILE_NAMES:
        return (0, rel)
    if "schema.prisma" in rel or "/api/" in rel or rel.startswith("api/"):
        return (1, rel)
    if rel.endswith((".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx", "_test.py", "_test.go")):
        return (2, rel)
    if rel.startswith(("apps/", "packages/", "src/", "lib/", "mini_devin/", "plodder/")):
        return (3, rel)
    return (9, rel)


def _code_paths(paths: list[str], *, max_files: int = 120) -> list[str]:
    code = [p for p in paths if Path(p).suffix.lower() in CODE_FILE_SUFFIXES]
    return sorted(code, key=_path_priority)[:max_files]


def _top_dirs(paths: list[str], depth: int = 2, limit: int = 30) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for rel in paths:
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        key = "/".join(parts[: min(depth, len(parts) - 1)])
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def _json_load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(_read_text_for_analysis(path, 120_000))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _mapping_keys(value: Any) -> list[str]:
    return sorted(value.keys()) if isinstance(value, dict) else []


def _build_dependency_summary(root: Path, paths: list[str], *, max_package_files: int = 12) -> str:
    sections: list[str] = []
    package_paths = [p for p in paths if p.endswith("package.json")]
    package_paths = sorted(package_paths, key=lambda p: (0 if p == "package.json" else 1, p))[:max_package_files]
    if package_paths:
        lines = ["### JavaScript/TypeScript packages"]
        for rel in package_paths:
            data = _json_load(root / rel)
            if not data:
                continue
            deps = _mapping_keys(data.get("dependencies"))
            dev = _mapping_keys(data.get("devDependencies"))
            scripts = _mapping_keys(data.get("scripts"))
            pkg_name = data.get("name") or rel
            pm = data.get("packageManager")
            workspaces = data.get("workspaces")
            bits = [f"- `{rel}`: `{pkg_name}`"]
            if pm:
                bits.append(f"  - packageManager: `{pm}`")
            if workspaces:
                bits.append(f"  - workspaces: `{workspaces}`")
            if scripts:
                bits.append(f"  - scripts: {', '.join(f'`{s}`' for s in scripts[:20])}")
            if deps:
                bits.append(f"  - dependencies: {', '.join(f'`{d}`' for d in deps[:25])}")
            if dev:
                bits.append(f"  - devDependencies: {', '.join(f'`{d}`' for d in dev[:18])}")
            sections.append("\n".join(bits))
        if len(sections) > 1:
            sections.insert(0, lines[0])

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = _read_text_for_analysis(pyproject, 20_000)
        interesting = [
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith(("name =", "requires-python", "dependencies =", "[tool.poetry", "[project", "[tool.pytest"))
        ][:40]
        if interesting:
            sections.append("### Python project hints\n" + "\n".join(f"- `{line}`" for line in interesting))

    go_mod = root / "go.mod"
    if go_mod.is_file():
        lines = [line.strip() for line in _read_text_for_analysis(go_mod, 20_000).splitlines() if line.strip()]
        sections.append("### Go module hints\n" + "\n".join(f"- `{line}`" for line in lines[:40]))

    if not sections:
        return "_No dependency manifests found in scanned files._\n"
    return "\n\n".join(sections) + "\n"


_TS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:type\s+)?(?:[^'"]+?\s+from\s+)?|export\s+[^'"]+?\s+from\s+|require\()\s*['"]([^'"]+)['"]""",
    re.MULTILINE,
)
_PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+.+|import\s+([\w.]+))", re.MULTILINE)


def _extract_imports(rel: str, text: str) -> list[str]:
    suffix = Path(rel).suffix.lower()
    imports: list[str] = []
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        imports = [m.group(1) for m in _TS_IMPORT_RE.finditer(text)]
    elif suffix == ".py":
        for m in _PY_IMPORT_RE.finditer(text):
            imports.append(m.group(1) or m.group(2) or "")
    return [i for i in imports if i][:20]


def _build_import_graph(root: Path, paths: list[str], *, max_files: int = 80) -> str:
    lines: list[str] = []
    for rel in _code_paths(paths, max_files=max_files):
        text = _read_text_for_analysis(root / rel, 80_000)
        imports = _extract_imports(rel, text)
        if not imports:
            continue
        local = [i for i in imports if i.startswith((".", "@/")) or i.startswith(("mini_devin", "plodder", "src/", "lib/"))]
        external = [i for i in imports if i not in local]
        shown = local[:10] + external[:6]
        lines.append(f"- `{rel}` -> " + ", ".join(f"`{i}`" for i in shown))
        if len(lines) >= 40:
            break
    return "\n".join(lines) + ("\n" if lines else "_No import edges extracted from selected code files._\n")


def _build_symbol_summary(root: Path, paths: list[str], *, max_files: int = 80, max_symbols: int = 160) -> str:
    try:
        from ..memory.symbol_index import create_symbol_index
    except Exception:
        return "_Symbol index unavailable._\n"

    index = create_symbol_index(str(root))
    for rel in _code_paths(paths, max_files=max_files):
        index.index_file(str(root / rel))
        if len(index.symbols) >= max_symbols:
            break

    stats = index.get_statistics()
    lines = [
        f"- files indexed for symbols: `{stats['files_indexed']}`",
        f"- total symbols: `{stats['total_symbols']}`",
    ]
    nonzero_types = {k: v for k, v in stats["symbols_by_type"].items() if v}
    if nonzero_types:
        lines.append("- by type: " + ", ".join(f"`{k}`={v}" for k, v in sorted(nonzero_types.items())))

    shown = list(index.symbols.values())[:max_symbols]
    if shown:
        lines.append("")
        lines.append("Key symbols:")
    for sym in shown[:80]:
        lines.append(
            f"- `{sym.symbol_type.value}` `{sym.qualified_name}` at `{sym.location.file_path}:{sym.location.start_line}`"
        )
    if len(shown) > 80:
        lines.append(f"- ... {len(shown) - 80} more symbols omitted")
    return "\n".join(lines) + "\n"


def _chunk_file(rel: str, text: str, *, chunk_lines: int = 80, max_chunks: int = 2) -> list[str]:
    lines = text.splitlines()
    chunks: list[str] = []
    for idx in range(0, min(len(lines), chunk_lines * max_chunks), chunk_lines):
        body = "\n".join(lines[idx : idx + chunk_lines]).strip()
        if not body:
            continue
        start = idx + 1
        end = min(idx + chunk_lines, len(lines))
        chunks.append(f"### `{rel}:{start}-{end}`\n\n```\n{body[:8_000]}\n```\n")
    return chunks


def _build_code_chunks(root: Path, paths: list[str], *, max_files: int = 12) -> str:
    interesting = _code_paths(paths, max_files=max_files)
    chunks: list[str] = []
    for rel in interesting:
        text = _read_text_for_analysis(root / rel, 40_000)
        chunks.extend(_chunk_file(rel, text, max_chunks=1))
    return "\n".join(chunks) if chunks else "_No source chunks selected._\n"


def build_repo_digest(
    root: Path,
    *,
    max_total_chars: int = 100_000,
    max_chars_per_file: int = 24_000,
    max_inventory_paths: int = 6_000,
) -> dict[str, Any]:
    """
    Build a markdown digest: key manifest files, dependency/code intelligence,
    selected source chunks, and file inventory.

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

    top_dirs = _top_dirs(paths)
    if top_dirs and budget > 2_000:
        section = "## Repository shape\n\n" + "\n".join(f"- `{name}`: {count} files" for name, count in top_dirs) + "\n\n"
        sections.append(section)
        budget -= len(section)

    if budget > 6_000:
        dep = "## Dependency and command map\n\n" + _build_dependency_summary(root, paths) + "\n"
        if len(dep) > budget * 0.25:
            dep = dep[: int(budget * 0.25)] + "\n...(dependency map truncated)\n\n"
            warnings.append("dependency_map_truncated")
        sections.append(dep)
        budget -= len(dep)

    if budget > 8_000:
        symbols = "## Symbol map\n\n" + _build_symbol_summary(root, paths) + "\n"
        if len(symbols) > budget * 0.25:
            symbols = symbols[: int(budget * 0.25)] + "\n...(symbol map truncated)\n\n"
            warnings.append("symbol_map_truncated")
        sections.append(symbols)
        budget -= len(symbols)

    if budget > 6_000:
        graph = "## Import/dependency edges\n\n" + _build_import_graph(root, paths) + "\n"
        if len(graph) > budget * 0.18:
            graph = graph[: int(budget * 0.18)] + "\n...(import graph truncated)\n\n"
            warnings.append("import_graph_truncated")
        sections.append(graph)
        budget -= len(graph)

    if budget > 12_000:
        chunks = "## Selected code chunks\n\n" + _build_code_chunks(root, paths) + "\n"
        if len(chunks) > budget * 0.35:
            chunks = chunks[: int(budget * 0.35)] + "\n...(code chunks truncated)\n\n"
            warnings.append("code_chunks_truncated")
        sections.append(chunks)
        budget -= len(chunks)

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
