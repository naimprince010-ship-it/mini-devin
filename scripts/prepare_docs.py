#!/usr/bin/env python3
"""
Copy Learn-X-in-Y-Minutes (or similar) Markdown trees from ``./temp_docs`` into
``./docs/languages`` with YAML front-matter Plodder can index (``language``, ``language_key``).

Example::

    git clone https://github.com/adambard/learnxinyminutes-docs.git temp_docs
    python scripts/prepare_docs.py
    python scripts/ingest_plodder_docs.py --docs-dir ./docs/languages
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Repo noise — not language tutorials
_SKIP_NAMES = frozenset(
    {
        "contributing.md",
        "readme.md",
        "changelog.md",
        "license",
        "code_of_conduct.md",
        "pull_request_template.md",
    }
)

_LEARN_TITLE = re.compile(
    r"^#\s*Learn\s+(.+?)\s+in\s+Y\s+Minutes",
    re.IGNORECASE | re.MULTILINE,
)


def _slug_key(name: str) -> str:
    """Stable key for filenames like ``c++.md`` → ``cpp``."""
    s = name.lower().strip()
    s = s.replace("++", "pp").replace("#", "sharp")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "unknown"


def _display_from_stem(stem: str) -> str:
    return stem.replace("__", " / ").replace("-", " ").replace("_", " ").strip().title()


def _extract_display_name(content: str, stem: str) -> str:
    m = _LEARN_TITLE.search(content)
    if m:
        return m.group(1).strip()
    return _display_from_stem(stem)


def _strip_existing_front_matter(text: str) -> str:
    if not text.lstrip().startswith("---"):
        return text
    rest = text.lstrip()[3:]
    if not rest.startswith("\n"):
        return text
    end = rest.find("\n---")
    if end == -1:
        return text
    # closing --- line
    close = rest.find("\n", end + 1)
    if close == -1:
        return text
    return rest[close + 1 :].lstrip("\n")


def _clean_body(text: str) -> str:
    t = _strip_existing_front_matter(text)
    # Collapse excessive blank lines
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip() + "\n"


def _safe_out_name(rel: Path) -> str:
    """Flat unique name: zh-cn/rust.md -> zh-cn__rust.md"""
    parts = rel.as_posix().replace("/", "__")
    if parts.endswith(".markdown"):
        return parts[: -len(".markdown")] + ".md"
    return parts if parts.endswith(".md") else parts + ".md"


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare Learn-X docs for Plodder RAG")
    p.add_argument("--src", type=Path, default=Path("temp_docs"), help="Cloned learnxinyminutes-docs root")
    p.add_argument("--dst", type=Path, default=Path("docs/languages"), help="Output directory under repo")
    p.add_argument(
        "--locale-subdirs",
        action="store_true",
        help="Only copy markdown directly under locale folders (e.g. temp_docs/zh-cn/*.md). "
        "If false, copy all *.md / *.markdown under src except skip-list.",
    )
    args = p.parse_args()
    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()

    if not src.is_dir():
        print(f"ERROR: --src is not a directory: {src}", file=sys.stderr)
        print("Clone learnxinyminutes-docs into ./temp_docs first.", file=sys.stderr)
        return 1

    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    patterns = ("*.md", "*.markdown")
    files: list[Path] = []
    if args.locale_subdirs:
        for sub in sorted(src.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            for pat in patterns:
                files.extend(sorted(sub.glob(pat)))
    else:
        for pat in patterns:
            files.extend(sorted(src.rglob(pat)))

    seen_out: set[str] = set()
    for path in sorted(set(files), key=lambda x: str(x).lower()):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if rel.parts and rel.parts[0].startswith("."):
            skipped += 1
            continue
        if path.name.lower() in _SKIP_NAMES:
            skipped += 1
            continue
        if ".github" in rel.parts:
            skipped += 1
            continue

        raw = path.read_text(encoding="utf-8", errors="replace")
        body = _clean_body(raw)
        stem_for_title = path.stem.replace(".html", "")  # e.g. foo.html.markdown
        display = _extract_display_name(body, stem_for_title)
        # Prefer filesystem stem for stable keys (c++.md → cpp)
        lang_key = _slug_key(stem_for_title)

        out_name = _safe_out_name(rel.with_suffix(".md"))
        if out_name in seen_out:
            base = out_name[:-3]
            out_name = f"{base}__dup{copied}.md"
        seen_out.add(out_name)
        out_path = dst / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fm = (
            "---\n"
            f"language: {display}\n"
            f"language_key: {lang_key}\n"
            f"source_repo_path: {rel.as_posix()}\n"
            "---\n\n"
        )
        out_path.write_text(fm + body, encoding="utf-8", newline="\n")
        copied += 1

    print(f"Prepared {copied} files into {dst} (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
