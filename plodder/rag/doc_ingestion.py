"""
Local Markdown RAG for Plodder: index ``./docs``, **LanceDB** + lightweight hash embeddings, watchdog.

Avoids ChromaDB's ``tokenizers`` pin that conflicts with LiteLLM in the same venv.

``UniversalPromptEngine.language_docs_retrieval_block(store, ...)`` then
``coder_user_prompt(..., retrieved_context=...)``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_EMB_DIM = 256


def _hash_bow_embed(text: str, dim: int = _EMB_DIM) -> np.ndarray:
    """
    Deterministic bag-of-tokens embedding (no ML deps). Good for cheat-sheet keyword overlap.
    """
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[\w\.#\+\-]+", text.lower()):
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16) % dim
        v[h] += 1.0
    n = float(np.linalg.norm(v)) or 1.0
    v /= n
    return v


# ── chunking ────────────────────────────────────────────────────────────────


def _chunk_markdown(text: str, *, max_chars: int = 1400, overlap: int = 160) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + max_chars)
        window = text[i:end]
        if end < n:
            br = window.rfind("\n\n")
            if br > max_chars // 3:
                window = window[: br + 2]
                end = i + len(window)
        chunks.append(window.strip())
        if end >= n:
            break
        i = max(i + 1, end - overlap)
    return [c for c in chunks if c]


def unfamiliar_stack_query(
    *,
    target_language: str,
    symptoms: str = "",
    stack_keywords: list[str] | None = None,
) -> str:
    """
    Build a retrieval query when Plodder is unsure about a stack.
    Pass error snippets, crate names, or framework names in ``symptoms`` / ``stack_keywords``.
    """
    parts = [
        f"Language and ecosystem: {target_language}",
        "idiomatic patterns, syntax, modules, error handling, project layout, tooling",
    ]
    if stack_keywords:
        parts.append("Keywords: " + ", ".join(stack_keywords))
    if symptoms.strip():
        parts.append("Context / errors:\n" + symptoms.strip()[:4000])
    return "\n".join(parts)


def _lance_escape(value: str) -> str:
    return value.replace("'", "''")


def _slug_language_key(stem: str) -> str:
    """Normalize a filename stem or label to a Lance-safe filter token (``rust``, ``cpp``)."""
    s = stem.lower().strip().replace(".html", "")
    s = s.replace("++", "pp").replace("#", "sharp")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "unknown"


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse simple ``key: value`` YAML front matter; return ``(meta, body)``."""
    text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end].strip()
    body = text[end + 4 :].lstrip()
    meta: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    return meta, body


def _language_defaults_from_source(source_key: str) -> tuple[str, str]:
    """``(display_name, language_key)`` when front matter is missing."""
    base = source_key.rsplit("/", 1)[-1]
    stem = Path(base).stem
    if "__" in stem:
        stem = stem.split("__")[-1]
    lk = _slug_language_key(stem)
    disp = stem.replace("-", " ").replace("_", " ").strip().title()
    return disp, lk


# ── store ───────────────────────────────────────────────────────────────────


class DocumentationStore:
    """
    LanceDB-backed store for Markdown under ``docs_dir``.

    - Each chunk stores ``language`` (display) and ``language_key`` (slug) for filtered search.
    - ``index_all`` / ``index_file`` chunk body text + hash embeddings (YAML front matter parsed).
    - ``retrieve`` / ``format_retrieval_block`` / ``format_retrieval_block_for_language`` for prompts.
    - ``start_watcher`` uses Watchdog to reindex on create/modify/delete.
    """

    _TABLE = "plodder_docs"

    def __init__(
        self,
        docs_dir: str | Path = "docs",
        persist_dir: str | Path = "data/plodder_lance",
    ) -> None:
        self.docs_dir = Path(docs_dir).resolve()
        self.persist_dir = Path(persist_dir).resolve()
        self._db: Any = None
        self._table: Any = None
        self._lock = threading.RLock()
        self._observer: Any = None
        self._observer_thread: threading.Thread | None = None
        self._debounce_sec = 1.0
        self._timers: dict[str, threading.Timer] = {}
        self._timers_lock = threading.Lock()

    def _ensure_db(self) -> Any:
        if self._db is not None and self._table is not None:
            return self._db
        try:
            import lancedb
            from lancedb.pydantic import LanceModel, Vector
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "lancedb is required for DocumentationStore. Install with: pip install lancedb"
            ) from e

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if self._db is None:
            self._db = lancedb.connect(str(self.persist_dir))

        class _Chunk(LanceModel):
            vector: Vector(_EMB_DIM)
            text: str
            source: str
            language: str = ""
            language_key: str = ""
            chunk_index: int = 0
            indexed_at: float = 0.0

        self._ChunkModel = _Chunk  # type: ignore[attr-defined]

        def _schema_ok(tbl: Any) -> bool:
            try:
                names = tbl.schema.names
            except Exception:
                return False
            return "language_key" in names and "language" in names

        self._table = None
        try:
            t = self._db.open_table(self._TABLE)
            if _schema_ok(t):
                self._table = t
            else:
                logger.warning("dropping legacy Lance table %s (missing language columns)", self._TABLE)
                self._db.drop_table(self._TABLE)
        except Exception:
            pass

        if self._table is None:
            dummy = _Chunk(
                vector=_hash_bow_embed("__plodder_schema_init__").tolist(),
                text="",
                source="__init__",
                language="",
                language_key="__init__",
                chunk_index=-1,
                indexed_at=0.0,
            )
            self._table = self._db.create_table(self._TABLE, [dummy], mode="overwrite")
            self._table.delete("source = '__init__'")

        return self._db

    def _rel_source(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.docs_dir).as_posix()
        except ValueError:
            return path.name

    def _delete_chunks_for_source(self, source_key: str) -> None:
        self._ensure_db()
        assert self._table is not None
        safe = _lance_escape(source_key)
        try:
            self._table.delete(f"source = '{safe}'")
        except Exception as e:
            logger.debug("delete chunks (maybe none): %s", e)

    def index_file(self, path: Path) -> int:
        path = path.resolve()
        if path.suffix.lower() != ".md":
            return 0
        if not path.is_file():
            return 0
        try:
            path.relative_to(self.docs_dir.resolve())
        except ValueError:
            logger.warning("skip index outside docs_dir: %s", path)
            return 0

        raw = path.read_text(encoding="utf-8", errors="replace")
        source_key = self._rel_source(path)
        meta, body = _parse_front_matter(raw)
        disp_default, key_default = _language_defaults_from_source(source_key)
        language_display = (meta.get("language") or disp_default).strip()
        language_key = (meta.get("language_key") or key_default).strip().lower()
        language_key = _slug_language_key(language_key)

        chunks = _chunk_markdown(body)
        with self._lock:
            self._ensure_db()
            assert self._table is not None
            self._delete_chunks_for_source(source_key)
            if not chunks:
                return 0
            rows: list[Any] = []
            now = time.time()
            CM = self._ChunkModel
            for i, chunk in enumerate(chunks):
                rows.append(
                    CM(
                        vector=_hash_bow_embed(chunk).tolist(),
                        text=chunk,
                        source=source_key,
                        language=language_display,
                        language_key=language_key,
                        chunk_index=i,
                        indexed_at=now,
                    )
                )
            self._table.add(rows)
        logger.info("indexed %s (%d chunks) [%s]", source_key, len(chunks), language_key)
        return len(chunks)

    def remove_file(self, path: Path) -> None:
        source_key = self._rel_source(Path(path))
        with self._lock:
            self._ensure_db()
            self._delete_chunks_for_source(source_key)

    def index_all(self) -> int:
        if not self.docs_dir.is_dir():
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            return 0
        total = 0
        for md in sorted(self.docs_dir.rglob("*.md")):
            if md.name.startswith("."):
                continue
            total += self.index_file(md)
        return total

    def retrieve(
        self,
        query: str,
        n_results: int = 6,
        *,
        language_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """List of rows: text, source, language, language_key, chunk_index, _distance."""
        with self._lock:
            self._ensure_db()
            assert self._table is not None
            n = max(1, min(n_results, 50))
            qtext = query
            if language_key:
                lk = _slug_language_key(language_key)
                qtext = (
                    f"{lk} programming language syntax rules standard library idioms\n" + (query or "")
                )
            qv = _hash_bow_embed(qtext)
            search = self._table.search(qv.tolist()).limit(n)
            if language_key:
                safe = _lance_escape(_slug_language_key(language_key))
                try:
                    search = search.where(f"language_key = '{safe}'", prefilter=True)
                except TypeError:
                    # older LanceDB: fall back to unfiltered search
                    logger.warning("LanceDB where() not available; search without language filter")
            return search.to_list()

    def format_retrieval_block(
        self,
        query: str,
        *,
        n_results: int = 6,
        max_chars: int = 8000,
        language_key: str | None = None,
    ) -> str:
        rows = self.retrieve(query, n_results=n_results, language_key=language_key)
        if not rows:
            return ""
        lines: list[str] = []
        acc = 0
        for i, row in enumerate(rows):
            if acc > max_chars:
                break
            src = row.get("source", "?")
            lang = row.get("language") or row.get("language_key") or "?"
            dist = row.get("_distance", None)
            doc = (row.get("text") or "").strip()
            head = f"### Snippet {i + 1} (`{lang}` — `{src}`)"
            if dist is not None:
                head += f" — distance `{float(dist):.4f}`"
            block = f"{head}\n```markdown\n{doc}\n```\n"
            lines.append(block)
            acc += len(block)
        return "\n".join(lines).strip()

    def format_retrieval_block_for_language(
        self,
        user_task: str,
        *,
        language_display: str | None = None,
        language_key: str | None = None,
        n_results: int = 8,
        max_chars: int = 8000,
    ) -> str:
        """
        Build a markdown block for ``coder_user_prompt(..., retrieved_context=...)``.

        Pass ``language_key`` (e.g. ``rust``) to restrict chunks to that language; the query
        vector also biases toward rules/syntax for that stack.
        """
        parts: list[str] = []
        if language_display:
            parts.append(f"Target stack: {language_display}")
        if language_key:
            parts.append(f"language_key: {_slug_language_key(language_key)}")
        q = "\n".join(parts + [user_task]).strip()
        return self.format_retrieval_block(
            q,
            n_results=n_results,
            max_chars=max_chars,
            language_key=language_key,
        )

    def has_language_key(self, language_key: str) -> bool:
        """True if at least one chunk exists for this ``language_key`` slug."""
        rows = self.retrieve("syntax overview", n_results=1, language_key=language_key)
        return len(rows) > 0

    # ── watcher ────────────────────────────────────────────────────────────

    def _schedule_reindex(self, path: Path, *, is_delete: bool) -> None:
        key = str(path.resolve())

        def job() -> None:
            with self._timers_lock:
                self._timers.pop(key, None)
            try:
                if is_delete or not path.exists():
                    self.remove_file(path)
                    logger.info("removed from index: %s", key)
                else:
                    self.index_file(path)
            except Exception:
                logger.exception("watch handler failed for %s", key)

        with self._timers_lock:
            old = self._timers.pop(key, None)
            if old:
                old.cancel()
            t = threading.Timer(self._debounce_sec, job)
            self._timers[key] = t
            t.start()

    def start_watcher(
        self,
        *,
        debounce_sec: float = 1.0,
        reindex_all_first: bool = True,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "watchdog is required for start_watcher. Install with: pip install watchdog"
            ) from e

        self._debounce_sec = debounce_sec
        if reindex_all_first:
            self.index_all()

        store = self

        class _Handler(FileSystemEventHandler):
            def _handle(self, src: str, *, is_delete: bool) -> None:
                p = Path(src)
                if p.name.startswith("."):
                    return
                if not is_delete and not p.suffix.lower() == ".md":
                    return
                if is_delete and not p.suffix.lower() == ".md":
                    return
                store._schedule_reindex(p, is_delete=is_delete)

            def on_created(self, event: Any) -> None:
                if not event.is_directory:
                    self._handle(event.src_path, is_delete=False)

            def on_modified(self, event: Any) -> None:
                if not event.is_directory:
                    self._handle(event.src_path, is_delete=False)

            def on_deleted(self, event: Any) -> None:
                if not event.is_directory:
                    self._handle(event.src_path, is_delete=True)

            def on_moved(self, event: Any) -> None:
                if event.is_directory:
                    return
                if str(event.src_path).lower().endswith(".md"):
                    self._handle(event.src_path, is_delete=True)
                if getattr(event, "dest_path", None) and str(event.dest_path).lower().endswith(".md"):
                    self._handle(event.dest_path, is_delete=False)

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_db()
        handler = _Handler()
        observer = Observer()
        observer.schedule(handler, str(self.docs_dir), recursive=True)
        self._observer = observer

        def _run() -> None:
            observer.start()
            observer.join()

        t = threading.Thread(target=_run, name="plodder-docs-watcher", daemon=True)
        self._observer_thread = t
        t.start()
        logger.info("documentation watcher started on %s", self.docs_dir)
        if on_ready:
            on_ready()

    def stop_watcher(self) -> None:
        obs = self._observer
        if obs is None:
            return
        try:
            obs.stop()
        except Exception:
            pass
        self._observer = None
        self._observer_thread = None
