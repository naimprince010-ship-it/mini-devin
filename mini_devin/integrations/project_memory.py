"""
Project Memory — Long-term Vector Memory for Multi-Session Projects

Stores and retrieves project-scoped information that persists across sessions:
  - Architecture decisions ("we chose PostgreSQL over SQLite because…")
  - Tech stack choices ("frontend: React + Vite, backend: FastAPI")
  - Key constraints ("must support 10 k concurrent users")
  - Lessons learned from previous milestones
  - Important variable names, API contracts, data models
  - Anything the agent should NOT forget even after a week

Each project is identified by a slug (e.g. "my-saas-app").
Memory entries are stored as JSON + embeddings so semantic search works
without any external vector DB.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Memory entry types
# ─────────────────────────────────────────────────────────────────────────────

class MemoryCategory(str, Enum):
    ARCHITECTURE    = "architecture"     # tech stack, DB choice, framework
    DECISION        = "decision"         # explicit design decisions + rationale
    CONSTRAINT      = "constraint"       # hard limits, requirements
    API_CONTRACT    = "api_contract"     # endpoint shapes, data schemas
    LESSON          = "lesson"           # what went wrong / what worked
    MILESTONE       = "milestone"        # completed milestone summaries
    USER_PREFERENCE = "user_preference"  # coding style, naming conventions
    CODE_SNIPPET    = "code_snippet"     # reusable patterns
    CONTEXT         = "context"          # general project background


@dataclass
class MemoryEntry:
    id: str
    project_id: str
    category: MemoryCategory
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    importance: int = 5          # 1 (low) – 10 (critical)
    session_id: Optional[str] = None
    milestone_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["embedding"] = None    # don't serialise floats in API responses
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Tiny TF-IDF embedding (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

_DIM = 256


def _embed(text: str) -> List[float]:
    text = text.lower()
    ngrams = []
    for n in (2, 3, 4):
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i + n])
    ngrams.extend(text.split())
    vec = [0.0] * _DIM
    for g in ngrams:
        vec[hash(g) % _DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# ─────────────────────────────────────────────────────────────────────────────
# ProjectMemory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProjectInfo:
    id: str
    name: str
    description: str
    repo_url: Optional[str] = None
    tech_stack: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProjectMemory:
    """
    Persistent, searchable long-term memory for a project.

    Storage layout (under memory_dir):
        <memory_dir>/
            <project_id>/
                project.json          ← project metadata
                entries.json          ← all memory entries
    """

    def __init__(self, memory_dir: str = "project_memory"):
        self._base = Path(memory_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._projects: Dict[str, ProjectInfo] = {}
        self._entries: Dict[str, MemoryEntry] = {}   # keyed by entry id
        self._load_all()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _project_dir(self, project_id: str) -> Path:
        d = self._base / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _memory_entry_from_dict(raw: dict[str, Any]) -> Optional[MemoryEntry]:
        try:
            e = dict(raw)
            cat = e.get("category")
            if isinstance(cat, str):
                e["category"] = MemoryCategory(cat)
            return MemoryEntry(**e)
        except Exception:
            return None

    def _memory_entry_to_doc(self, e: MemoryEntry) -> dict[str, Any]:
        d = asdict(e)
        d["category"] = e.category.value
        d.pop("embedding", None)
        return d

    def _load_all(self) -> None:
        for p_dir in self._base.iterdir():
            if not p_dir.is_dir():
                continue
            p_file = p_dir / "project.json"
            e_file = p_dir / "entries.json"
            if p_file.exists():
                try:
                    data = json.loads(p_file.read_text(encoding="utf-8"))
                    proj = ProjectInfo(**data)
                    self._projects[proj.id] = proj
                except Exception:
                    pass
            if e_file.exists():
                try:
                    entries = json.loads(e_file.read_text(encoding="utf-8"))
                    for e in entries:
                        if not isinstance(e, dict):
                            continue
                        entry = self._memory_entry_from_dict(e)
                        if entry is not None:
                            self._entries[entry.id] = entry
                except Exception:
                    pass

    def _save_project(self, project_id: str) -> None:
        proj = self._projects.get(project_id)
        if proj:
            d = self._project_dir(project_id)
            (d / "project.json").write_text(
                json.dumps(proj.to_dict(), indent=2), encoding="utf-8"
            )

    def _save_entries(self, project_id: str) -> None:
        d = self._project_dir(project_id)
        entries = [self._memory_entry_to_doc(e) for e in self._entries.values() if e.project_id == project_id]
        (d / "entries.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")

    # ── Project CRUD ─────────────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        description: str = "",
        repo_url: Optional[str] = None,
        tech_stack: Optional[List[str]] = None,
        project_id: Optional[str] = None,
    ) -> ProjectInfo:
        pid = project_id or str(uuid.uuid4())[:8]
        proj = ProjectInfo(
            id=pid,
            name=name,
            description=description,
            repo_url=repo_url,
            tech_stack=tech_stack or [],
        )
        self._projects[pid] = proj
        self._save_project(pid)
        return proj

    def get_project(self, project_id: str) -> Optional[ProjectInfo]:
        return self._projects.get(project_id)

    def list_projects(self) -> List[ProjectInfo]:
        return list(self._projects.values())

    def update_project(self, project_id: str, **kwargs) -> Optional[ProjectInfo]:
        proj = self._projects.get(project_id)
        if not proj:
            return None
        for k, v in kwargs.items():
            if hasattr(proj, k):
                setattr(proj, k, v)
        proj.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_project(project_id)
        return proj

    def delete_project(self, project_id: str) -> bool:
        if project_id not in self._projects:
            return False
        import shutil
        shutil.rmtree(str(self._project_dir(project_id)), ignore_errors=True)
        del self._projects[project_id]
        # Remove associated entries
        to_remove = [eid for eid, e in self._entries.items() if e.project_id == project_id]
        for eid in to_remove:
            del self._entries[eid]
        return True

    # ── Memory entry CRUD ────────────────────────────────────────────────────

    def add_entry(
        self,
        project_id: str,
        category: MemoryCategory,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        importance: int = 5,
        session_id: Optional[str] = None,
        milestone_id: Optional[str] = None,
    ) -> MemoryEntry:
        if project_id not in self._projects:
            raise ValueError(f"Project '{project_id}' not found")
        entry = MemoryEntry(
            id=str(uuid.uuid4())[:12],
            project_id=project_id,
            category=category,
            title=title,
            content=content,
            tags=tags or [],
            importance=importance,
            session_id=session_id,
            milestone_id=milestone_id,
            embedding=_embed(f"{title} {content}"),
        )
        self._entries[entry.id] = entry
        self._save_entries(project_id)
        return entry

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(entry_id)

    def list_entries(
        self,
        project_id: str,
        category: Optional[MemoryCategory] = None,
        min_importance: int = 1,
    ) -> List[MemoryEntry]:
        return [
            e for e in self._entries.values()
            if e.project_id == project_id
            and e.importance >= min_importance
            and (category is None or e.category == category)
        ]

    def delete_entry(self, entry_id: str) -> bool:
        entry = self._entries.get(entry_id)
        if not entry:
            return False
        project_id = entry.project_id
        del self._entries[entry_id]
        self._save_entries(project_id)
        return True

    # ── Semantic search ───────────────────────────────────────────────────────

    def search(
        self,
        project_id: str,
        query: str,
        top_k: int = 5,
        category: Optional[MemoryCategory] = None,
        min_importance: int = 1,
        min_score: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over project memory.
        Returns list of {entry, score} dicts sorted by relevance.
        """
        q_emb = _embed(query)
        candidates = self.list_entries(project_id, category=category, min_importance=min_importance)
        scored = []
        for e in candidates:
            if e.embedding:
                score = _cosine(q_emb, e.embedding)
                if score >= min_score:
                    scored.append({"entry": e.to_dict(), "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_context_for_task(
        self,
        project_id: str,
        task_description: str,
        max_tokens: int = 2000,
    ) -> str:
        """
        Build a context string for injection into a new agent task.
        Retrieves the most relevant memories + all high-importance entries.
        """
        results = self.search(project_id, task_description, top_k=8)
        # Always include critical entries
        critical = [
            e for e in self.list_entries(project_id, min_importance=9)
            if e.id not in {r["entry"]["id"] for r in results}
        ]

        lines = [f"## Project Memory: {self._projects[project_id].name}\n"]

        for item in critical:
            lines.append(f"[{item.category.upper()} ★CRITICAL] {item.title}: {item.content}")

        if results:
            lines.append("\n### Relevant Context for This Task")
            for r in results:
                e = r["entry"]
                lines.append(
                    f"[{e['category'].upper()}] {e['title']}: {e['content']}"
                )

        text = "\n".join(lines)
        # Rough token limit (4 chars ≈ 1 token)
        if len(text) > max_tokens * 4:
            text = text[: max_tokens * 4] + "\n…(truncated)"
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[ProjectMemory] = None


def default_project_memory_dir() -> str:
    """Under ``PLODDER_DATA`` (default ``data``), same family as training logs — avoids cwd surprises."""
    base = os.environ.get("PLODDER_DATA") or os.environ.get("MINI_DEVIN_DATA", "data")
    return str(Path(base) / "project_memory")


def get_project_memory(memory_dir: str | None = None) -> ProjectMemory:
    global _instance
    if _instance is None:
        _instance = ProjectMemory(memory_dir or default_project_memory_dir())
    return _instance
