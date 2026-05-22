"""Chunk-level retrieval index over Project Memory repository ingests."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .project_memory import _cosine, _embed


STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "app",
    "code",
    "for",
    "framework",
    "in",
    "of",
    "on",
    "or",
    "project",
    "server",
    "the",
    "to",
    "with",
}


@dataclass
class RetrievalDoc:
    project_id: str
    repo: str
    entry_id: str
    chunk_id: str
    chunk_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    tokens: set[str] | None = None

    def prepare(self, *, embed: bool = True) -> "RetrievalDoc":
        text = self.search_text()
        self.tokens = self.tokens or tokenize(text)
        if embed:
            self.embedding = self.embedding or _embed(text)
        return self

    def search_text(self) -> str:
        return "\n".join([self.repo, self.title, self.chunk_type, self.content])

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["tokens"] = sorted(self.tokens or [])
        return d

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "RetrievalDoc":
        d = dict(raw)
        toks = d.get("tokens")
        d["tokens"] = set(toks) if isinstance(toks, list) else None
        return cls(**d)


def tokenize(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_.+-]{1,}", text.lower())
    original_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{1,}", text)
    out: set[str] = set()
    for token in raw:
        if token in STOPWORDS:
            continue
        out.add(token)
        for part in re.split(r"[-_/.:+]", token):
            if len(part) >= 2 and part not in STOPWORDS:
                out.add(part)
    for token in original_tokens:
        for part in re.split(r"[-_/.:+]", token):
            for word in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part):
                low = word.lower()
                if len(low) >= 2 and low not in STOPWORDS:
                    out.add(low)
    return out


def _repo_from_project(project: dict[str, Any], project_id: str) -> str:
    name = str(project.get("name") or "")
    if "/" in name:
        return name
    if project_id.startswith("gh-"):
        return project_id[3:].replace("-", "/")
    return project_id


def _chunk_text(text: str, *, max_chars: int = 3500) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    lines = text.splitlines()
    cur: list[str] = []
    cur_len = 0
    for line in lines:
        if cur and cur_len + len(line) + 1 > max_chars:
            chunks.append("\n".join(cur).strip())
            cur = []
            cur_len = 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c]


def _section_type(title: str) -> str:
    low = title.lower()
    if "dependency" in low or "command" in low:
        return "dependency_map"
    if "symbol" in low:
        return "symbol_map"
    if "import" in low:
        return "import_edges"
    if "selected code" in low:
        return "code_section"
    if "file inventory" in low:
        return "file_inventory"
    if "key files" in low:
        return "key_files"
    if "repository shape" in low:
        return "repository_shape"
    return "section"


_HEADING_RE = re.compile(r"(?m)^\s*(#{1,3})\s+(.+?)\s*$")


def docs_from_digest(
    *,
    project_id: str,
    repo: str,
    entry_id: str,
    title: str,
    content: str,
    max_section_chars: int = 3500,
) -> list[RetrievalDoc]:
    docs: list[RetrievalDoc] = []
    headings = list(_HEADING_RE.finditer(content))
    for idx, match in enumerate(headings):
        level, heading = match.group(1), match.group(2).strip()
        start = match.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(content)
        body = content[start:end].strip()
        if not body:
            continue

        section_kind = _section_type(heading)
        if level == "###" and "`" in heading:
            section_kind = "file_chunk"

        for part_no, chunk in enumerate(_chunk_text(body, max_chars=max_section_chars), start=1):
            docs.append(
                RetrievalDoc(
                    project_id=project_id,
                    repo=repo,
                    entry_id=entry_id,
                    chunk_id=f"{entry_id}:{idx}:{part_no}",
                    chunk_type=section_kind,
                    title=heading,
                    content=chunk,
                    metadata={"source_title": title},
                ).prepare(embed=False)
            )

    if not docs and content.strip():
        for part_no, chunk in enumerate(_chunk_text(content, max_chars=max_section_chars), start=1):
            docs.append(
                RetrievalDoc(
                    project_id=project_id,
                    repo=repo,
                    entry_id=entry_id,
                    chunk_id=f"{entry_id}:body:{part_no}",
                    chunk_type="body",
                    title=title,
                    content=chunk,
                ).prepare(embed=False)
            )
    return docs


def load_project_memory_docs(memory_dir: Path, *, max_section_chars: int = 3500) -> list[RetrievalDoc]:
    docs: list[RetrievalDoc] = []
    if not memory_dir.is_dir():
        return docs
    for project_dir in sorted(p for p in memory_dir.iterdir() if p.is_dir()):
        project_file = project_dir / "project.json"
        entries_file = project_dir / "entries.json"
        if not project_file.is_file() or not entries_file.is_file():
            continue
        try:
            project = json.loads(project_file.read_text(encoding="utf-8"))
            entries = json.loads(entries_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        repo = _repo_from_project(project, project_dir.name)
        project_text = "\n".join(
            [
                f"Repository: {repo}",
                f"Description: {project.get('description') or ''}",
                f"Tech stack: {', '.join(str(x) for x in (project.get('tech_stack') or []))}",
                f"URL: {project.get('repo_url') or ''}",
            ]
        )
        docs.append(
            RetrievalDoc(
                project_id=project_dir.name,
                repo=repo,
                entry_id=f"{project_dir.name}:project",
                chunk_id=f"{project_dir.name}:project",
                chunk_type="project_metadata",
                title=f"Project metadata: {repo}",
                content=project_text,
            ).prepare(embed=False)
        )
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            content = str(entry.get("content") or "")
            if not content:
                continue
            docs.extend(
                docs_from_digest(
                    project_id=project_dir.name,
                    repo=repo,
                    entry_id=str(entry.get("id") or ""),
                    title=str(entry.get("title") or "memory entry"),
                    content=content,
                    max_section_chars=max_section_chars,
                )
            )
    return docs


def save_index(docs: list[RetrievalDoc], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "documents": [doc.prepare(embed=False).to_json() for doc in docs]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_index(path: Path) -> list[RetrievalDoc]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs = [RetrievalDoc.from_json(d).prepare(embed=False) for d in raw.get("documents", []) if isinstance(d, dict)]
    return docs


def document_frequencies(docs: list[RetrievalDoc]) -> dict[str, int]:
    df: dict[str, int] = {}
    for doc in docs:
        for token in doc.prepare(embed=False).tokens or set():
            df[token] = df.get(token, 0) + 1
    return df


def lexical_score(query_tokens: set[str], doc: RetrievalDoc, df: dict[str, int], doc_count: int) -> float:
    doc_tokens = doc.prepare(embed=False).tokens or set()
    if not query_tokens or not doc_tokens:
        return 0.0
    score = 0.0
    for token in query_tokens:
        if token not in doc_tokens:
            continue
        score += math.log((doc_count + 1) / (df.get(token, 0) + 1)) + 1.0
    return score / max(len(query_tokens), 1)


def search_docs(
    docs: list[RetrievalDoc],
    query: str,
    *,
    top_k: int = 8,
    scorer: str = "hybrid",
    df: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    q_tokens = tokenize(query)
    df = df or document_frequencies(docs)
    doc_count = len(docs)
    scored: list[tuple[RetrievalDoc, float]] = []
    for doc in docs:
        lexical = lexical_score(q_tokens, doc, df, doc_count)
        scored.append((doc, lexical))

    semantic_candidates: set[str] = set()
    if scorer in {"hybrid", "semantic"}:
        candidate_count = len(scored) if scorer == "semantic" else min(len(scored), max(min(top_k, 10) * 25, 100))
        semantic_candidates = {
            doc.chunk_id
            for doc, _lexical in sorted(scored, key=lambda item: item[1], reverse=True)[:candidate_count]
        }

    q_emb = _embed(query) if semantic_candidates else []
    rows: list[dict[str, Any]] = []
    for doc, lexical in scored:
        semantic = 0.0
        if doc.chunk_id in semantic_candidates:
            semantic = _cosine(q_emb, doc.embedding or _embed(doc.search_text()))
        if scorer == "semantic":
            score = semantic
        elif scorer == "lexical":
            score = lexical
        else:
            score = (0.25 * semantic) + (0.75 * lexical)
        rows.append(
            {
                "score": round(score, 4),
                "repo": doc.repo,
                "project_id": doc.project_id,
                "entry_id": doc.entry_id,
                "chunk_id": doc.chunk_id,
                "chunk_type": doc.chunk_type,
                "title": doc.title,
                "snippet": doc.content[:800],
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_k]
