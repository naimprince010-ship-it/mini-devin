"""
Optional "lazy load" UX when a language has no rows in ``DocumentationStore``.

Wire this in your session loop: if ``should_offer_lazy_load(...)`` is true, surface
``lazy_load_user_message(...)`` to the human; on confirmation, call your LLM to draft a
``.md`` with front matter, write it under ``docs/languages/``, then ``index_file``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plodder.rag import doc_ingestion as _ingest

if TYPE_CHECKING:
    from plodder.rag.doc_ingestion import DocumentationStore


def should_offer_lazy_load(store: DocumentationStore, language_key: str) -> bool:
    """True when no chunks exist for ``language_key`` (slug)."""
    lk = _ingest._slug_language_key(language_key)
    return not store.has_language_key(lk)


def lazy_load_user_message(*, language_display: str, language_key: str) -> str:
    """Human-facing prompt to ask for permission to generate + index a cheat-sheet."""
    return (
        f"Boss — I do not have indexed docs for **{language_display}** (`{language_key}`) in "
        "Plodder LanceDB yet.\n\n"
        "Reply **Yes** if you want me to draft a short cheat-sheet (Markdown), save it under "
        "`docs/languages/`, and re-index the vector store."
    )


def cheatsheet_stub_markdown(*, language_display: str, language_key: str) -> str:
    """
    Placeholder body until you plug in an LLM call.

    Returns minimal Markdown + front matter suitable for ``DocumentationStore.index_file``.
    """
    return (
        "---\n"
        f"language: {language_display}\n"
        f"language_key: {language_key}\n"
        "---\n\n"
        f"# {language_display} — quick stub\n\n"
        "_Replace this file with a real LLM-generated cheat-sheet._\n"
    )
