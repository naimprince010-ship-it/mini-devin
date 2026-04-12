"""
Plodder — polyglot autonomous engineer (architecture shell).

Core pieces live in ``plodder.core`` and ``plodder.sandbox``.
"""

from plodder.core.universal_prompt_engine import (
    PolyglotSystemPrompt,
    PseudoLogicPlan,
    UniversalPromptEngine,
)
from plodder.rag.doc_ingestion import DocumentationStore, unfamiliar_stack_query

__all__ = [
    "PolyglotSystemPrompt",
    "PseudoLogicPlan",
    "UniversalPromptEngine",
    "DocumentationStore",
    "unfamiliar_stack_query",
]
