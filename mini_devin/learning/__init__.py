"""
Self-improvement and training-data hooks (teacher review, JSONL logs).

Providers start with OpenAI; extend ``teacher_review`` with new backends later.
"""

from .teacher_review import (
    LearningSettings,
    maybe_log_teacher_review,
    parse_teacher_json,
    serialize_conversation_excerpt,
)

__all__ = [
    "LearningSettings",
    "maybe_log_teacher_review",
    "parse_teacher_json",
    "serialize_conversation_excerpt",
]
