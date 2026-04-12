"""
Self-improvement and training-data hooks (teacher review, JSONL logs).

Providers start with OpenAI; extend ``teacher_review`` with new backends later.
"""

from .export_sft import export_reviews_to_jsonl, record_to_sft_row, validate_row_for_first_party_commercial
from .provenance import (
    SOURCE_MINI_DEVIN_SYNTHETIC,
    SOURCE_MINI_DEVIN_TEACHER,
    default_synthetic_provenance,
    default_teacher_review_provenance,
    merge_export_row,
)
from .teacher_review import (
    LearningSettings,
    maybe_log_teacher_review,
    parse_teacher_json,
    serialize_conversation_excerpt,
)

__all__ = [
    "LearningSettings",
    "SOURCE_MINI_DEVIN_SYNTHETIC",
    "SOURCE_MINI_DEVIN_TEACHER",
    "default_synthetic_provenance",
    "default_teacher_review_provenance",
    "export_reviews_to_jsonl",
    "maybe_log_teacher_review",
    "merge_export_row",
    "parse_teacher_json",
    "record_to_sft_row",
    "serialize_conversation_excerpt",
    "validate_row_for_first_party_commercial",
]
