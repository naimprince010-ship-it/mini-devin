"""
Training-data provenance for commercial-safe workflows.

This is not legal advice: your counsel must approve dataset use for commercial models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# Known origins you control end-to-end (safest for commercial fine-tuning after ToS review).
SOURCE_MINI_DEVIN_TEACHER = "mini_devin_teacher_review"
SOURCE_MINI_DEVIN_SYNTHETIC = "mini_devin_synthetic_challenges"


@dataclass(frozen=True)
class FirstPartyProvenance:
    """Attach to JSONL rows exported for SFT/DPO."""

    source: str
    pipeline: str
    rights_basis: str  # e.g. "organization_internal_sessions" — set in your policy
    commercial_model_training_ok: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pipeline": self.pipeline,
            "rights_basis": self.rights_basis,
            "commercial_model_training_ok": self.commercial_model_training_ok,
            "notes": self.notes,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }


def default_teacher_review_provenance() -> dict[str, Any]:
    """Provenance block appended to each teacher JSONL row (env overridable)."""
    basis = __import__("os").environ.get(
        "TRAINING_RIGHTS_BASIS",
        "first_party_mini_devin_sessions_review_with_teacher_model",
    )
    ok = __import__("os").environ.get("TRAINING_COMMERCIAL_OK", "true").lower() == "true"
    return FirstPartyProvenance(
        source=SOURCE_MINI_DEVIN_TEACHER,
        pipeline="post_task_teacher_review",
        rights_basis=basis,
        commercial_model_training_ok=ok,
        notes=(
            "Derived from Plodder runs. Ensure end-user / customer agreements allow "
            "model training before using exports commercially or redistributing datasets."
        ),
    ).to_dict()


def default_synthetic_provenance() -> dict[str, Any]:
    """For scripts/generate_synthetic_challenges.py outputs."""
    return FirstPartyProvenance(
        source=SOURCE_MINI_DEVIN_SYNTHETIC,
        pipeline="generate_synthetic_challenges",
        rights_basis=__import__("os").environ.get(
            "TRAINING_RIGHTS_BASIS",
            "first_party_synthetic_from_org_repo_context",
        ),
        commercial_model_training_ok=__import__("os").environ.get(
            "TRAINING_COMMERCIAL_OK", "true"
        ).lower()
        == "true",
        notes="Synthetic prompts/tests generated via your API keys and repo context.",
    ).to_dict()


def merge_export_row(
    messages: list[dict[str, Any]],
    *,
    provenance: dict[str, Any],
    source_record_id: str | None = None,
) -> dict[str, Any]:
    """Single JSONL object for Llama-Factory / many chat trainers (messages + metadata)."""
    row: dict[str, Any] = {
        "messages": messages,
        "provenance": provenance,
    }
    if source_record_id:
        row["source_record_id"] = source_record_id
    return row
