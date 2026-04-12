"""
Export teacher `reviews.jsonl` into chat JSONL for SFT (Llama-Factory, Unsloth, etc.).

Filters help keep commercial-first-party slices; external OS datasets should add their own
provenance before merging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .provenance import SOURCE_MINI_DEVIN_TEACHER, default_teacher_review_provenance, merge_export_row


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def record_to_sft_row(
    record: dict[str, Any],
    *,
    mode: str = "teacher_critique",
    provenance_override: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Convert one reviews.jsonl record to one training row.

    mode:
      - teacher_critique: uses fine_tune_exports.sft_teacher_critique_messages
    """
    prov = provenance_override or record.get("provenance") or default_teacher_review_provenance()
    # Re-stamp with export metadata
    if isinstance(prov, dict):
        prov = {**prov, "export_mode": mode}

    if mode == "teacher_critique":
        ft = record.get("fine_tune_exports") or {}
        msgs = ft.get("sft_teacher_critique_messages")
        if not msgs or not isinstance(msgs, list):
            return None
        return merge_export_row(
            messages=msgs,
            provenance=prov,
            source_record_id=record.get("id"),
        )
    return None


def export_reviews_to_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    mode: str = "teacher_critique",
    verdict_in: set[str] | None = None,
    min_teacher_confidence: float = 0.0,
    require_verdict_not_pass: bool = False,
) -> tuple[int, int]:
    """
    Read reviews JSONL, write SFT JSONL.

    Returns (written_count, skipped_count).
    """
    written = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for record in _iter_jsonl(input_path):
            tr = record.get("teacher_review")
            if not isinstance(tr, dict):
                skipped += 1
                continue
            verdict = str(tr.get("verdict", "")).lower()
            if verdict_in is not None and verdict not in verdict_in:
                skipped += 1
                continue
            if require_verdict_not_pass and verdict == "pass":
                skipped += 1
                continue
            try:
                conf = float(tr.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < min_teacher_confidence:
                skipped += 1
                continue

            row = record_to_sft_row(record, mode=mode)
            if row is None:
                skipped += 1
                continue

            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            written += 1

    return written, skipped


def validate_row_for_first_party_commercial(row: dict[str, Any]) -> bool:
    """Light check before merging external data: require explicit provenance.source."""
    prov = row.get("provenance")
    if not isinstance(prov, dict):
        return False
    src = prov.get("source")
    if src == SOURCE_MINI_DEVIN_TEACHER:
        return bool(prov.get("commercial_model_training_ok", False))
    # External rows must set commercial_model_training_ok and license_spdx or license_url
    if not prov.get("commercial_model_training_ok"):
        return False
    return bool(prov.get("license_spdx") or prov.get("license_url"))
