#!/usr/bin/env python3
"""
Export Mini-Devin teacher reviews JSONL → SFT chat JSONL (messages + provenance).

Commercial use: set TRAINING_RIGHTS_BASIS and TRAINING_COMMERCIAL_OK in .env; verify ToS.

Usage (from mini-devin repo root):
  python scripts/prepare_training_export.py \\
    --input data/training_logs/reviews.jsonl \\
    --output data/training_exports/sft_teacher_critique.jsonl

Optional filters:
  --verdicts issues,fail     (default: all rows that have export payload)
  --min-confidence 0.5
  --skip-pass                (exclude teacher verdict == pass)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mini_devin.learning.export_sft import export_reviews_to_jsonl

    p = argparse.ArgumentParser(description="Export reviews.jsonl to SFT JSONL")
    p.add_argument(
        "--input",
        type=str,
        default="data/training_logs/reviews.jsonl",
        help="Path to reviews.jsonl",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/training_exports/sft_teacher_critique.jsonl",
        help="Output JSONL path",
    )
    p.add_argument(
        "--verdicts",
        type=str,
        default="",
        help="Comma-separated verdicts to keep: pass,issues,fail (empty = all)",
    )
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument(
        "--skip-pass",
        action="store_true",
        help="Exclude rows where teacher verdict is pass",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    verdict_in = None
    if args.verdicts.strip():
        verdict_in = {v.strip().lower() for v in args.verdicts.split(",") if v.strip()}

    w, s = export_reviews_to_jsonl(
        in_path,
        out_path,
        mode="teacher_critique",
        verdict_in=verdict_in,
        min_teacher_confidence=args.min_confidence,
        require_verdict_not_pass=args.skip_pass,
    )
    print(f"Wrote {w} rows to {out_path} (skipped {s})")


if __name__ == "__main__":
    main()
