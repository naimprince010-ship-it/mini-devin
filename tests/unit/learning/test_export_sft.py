"""Tests for SFT export from teacher review records."""

from pathlib import Path

from mini_devin.learning.export_sft import export_reviews_to_jsonl, record_to_sft_row
from mini_devin.learning.provenance import merge_export_row


def test_record_to_sft_row():
    record = {
        "id": "abc",
        "fine_tune_exports": {
            "sft_teacher_critique_messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "{}"},
            ],
        },
        "teacher_review": {"verdict": "issues", "confidence": 0.9},
    }
    row = record_to_sft_row(record)
    assert row is not None
    assert row["source_record_id"] == "abc"
    assert len(row["messages"]) == 2
    assert "provenance" in row


def test_export_reviews_to_jsonl(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(
        '{"id":"1","teacher_review":{"verdict":"fail","confidence":1},'
        '"fine_tune_exports":{"sft_teacher_critique_messages":['
        '{"role":"user","content":"t"},{"role":"assistant","content":"{}"}]}}\n',
        encoding="utf-8",
    )
    w, s = export_reviews_to_jsonl(inp, out)
    assert w == 1 and s == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_validate_merge_export_row():
    row = merge_export_row(
        [{"role": "user", "content": "x"}],
        provenance={"source": "mini_devin_teacher_review", "commercial_model_training_ok": True},
        source_record_id="z",
    )
    assert row["source_record_id"] == "z"
