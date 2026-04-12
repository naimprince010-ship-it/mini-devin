"""Tests for teacher review JSON parsing."""

import json

from mini_devin.learning.teacher_review import parse_teacher_json


def test_parse_plain_json():
    raw = '{"verdict": "pass", "issues": [], "mistake_analysis": "", "correct_approach": "ok", "suggested_followups": [], "confidence": 0.9}'
    out = parse_teacher_json(raw)
    assert out["verdict"] == "pass"
    assert out["confidence"] == 0.9


def test_parse_fenced_json():
    raw = """Here you go:
```json
{"verdict": "fail", "issues": ["x"], "mistake_analysis": "m", "correct_approach": "c", "suggested_followups": [], "confidence": 0.5}
```
"""
    out = parse_teacher_json(raw)
    assert out["verdict"] == "fail"
    assert "x" in out["issues"]


def test_parse_invalid_fallback():
    out = parse_teacher_json("not json at all")
    assert out["verdict"] == "issues"
    assert "teacher_json_parse_failed" in out["issues"]
