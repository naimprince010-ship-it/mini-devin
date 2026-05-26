"""Tests for static workspace preview helpers."""

from pathlib import Path

import pytest
from fastapi import HTTPException

from mini_devin.api.app import _workspace_static_preview_target


def test_static_preview_target_serves_index(tmp_path: Path):
    index = tmp_path / "index.html"
    index.write_text("<h1>Hello</h1>", encoding="utf-8")

    assert _workspace_static_preview_target(str(tmp_path), "") == str(index.resolve())


def test_static_preview_target_blocks_traversal(tmp_path: Path):
    with pytest.raises(HTTPException) as exc:
        _workspace_static_preview_target(str(tmp_path), "../secret.txt")

    assert exc.value.status_code == 400


def test_static_preview_target_requires_file(tmp_path: Path):
    with pytest.raises(HTTPException) as exc:
        _workspace_static_preview_target(str(tmp_path), "missing.html")

    assert exc.value.status_code == 404
