"""
Visual Regression Engine

Captures page screenshots, stores baselines, and performs pixel-diff
comparisons to detect unintended UI changes between versions.

No heavy dependency on Pillow — uses stdlib only for PNG manipulation,
but if Pillow is available it produces colored diff images.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import time
import zlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lightweight PNG pixel reader (no dependency)
# ---------------------------------------------------------------------------

def _read_png_pixels(data: bytes) -> Tuple[int, int, List[Tuple[int, int, int]]]:
    """
    Read a PNG and return (width, height, flat_rgb_list).
    Falls back to (0, 0, []) if the PNG cannot be parsed.
    Supports 8-bit RGBA and RGB only (the common case for screenshots).
    """
    if data[1:4] != b"PNG":
        return 0, 0, []

    pos = 8
    chunks: Dict[str, bytes] = {}
    raw_idat = b""
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8].decode("ascii", errors="replace")
        chunk_data = data[pos + 8:pos + 8 + length]
        if chunk_type == "IHDR":
            chunks["IHDR"] = chunk_data
        elif chunk_type == "IDAT":
            raw_idat += chunk_data
        pos += 12 + length
        if chunk_type == "IEND":
            break

    if "IHDR" not in chunks:
        return 0, 0, []

    width = struct.unpack(">I", chunks["IHDR"][0:4])[0]
    height = struct.unpack(">I", chunks["IHDR"][4:8])[0]
    bit_depth = chunks["IHDR"][8]
    color_type = chunks["IHDR"][9]

    if bit_depth != 8 or color_type not in (2, 6):   # 2=RGB, 6=RGBA
        return width, height, []

    try:
        raw = zlib.decompress(raw_idat)
    except zlib.error:
        return width, height, []

    channels = 3 if color_type == 2 else 4
    stride = width * channels + 1  # +1 for filter byte

    pixels: List[Tuple[int, int, int]] = []
    prev_row = bytes(width * channels)
    for y in range(height):
        row_start = y * stride
        filter_byte = raw[row_start]
        row = bytearray(raw[row_start + 1:row_start + 1 + width * channels])

        if filter_byte == 1:    # Sub
            for x in range(channels, len(row)):
                row[x] = (row[x] + row[x - channels]) & 0xFF
        elif filter_byte == 2:  # Up
            for x in range(len(row)):
                row[x] = (row[x] + prev_row[x]) & 0xFF
        elif filter_byte == 3:  # Average
            for x in range(len(row)):
                a = row[x - channels] if x >= channels else 0
                b = prev_row[x]
                row[x] = (row[x] + (a + b) // 2) & 0xFF
        elif filter_byte == 4:  # Paeth
            for x in range(len(row)):
                a = row[x - channels] if x >= channels else 0
                b = prev_row[x]
                c = prev_row[x - channels] if x >= channels else 0
                pa = abs(b - c)
                pb = abs(a - c)
                pc = abs(a + b - 2 * c)
                paeth = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                row[x] = (row[x] + paeth) & 0xFF

        prev_row = bytes(row)
        for x in range(width):
            base = x * channels
            pixels.append((row[base], row[base + 1], row[base + 2]))

    return width, height, pixels


def _pixel_diff_stats(
    a: List[Tuple[int, int, int]],
    b: List[Tuple[int, int, int]],
    threshold: int = 10,
) -> Tuple[int, float, List[int]]:
    """
    Compare two flat RGB pixel lists.

    Returns (changed_pixels, changed_percent, changed_indices).
    Pixels with per-channel delta > threshold are considered changed.
    """
    changed: List[int] = []
    count = min(len(a), len(b))
    for i in range(count):
        dr = abs(int(a[i][0]) - int(b[i][0]))
        dg = abs(int(a[i][1]) - int(b[i][1]))
        db = abs(int(a[i][2]) - int(b[i][2]))
        if dr > threshold or dg > threshold or db > threshold:
            changed.append(i)
    pct = (len(changed) / count * 100) if count else 0.0
    return len(changed), round(pct, 3), changed


def _try_pillow_diff(
    baseline_bytes: bytes,
    current_bytes: bytes,
    changed_indices: List[int],
    width: int,
    height: int,
) -> Optional[bytes]:
    """
    Generate a colored diff PNG using Pillow (if installed).
    Returns PNG bytes or None.
    """
    try:
        from PIL import Image, ImageDraw  # type: ignore
        import io

        base_img = Image.open(io.BytesIO(baseline_bytes)).convert("RGB")
        curr_img = Image.open(io.BytesIO(current_bytes)).convert("RGB")

        # Create a blended diff: darken unchanged, highlight diff in red
        diff_img = curr_img.copy().point(lambda p: p // 3)  # dim unchanged
        draw = ImageDraw.Draw(diff_img)
        for idx in changed_indices[:50_000]:  # cap for performance
            x = idx % width
            y = idx // width
            draw.point((x, y), fill=(255, 0, 0))

        buf = io.BytesIO()
        diff_img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ScreenshotRecord:
    """A single captured screenshot with metadata."""
    name: str                    # e.g. "homepage", "dashboard"
    url: str
    session_id: Optional[str]
    width: int
    height: int
    captured_at: str
    png_path: str                # relative to baseline_dir
    is_baseline: bool = False


@dataclass
class DiffResult:
    """Result of comparing two screenshots."""
    name: str
    baseline_path: str
    current_path: str
    diff_path: Optional[str]    # colored diff PNG (if generated)
    changed_pixels: int
    changed_percent: float
    width: int
    height: int
    passed: bool                 # True if below threshold
    threshold_percent: float
    compared_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VisualRegressionEngine:
    """
    Manages screenshot baselines and performs pixel-diff comparisons.

    Directory layout:
        baseline_dir/
            baselines/<name>.png          ← current baseline
            runs/<timestamp>/<name>.png   ← screenshots per test run
            diffs/<timestamp>/<name>_diff.png
            index.json                    ← metadata
    """

    DEFAULT_THRESHOLD = 0.5   # % changed pixels to fail

    def __init__(self, baseline_dir: str = "visual_baselines"):
        self.base = Path(baseline_dir)
        self._baselines_dir = self.base / "baselines"
        self._runs_dir = self.base / "runs"
        self._diffs_dir = self.base / "diffs"
        self._index_path = self.base / "index.json"

        for d in [self._baselines_dir, self._runs_dir, self._diffs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._index: Dict[str, Any] = self._load_index()

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, Any]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def save_screenshot(
        self,
        name: str,
        png_bytes: bytes,
        url: str = "",
        session_id: Optional[str] = None,
        set_as_baseline: bool = False,
    ) -> ScreenshotRecord:
        """
        Save a PNG screenshot.

        If set_as_baseline=True, stores it in baselines/ (replaces previous baseline).
        Otherwise stores in runs/<timestamp>/.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        w, h, _ = _read_png_pixels(png_bytes)

        if set_as_baseline:
            path = self._baselines_dir / f"{name}.png"
            path.write_bytes(png_bytes)
            rel = str(path.relative_to(self.base))
        else:
            run_dir = self._runs_dir / ts
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / f"{name}.png"
            path.write_bytes(png_bytes)
            rel = str(path.relative_to(self.base))

        record = ScreenshotRecord(
            name=name,
            url=url,
            session_id=session_id,
            width=w,
            height=h,
            captured_at=datetime.now(timezone.utc).isoformat(),
            png_path=rel,
            is_baseline=set_as_baseline,
        )

        # Update index
        if name not in self._index:
            self._index[name] = {"baseline": None, "runs": []}
        if set_as_baseline:
            self._index[name]["baseline"] = asdict(record)
        else:
            runs = self._index[name].setdefault("runs", [])
            runs.append(asdict(record))
            self._index[name]["runs"] = runs[-20:]   # keep last 20 runs

        self._save_index()
        return record

    def save_screenshot_b64(
        self,
        name: str,
        b64: str,
        url: str = "",
        session_id: Optional[str] = None,
        set_as_baseline: bool = False,
    ) -> ScreenshotRecord:
        return self.save_screenshot(
            name,
            base64.b64decode(b64),
            url=url,
            session_id=session_id,
            set_as_baseline=set_as_baseline,
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        name: str,
        current_png: bytes,
        threshold_percent: Optional[float] = None,
    ) -> DiffResult:
        """
        Compare current_png against the stored baseline for `name`.

        Returns a DiffResult with pass/fail and pixel stats.
        Raises FileNotFoundError if no baseline exists.
        """
        threshold = threshold_percent if threshold_percent is not None else self.DEFAULT_THRESHOLD

        baseline_path = self._baselines_dir / f"{name}.png"
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"No baseline for '{name}'. "
                f"Run set_baseline=True first to capture a baseline."
            )

        baseline_bytes = baseline_path.read_bytes()
        bw, bh, b_pixels = _read_png_pixels(baseline_bytes)
        cw, ch, c_pixels = _read_png_pixels(current_png)

        # Save current as a run screenshot
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = self._runs_dir / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        current_path = run_dir / f"{name}.png"
        current_path.write_bytes(current_png)

        diff_path: Optional[Path] = None
        changed_pixels = 0
        changed_percent = 0.0

        if b_pixels and c_pixels:
            changed_pixels, changed_percent, changed_indices = _pixel_diff_stats(
                b_pixels, c_pixels
            )
            # Generate diff image
            w = min(bw, cw)
            h = min(bh, ch)
            diff_png = _try_pillow_diff(
                baseline_bytes, current_png, changed_indices, w, h
            )
            if diff_png:
                diff_dir = self._diffs_dir / ts
                diff_dir.mkdir(parents=True, exist_ok=True)
                diff_path = diff_dir / f"{name}_diff.png"
                diff_path.write_bytes(diff_png)
        else:
            # Can't parse pixels — compare raw byte hashes
            if hashlib.md5(baseline_bytes).hexdigest() != hashlib.md5(current_png).hexdigest():
                changed_pixels = -1
                changed_percent = 100.0

        passed = changed_percent <= threshold
        result = DiffResult(
            name=name,
            baseline_path=str(baseline_path),
            current_path=str(current_path),
            diff_path=str(diff_path) if diff_path else None,
            changed_pixels=changed_pixels,
            changed_percent=changed_percent,
            width=bw,
            height=bh,
            passed=passed,
            threshold_percent=threshold,
        )

        # Update index
        if name not in self._index:
            self._index[name] = {"baseline": None, "runs": [], "diffs": []}
        diffs = self._index[name].setdefault("diffs", [])
        diffs.append(result.to_dict())
        self._index[name]["diffs"] = diffs[-20:]
        self._save_index()

        return result

    def compare_b64(
        self,
        name: str,
        current_b64: str,
        threshold_percent: Optional[float] = None,
    ) -> DiffResult:
        return self.compare(name, base64.b64decode(current_b64), threshold_percent)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_baselines(self) -> List[Dict[str, Any]]:
        return [
            {"name": name, **data.get("baseline", {})}
            for name, data in self._index.items()
            if data.get("baseline")
        ]

    def get_history(self, name: str) -> Dict[str, Any]:
        return self._index.get(name, {})

    def get_index(self) -> Dict[str, Any]:
        return self._index

    def delete_baseline(self, name: str) -> bool:
        p = self._baselines_dir / f"{name}.png"
        if p.exists():
            p.unlink()
        if name in self._index:
            self._index[name]["baseline"] = None
            self._save_index()
            return True
        return False

    def get_screenshot_b64(self, rel_path: str) -> Optional[str]:
        """Return base64 of a screenshot by relative path (from index)."""
        p = self.base / rel_path
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
        return None


# ---------------------------------------------------------------------------
# Singleton per working directory
# ---------------------------------------------------------------------------

_engines: Dict[str, VisualRegressionEngine] = {}


def get_engine(working_dir: str = ".") -> VisualRegressionEngine:
    baseline_dir = str(Path(working_dir) / "visual_baselines")
    if baseline_dir not in _engines:
        _engines[baseline_dir] = VisualRegressionEngine(baseline_dir)
    return _engines[baseline_dir]
