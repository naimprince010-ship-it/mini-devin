"""
Autonomous browser module (Playwright) for Plodder — OpenHands-style UI inspection.

Sync API so callers can wrap with ``asyncio.to_thread`` from async orchestration.
"""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote_plus

from plodder.tools.browser_heuristics import map_llm_click_to_pixels

# Playwright is an optional heavy dep at import sites; fail clearly at start().
try:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None  # type: ignore[misc, assignment]
    Playwright = Browser = BrowserContext = Page = Any  # type: ignore[misc, assignment]


ScrollDirection = Literal["up", "down", "top", "bottom"]


@dataclass
class BrowserActionResult:
    ok: bool
    action: str
    detail: str = ""
    url: str = ""
    title: str = ""


class BrowserManager:
    """
    Playwright-backed browser for navigation, clicks, typing, scroll, screenshots.

    Use as context manager or call ``start()`` / ``close()`` explicitly.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        viewport: tuple[int, int] = (1280, 720),
        default_timeout_ms: int = 30_000,
    ) -> None:
        self.headless = headless
        self.viewport = {"width": viewport[0], "height": viewport[1]}
        self.default_timeout_ms = default_timeout_ms
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def __enter__(self) -> BrowserManager:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserManager not started")
        return self._page

    def start(self) -> None:
        if sync_playwright is None:
            raise ImportError("playwright is not installed; add playwright to the environment.")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            viewport=self.viewport,
            ignore_https_errors=True,
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(self.default_timeout_ms)

    def close(self) -> None:
        for attr in ("_page", "_context", "_browser"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def navigate(self, url: str, *, wait_until: str = "domcontentloaded") -> dict[str, Any]:
        """Go to ``url`` (http(s) or file://)."""
        self.page.goto(url, wait_until=wait_until)
        return self._state("navigate", ok=True)

    def maps(self, url_or_query: str) -> dict[str, Any]:
        """
        Open Google Maps from a lat,lon pair or a free-text search.

        Examples: ``"48.8584, 2.2945"``, ``"Brooklyn Bridge"``.
        """
        raw = url_or_query.strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            return self.navigate(raw)
        coord = re.match(
            r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$",
            raw,
        )
        if coord:
            lat, lon = coord.group(1), coord.group(2)
            target = f"https://www.google.com/maps?q={lat},{lon}"
        else:
            target = f"https://www.google.com/maps/search/{quote_plus(raw)}"
        return self.navigate(target)

    def click(self, selector: str, *, timeout_ms: int | None = None) -> dict[str, Any]:
        """Click the first element matching ``selector`` (CSS or text engine prefix)."""
        timeout = timeout_ms or self.default_timeout_ms
        self.page.click(selector, timeout=timeout)
        return self._state("click", ok=True, detail=selector)

    def type(self, selector: str, text: str, *, clear: bool = True) -> dict[str, Any]:
        """Type/fill into ``selector``."""
        loc = self.page.locator(selector).first
        loc.wait_for(state="visible", timeout=self.default_timeout_ms)
        if clear:
            loc.fill("")
        loc.fill(text)
        return self._state("type", ok=True, detail=f"{selector!r} ({len(text)} chars)")

    def scroll(self, direction: ScrollDirection, *, pixels: int = 600) -> dict[str, Any]:
        """Scroll the main viewport (wheel)."""
        d = direction.lower()  # type: ignore[assignment]
        if d == "down":
            self.page.mouse.wheel(0, pixels)
        elif d == "up":
            self.page.mouse.wheel(0, -pixels)
        elif d == "top":
            self.page.evaluate("window.scrollTo(0, 0)")
        elif d == "bottom":
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        else:
            return self._state("scroll", ok=False, detail=f"unknown direction: {direction}")
        return self._state("scroll", ok=True, detail=f"{direction} ~{pixels}px")

    def click_xy(self, x: int, y: int) -> dict[str, Any]:
        """Pixel click at (x, y) in viewport coordinates."""
        self.page.mouse.click(x, y)
        return self._state("click_xy", ok=True, detail=f"{x},{y}")

    def click_from_llm_coords(self, x: float, y: float) -> dict[str, Any]:
        """Resolve LLM (normalized or 0–1000) coordinates and click."""
        vp = self.page.viewport_size or self.viewport
        w, h = int(vp["width"]), int(vp["height"])
        px, py = map_llm_click_to_pixels(x, y, w, h)
        return self.click_xy(px, py) | {"resolved": [px, py], "input": [x, y]}

    def take_screenshot(self, *, full_page: bool = False) -> bytes:
        """PNG screenshot bytes of the current page."""
        return self.page.screenshot(type="png", full_page=full_page)

    def take_screenshot_base64(self, *, full_page: bool = False) -> str:
        """Base64-encoded PNG (no data URL prefix)."""
        return base64.standard_b64encode(self.take_screenshot(full_page=full_page)).decode("ascii")

    def snapshot_for_llm(self, *, full_page: bool = False) -> dict[str, Any]:
        """Bundle url/title + base64 image for vision models."""
        png = self.take_screenshot(full_page=full_page)
        st = self._state("screenshot", ok=True)
        st["image_base64"] = base64.standard_b64encode(png).decode("ascii")
        st["image_mime"] = "image/png"
        return st

    def _state(self, action: str, *, ok: bool, detail: str = "") -> dict[str, Any]:
        try:
            url = self.page.url
            title = self.page.title()
        except Exception:
            url, title = "", ""
        return {
            "ok": ok,
            "action": action,
            "detail": detail,
            "url": url,
            "title": title,
        }


def capture_url_screenshot_with_console(
    url: str,
    *,
    headless: bool = True,
    full_page: bool = False,
    wait_until: str = "domcontentloaded",
    wait_after_load_ms: int = 900,
) -> dict[str, Any]:
    """
    One-shot navigation + screenshot + browser **console** + uncaught page errors.

    Used when ``npm run dev`` / Vite fails and the agent needs runtime signals like OpenHands.
    """
    console_lines: list[str] = []
    page_errors: list[str] = []
    if sync_playwright is None:
        return {"ok": False, "error": "playwright is not installed", "url": url}
    try:
        wait_s = max(0.25, min(wait_after_load_ms / 1000.0, 12.0))
        with BrowserManager(headless=headless) as bm:
            page = bm.page

            def on_console(msg: Any) -> None:
                try:
                    console_lines.append(f"{msg.type}: {msg.text}")
                except Exception:
                    pass

            def on_page_error(err: Any) -> None:
                try:
                    page_errors.append(str(err))
                except Exception:
                    pass

            page.on("console", on_console)
            page.on("pageerror", on_page_error)
            bm.navigate(url, wait_until=wait_until)
            time.sleep(wait_s)
            b64 = bm.take_screenshot_base64(full_page=full_page)
            cap = 48_000
            img = b64 or ""
            return {
                "ok": True,
                "url": url,
                "image_base64": img[:cap],
                "image_truncated": len(img) > cap,
                "console_messages": console_lines[:160],
                "console_truncated": len(console_lines) > 160,
                "page_errors": page_errors[:48],
            }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(e),
            "url": url,
            "console_messages": console_lines[:80],
            "page_errors": page_errors[:24],
        }


def capture_url_screenshot_base64(
    url: str,
    *,
    headless: bool = True,
    full_page: bool = False,
    wait_until: str = "domcontentloaded",
) -> str | None:
    """
    One-shot: open ``url``, screenshot, close browser. For ``asyncio.to_thread``.

    Returns base64 PNG or ``None`` on failure.
    """
    try:
        with BrowserManager(headless=headless) as bm:
            bm.navigate(url, wait_until=wait_until)
            time.sleep(0.35)
            return bm.take_screenshot_base64(full_page=full_page)
    except Exception:
        return None
