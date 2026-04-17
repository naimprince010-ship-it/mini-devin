"""
OpenHands-style **browser agent** for Plodder: persistent Playwright page, interactive map,
console + failed network capture, and human-like click/type after ``playwright_observe`` grounding.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from plodder.tools.browser_manager import BrowserManager, sync_playwright

ScrollDirection = Literal["up", "down", "top", "bottom"]


def _normalize_element_id(raw: str) -> str | None:
    s = (raw or "").strip()
    if not s:
        return None
    if re.match(r"^p\d+$", s, re.I):
        return s.lower()
    if s.isdigit():
        return f"p{int(s)}"
    return None


def _build_inject_script(max_elements: int) -> str:
    return f"""
(() => {{
  const sel = [
    'button','a[href]','input:not([type=hidden])','textarea','select',
    '[role="button"]','[role="link"]','[role="textbox"]','[role="checkbox"]','[role="radio"]','[role="menuitem"]',
    '[tabindex]:not([tabindex="-1"])'
  ].join(',');
  const nodes = Array.from(document.querySelectorAll(sel));
  const out = [];
  let n = 0;
  for (const el of nodes) {{
    let style;
    try {{ style = window.getComputedStyle(el); }} catch (e) {{ continue; }}
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    n += 1;
    const id = 'p' + n;
    try {{ el.setAttribute('data-plodder-id', id); }} catch (e) {{}}
    const txt = (el.innerText || '').trim().slice(0, 120);
    out.push({{
      id, tag: el.tagName.toLowerCase(),
      role: el.getAttribute('role') || '',
      aria_label: el.getAttribute('aria-label') || '',
      name: el.getAttribute('name') || '',
      type: el.getAttribute('type') || '',
      placeholder: (el.getAttribute('placeholder') || '').slice(0, 80),
      href: el.tagName === 'A' ? (el.href || '').slice(0, 200) : '',
      text: txt,
      x: Math.round(r.x), y: Math.round(r.y),
      width: Math.round(r.width), height: Math.round(r.height),
      disabled: !!el.disabled,
    }});
    if (n >= {max_elements}) break;
  }}
  return out;
}})()
"""


@dataclass
class PlodderPlaywrightSession:
    """
    One Chromium page for a Plodder unified-driver run: listeners, interactive map, observe bundle.
    """

    headless: bool = True
    max_image_b64: int = 48_000
    _bm: BrowserManager | None = field(default=None, repr=False)
    _last_elements: list[dict[str, Any]] = field(default_factory=list)
    _console: list[str] = field(default_factory=list)
    _page_errors: list[str] = field(default_factory=list)
    _network_failures: list[str] = field(default_factory=list)

    def _ensure(self) -> BrowserManager:
        if sync_playwright is None:
            raise ImportError("playwright is not installed; pip install playwright && playwright install chromium")
        if self._bm is None:
            self._bm = BrowserManager(headless=self.headless)
            self._bm.start()
            self._attach_listeners()
        return self._bm

    def _attach_listeners(self) -> None:
        assert self._bm is not None
        page = self._bm.page

        def on_console(msg: Any) -> None:
            try:
                self._console.append(f"{msg.type}: {msg.text}")
            except Exception:
                pass

        def on_page_error(err: Any) -> None:
            try:
                self._page_errors.append(str(err))
            except Exception:
                pass

        def on_response(resp: Any) -> None:
            try:
                st = resp.status
                if st >= 400:
                    req = resp.request
                    meth = getattr(req, "method", "?") or "?"
                    u = str(resp.url or "")[:400]
                    self._network_failures.append(f"{st} {meth} {u}")
            except Exception:
                pass

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("response", on_response)

    def clear_transient_diagnostics(self) -> None:
        self._console.clear()
        self._page_errors.clear()
        self._network_failures.clear()

    def close(self) -> None:
        if self._bm is not None:
            try:
                self._bm.close()
            except Exception:
                pass
            self._bm = None
        self._last_elements = []

    def navigate(self, url: str, *, wait_until: str = "domcontentloaded") -> dict[str, Any]:
        bm = self._ensure()
        bm.navigate(url, wait_until=wait_until)
        return {"ok": True, "url": bm.page.url, "title": bm.page.title()}

    def _accessibility_json(self, page: Any, max_chars: int) -> str:
        try:
            snap = page.accessibility.snapshot(interesting_only=True)
            raw = json.dumps(snap, default=str, ensure_ascii=False)
            if len(raw) > max_chars:
                return raw[:max_chars] + "\n...(accessibility tree truncated)...\n"
            return raw
        except Exception as e:
            return json.dumps({"error": str(e)})

    def observe_bundle(
        self,
        *,
        url: str | None,
        wait_ms: int,
        capture_console: bool,
        include_accessibility: bool,
        max_elements: int,
        full_page: bool,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> dict[str, Any]:
        """
        Navigate (if ``url``), wait, inject ``data-plodder-id`` map, screenshot, a11y, diagnostics.
        """
        bm = self._ensure()
        page = bm.page

        vw_out: int | None = None
        vh_out: int | None = None
        if viewport_width is not None and viewport_height is not None:
            try:
                vw_out = max(320, min(int(viewport_width), 3840))
                vh_out = max(240, min(int(viewport_height), 2160))
                page.set_viewport_size({"width": vw_out, "height": vh_out})
            except Exception:
                vw_out, vh_out = None, None

        nav = (url or "").strip()
        if not nav:
            try:
                cur = (page.url or "").strip()
            except Exception:
                cur = ""
            if cur in ("about:blank", "") or cur.startswith("chrome://"):
                nav = "http://127.0.0.1:5173"
        if nav:
            self.clear_transient_diagnostics()
            bm.navigate(nav, wait_until="domcontentloaded")

        wait_s = max(0.08, min(wait_ms / 1000.0, 12.0))
        time.sleep(wait_s)

        script = _build_inject_script(max(10, min(max_elements, 200)))
        try:
            elements_raw = page.evaluate(script)
        except Exception as e:
            elements_raw = []
            inj_err = str(e)
        else:
            inj_err = ""

        elements: list[dict[str, Any]] = elements_raw if isinstance(elements_raw, list) else []
        self._last_elements = list(elements)

        a11y = ""
        if include_accessibility:
            a11y = self._accessibility_json(page, max_chars=28_000)

        png = bm.take_screenshot(full_page=full_page)
        b64 = base64.standard_b64encode(png).decode("ascii")
        cap = self.max_image_b64
        img = b64[:cap]

        out: dict[str, Any] = {
            "ok": True,
            "url": page.url,
            "title": page.title(),
            "image_base64": img,
            "image_truncated": len(b64) > cap,
            "interactive_elements": elements[: max_elements + 5],
            "interactive_element_count": len(elements),
            "inject_error": inj_err or None,
        }
        if vw_out is not None and vh_out is not None:
            out["viewport_width"] = vw_out
            out["viewport_height"] = vh_out
        if include_accessibility:
            out["accessibility_tree_json"] = a11y
        if capture_console:
            out["console_messages"] = self._console[-200:]
            out["console_truncated"] = len(self._console) > 200
            out["page_errors"] = self._page_errors[-48:]
            out["network_failures"] = self._network_failures[-80:]
            out["network_failure_count"] = len(self._network_failures)
        return out

    def _post_action_verify(
        self,
        *,
        wait_ms: int,
        capture_console: bool,
        include_accessibility: bool,
        max_elements: int,
        full_page: bool,
        include_screenshot: bool,
    ) -> dict[str, Any]:
        """Fresh observe after click/type (optional screenshot to save tokens)."""
        thin = not include_screenshot
        bundle = self.observe_bundle(
            url=None,
            wait_ms=wait_ms,
            capture_console=capture_console,
            include_accessibility=include_accessibility,
            max_elements=max_elements,
            full_page=full_page,
            viewport_width=None,
            viewport_height=None,
        )
        if thin and bundle.get("ok"):
            bundle.pop("image_base64", None)
            bundle.pop("image_truncated", None)
            bundle["screenshot_omitted"] = True
        return bundle

    def click_element(
        self,
        element_id: str,
        *,
        post_wait_ms: int,
        verify: bool,
        verify_include_screenshot: bool,
        capture_console: bool,
        include_accessibility: bool,
        max_elements: int,
        full_page: bool,
    ) -> dict[str, Any]:
        bid = _normalize_element_id(element_id)
        if not bid:
            return {"ok": False, "error": f"Invalid element_id {element_id!r}; use id from interactive_elements (e.g. p12 or 12)."}
        bm = self._ensure()
        page = bm.page
        loc = page.locator(f'[data-plodder-id="{bid}"]')
        try:
            n = loc.count()
        except Exception as e:
            return {"ok": False, "error": f"locator error: {e}"}
        if n == 0:
            return {
                "ok": False,
                "error": f"No element with data-plodder-id={bid!r}. Call playwright_observe first to refresh the map.",
            }
        first = loc.first
        try:
            first.scroll_into_view_if_needed(timeout=10_000)
            first.click(timeout=15_000)
        except Exception as e:
            return {"ok": False, "error": str(e), "element_id": bid}

        result: dict[str, Any] = {
            "ok": True,
            "action": "browser_click",
            "element_id": bid,
            "url": page.url,
            "title": page.title(),
        }
        if verify:
            result["post_action_observe"] = self._post_action_verify(
                wait_ms=post_wait_ms,
                capture_console=capture_console,
                include_accessibility=include_accessibility,
                max_elements=max_elements,
                full_page=full_page,
                include_screenshot=verify_include_screenshot,
            )
        return result

    def type_element(
        self,
        element_id: str,
        text: str,
        *,
        submit: bool,
        post_wait_ms: int,
        verify: bool,
        verify_include_screenshot: bool,
        capture_console: bool,
        include_accessibility: bool,
        max_elements: int,
        full_page: bool,
    ) -> dict[str, Any]:
        bid = _normalize_element_id(element_id)
        if not bid:
            return {"ok": False, "error": f"Invalid element_id {element_id!r}"}
        bm = self._ensure()
        page = bm.page
        loc = page.locator(f'[data-plodder-id="{bid}"]')
        if loc.count() == 0:
            return {
                "ok": False,
                "error": f"No element {bid!r}; run playwright_observe first.",
            }
        first = loc.first
        try:
            first.scroll_into_view_if_needed(timeout=10_000)
            first.fill(text, timeout=15_000)
            if submit:
                page.keyboard.press("Enter")
        except Exception as e:
            return {"ok": False, "error": str(e), "element_id": bid}

        result: dict[str, Any] = {
            "ok": True,
            "action": "browser_type",
            "element_id": bid,
            "chars": len(text),
            "url": page.url,
            "title": page.title(),
        }
        if verify:
            result["post_action_observe"] = self._post_action_verify(
                wait_ms=post_wait_ms,
                capture_console=capture_console,
                include_accessibility=include_accessibility,
                max_elements=max_elements,
                full_page=full_page,
                include_screenshot=verify_include_screenshot,
            )
        return result

    def scroll_viewport(self, direction: ScrollDirection, *, pixels: int) -> dict[str, Any]:
        bm = self._ensure()
        return bm.scroll(direction, pixels=pixels)
