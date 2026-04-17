"""
Advanced Playwright browser tools with persistent session state.

These tools expose human-like browser interactions to the agent:
- ``browser_navigate``
- ``browser_click``
- ``browser_type``
- ``browser_scroll``
- ``browser_screenshot``

The implementation keeps a single Chromium context alive across calls so login
state, cookies, and in-page navigation persist for the full agent session.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .base import BaseBrowserTool, ToolResult


def _safe_json(value: Any, *, max_chars: int = 12_000) -> str:
    try:
        blob = json.dumps(value, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001
        blob = json.dumps({"serialization_error": str(exc)}, ensure_ascii=False)
    if len(blob) > max_chars:
        return blob[: max_chars - 24] + "\n...(truncated)..."
    return blob


def _interactive_map_script(max_elements: int) -> str:
    return f"""
(() => {{
  const selectors = [
    'button',
    'a[href]',
    'input:not([type="hidden"])',
    'textarea',
    'select',
    '[role="button"]',
    '[role="link"]',
    '[role="textbox"]',
    '[role="combobox"]',
    '[role="checkbox"]',
    '[role="radio"]',
    '[role="menuitem"]',
    '[tabindex]:not([tabindex="-1"])'
  ].join(',');
  const seen = new Set();
  const nodes = Array.from(document.querySelectorAll(selectors));
  const out = [];
  let counter = 0;
  for (const el of nodes) {{
    if (seen.has(el)) continue;
    seen.add(el);
    let style;
    try {{ style = window.getComputedStyle(el); }} catch (e) {{ continue; }}
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
    const rect = el.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) continue;
    counter += 1;
    const id = `b${{counter}}`;
    try {{ el.setAttribute('data-mini-devin-target', id); }} catch (e) {{}}
    out.push({{
      id,
      tag: (el.tagName || '').toLowerCase(),
      role: el.getAttribute('role') || '',
      text: (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 140),
      aria_label: (el.getAttribute('aria-label') || '').slice(0, 140),
      placeholder: (el.getAttribute('placeholder') || '').slice(0, 140),
      name: (el.getAttribute('name') || '').slice(0, 140),
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      disabled: !!el.disabled
    }});
    if (counter >= {max_elements}) break;
  }}
  return out;
}})()
"""


@dataclass
class BrowserObservation:
    """Serializable browser state returned after every action."""

    action: str
    url: str
    title: str
    screenshot_base64: str | None = None
    accessibility_tree_json: str | None = None
    interactive_elements: list[dict[str, Any]] = field(default_factory=list)
    console_messages: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    network_failures: list[str] = field(default_factory=list)
    detail: str = ""
    action_time_ms: int = 0
    viewport_width: int | None = None
    viewport_height: int | None = None


class AdvancedBrowserSession:
    """Persistent Playwright Chromium session shared across browser tools."""

    def __init__(
        self,
        *,
        headless: bool = True,
        window_width: int = 1440,
        window_height: int = 900,
        timeout_ms: int = 30_000,
        max_interactive_elements: int = 80,
        on_browser_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.headless = headless
        self.window_width = window_width
        self.window_height = window_height
        self.timeout_ms = timeout_ms
        self.max_interactive_elements = max_interactive_elements
        self.on_browser_event = on_browser_event

        self._playwright = None
        self._playwright_manager = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False
        self._console: list[str] = []
        self._page_errors: list[str] = []
        self._network_failures: list[str] = []
        self._last_mouse_position: tuple[float, float] = (
            self.window_width / 2,
            self.window_height / 2,
        )

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required. Install with `pip install playwright` and "
                "`playwright install chromium`."
            ) from exc

        self._playwright_manager = async_playwright()
        self._playwright = await self._playwright_manager.start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": self.window_width, "height": self.window_height},
            ignore_https_errors=True,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.timeout_ms)
        self._attach_debug_listeners()
        self._initialized = True

    def _attach_debug_listeners(self) -> None:
        if not self._page:
            return
        page = self._page

        def on_console(msg: Any) -> None:
            try:
                line = f"[{msg.type}] {msg.text}"[:2000]
                self._console.append(line)
                self._console[:] = self._console[-120:]
            except Exception:
                pass

        def on_page_error(err: Any) -> None:
            try:
                self._page_errors.append(str(err)[:2000])
                self._page_errors[:] = self._page_errors[-40:]
            except Exception:
                pass

        def on_response(resp: Any) -> None:
            try:
                if int(resp.status) < 400:
                    return
                req = resp.request
                line = f"{req.method} {resp.status} {resp.url}"[:2000]
                self._network_failures.append(line)
                self._network_failures[:] = self._network_failures[-60:]
            except Exception:
                pass

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("response", on_response)

    async def _emit_browser_event(self, event_type: str, observation: BrowserObservation) -> None:
        if not self.on_browser_event:
            return
        payload = {
            "event_type": event_type,
            "url": observation.url,
            "query": observation.detail or None,
            "screenshot_base64": observation.screenshot_base64,
        }
        try:
            result = self.on_browser_event(payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    async def _take_screenshot(self, *, full_page: bool = False) -> str | None:
        if not self._page:
            return None
        png = await self._page.screenshot(type="png", full_page=full_page)
        return base64.b64encode(png).decode("ascii")

    async def _accessibility_tree_json(self) -> str:
        if not self._page:
            return "{}"
        try:
            snapshot = await self._page.accessibility.snapshot(interesting_only=True)
        except Exception as exc:
            return _safe_json({"error": str(exc)}, max_chars=4000)
        return _safe_json(snapshot, max_chars=18_000)

    async def _interactive_elements(self) -> list[dict[str, Any]]:
        if not self._page:
            return []
        script = _interactive_map_script(self.max_interactive_elements)
        try:
            data = await self._page.evaluate(script)
        except Exception:
            return []
        return data if isinstance(data, list) else []

    async def _capture_observation(
        self,
        *,
        action: str,
        detail: str = "",
        include_accessibility: bool = True,
        full_page: bool = False,
    ) -> BrowserObservation:
        assert self._page is not None
        screenshot = await self._take_screenshot(full_page=full_page)
        accessibility_tree = await self._accessibility_tree_json() if include_accessibility else None
        interactive_elements = await self._interactive_elements()
        viewport = self._page.viewport_size or {
            "width": self.window_width,
            "height": self.window_height,
        }
        observation = BrowserObservation(
            action=action,
            url=self._page.url,
            title=await self._page.title(),
            screenshot_base64=screenshot,
            accessibility_tree_json=accessibility_tree,
            interactive_elements=interactive_elements,
            console_messages=list(self._console[-40:]),
            page_errors=list(self._page_errors[-20:]),
            network_failures=list(self._network_failures[-20:]),
            detail=detail,
            viewport_width=int(viewport.get("width", self.window_width)),
            viewport_height=int(viewport.get("height", self.window_height)),
        )
        await self._emit_browser_event(action, observation)
        return observation

    async def _sleep_ms(self, ms: int) -> None:
        await asyncio.sleep(max(0, ms) / 1000.0)

    async def _move_mouse_human_like(self, x: float, y: float) -> None:
        assert self._page is not None
        start_x, start_y = self._last_mouse_position
        distance = math.hypot(x - start_x, y - start_y)
        steps = max(8, min(28, int(distance / 35) + 8))
        for step in range(1, steps + 1):
            t = step / steps
            eased = t * t * (3 - 2 * t)
            jitter_x = math.sin(step * 0.8) * min(2.5, distance / 120)
            jitter_y = math.cos(step * 0.6) * min(2.0, distance / 140)
            nx = start_x + (x - start_x) * eased + jitter_x
            ny = start_y + (y - start_y) * eased + jitter_y
            await self._page.mouse.move(nx, ny, steps=1)
            await asyncio.sleep(0.008 + ((step % 4) * 0.004))
        self._last_mouse_position = (x, y)

    async def _settle_after_action(self, base_ms: int = 250) -> None:
        assert self._page is not None
        try:
            await self._page.wait_for_load_state("domcontentloaded", timeout=base_ms)
        except Exception:
            pass
        await self._sleep_ms(base_ms)

    async def navigate(self, url: str, *, wait_until: str = "domcontentloaded") -> BrowserObservation:
        await self._ensure_initialized()
        assert self._page is not None
        start = datetime.now(timezone.utc)
        self._console.clear()
        self._page_errors.clear()
        self._network_failures.clear()
        await self._page.goto(url, wait_until=wait_until)
        await self._settle_after_action(350)
        observation = await self._capture_observation(
            action="browser_navigate",
            detail=url,
            include_accessibility=True,
            full_page=False,
        )
        observation.action_time_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return observation

    async def click(
        self,
        *,
        selector: str | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> BrowserObservation:
        await self._ensure_initialized()
        assert self._page is not None
        start = datetime.now(timezone.utc)
        detail = selector or f"({x}, {y})"
        if selector:
            locator = self._page.locator(selector).first
            await locator.wait_for(state="visible", timeout=self.timeout_ms)
            await locator.scroll_into_view_if_needed(timeout=self.timeout_ms)
            bbox = await locator.bounding_box()
            if bbox:
                tx = float(bbox["x"]) + float(bbox["width"]) / 2.0
                ty = float(bbox["y"]) + float(bbox["height"]) / 2.0
                await self._move_mouse_human_like(tx, ty)
                await self._sleep_ms(40)
                await self._page.mouse.click(tx, ty, delay=70)
            else:
                await locator.click(timeout=self.timeout_ms)
        elif x is not None and y is not None:
            await self._move_mouse_human_like(float(x), float(y))
            await self._sleep_ms(35)
            await self._page.mouse.click(float(x), float(y), delay=70)
        else:
            raise ValueError("browser_click requires either 'selector' or 'x'/'y'")
        await self._settle_after_action(280)
        observation = await self._capture_observation(
            action="browser_click",
            detail=detail,
            include_accessibility=True,
        )
        observation.action_time_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return observation

    async def type_text(
        self,
        *,
        selector: str,
        text: str,
        clear_first: bool = False,
        submit: bool = False,
    ) -> BrowserObservation:
        await self._ensure_initialized()
        assert self._page is not None
        start = datetime.now(timezone.utc)
        locator = self._page.locator(selector).first
        await locator.wait_for(state="visible", timeout=self.timeout_ms)
        await locator.scroll_into_view_if_needed(timeout=self.timeout_ms)
        bbox = await locator.bounding_box()
        if bbox:
            tx = float(bbox["x"]) + min(float(bbox["width"]) * 0.35, 18.0)
            ty = float(bbox["y"]) + float(bbox["height"]) / 2.0
            await self._move_mouse_human_like(tx, ty)
            await self._page.mouse.click(tx, ty, delay=55)
        else:
            await locator.click(timeout=self.timeout_ms)
        if clear_first:
            await self._page.keyboard.press("Control+A")
            await self._sleep_ms(25)
            await self._page.keyboard.press("Backspace")
            await self._sleep_ms(40)
        for idx, char in enumerate(text):
            delay_ms = 28 + ((idx % 5) * 11)
            await self._page.keyboard.type(char, delay=delay_ms)
            if char in ",.!?":
                await self._sleep_ms(55)
        if submit:
            await self._sleep_ms(70)
            await self._page.keyboard.press("Enter")
        await self._settle_after_action(220)
        observation = await self._capture_observation(
            action="browser_type",
            detail=f"{selector} ({len(text)} chars{' + submit' if submit else ''})",
            include_accessibility=True,
        )
        observation.action_time_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return observation

    async def scroll(self, *, direction: str = "down", amount: int = 600) -> BrowserObservation:
        await self._ensure_initialized()
        assert self._page is not None
        start = datetime.now(timezone.utc)
        steps = max(1, min(8, int(abs(amount) / 160) + 1))
        chunk = max(40, int(abs(amount) / steps))
        normalized = direction.lower()
        if normalized == "top":
            await self._page.evaluate("window.scrollTo({ top: 0, behavior: 'smooth' })")
        elif normalized == "bottom":
            await self._page.evaluate("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })")
        else:
            wheel_y = chunk if normalized == "down" else -chunk
            for _ in range(steps):
                await self._page.mouse.wheel(0, wheel_y)
                await self._sleep_ms(70)
        await self._settle_after_action(200)
        observation = await self._capture_observation(
            action="browser_scroll",
            detail=f"{normalized} {amount}",
            include_accessibility=True,
        )
        observation.action_time_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return observation

    async def screenshot(self, *, full_page: bool = False) -> BrowserObservation:
        await self._ensure_initialized()
        start = datetime.now(timezone.utc)
        observation = await self._capture_observation(
            action="browser_screenshot",
            detail="full_page" if full_page else "viewport",
            include_accessibility=True,
            full_page=full_page,
        )
        observation.action_time_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return observation

    async def close(self) -> None:
        for obj_name in ("_page", "_context", "_browser"):
            obj = getattr(self, obj_name, None)
            if obj is not None:
                try:
                    await obj.close()
                except Exception:
                    pass
                setattr(self, obj_name, None)
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        if self._playwright_manager is not None:
            self._playwright_manager = None
        self._playwright = None
        self._initialized = False


class AdvancedBrowserActionTool(BaseBrowserTool):
    """Thin wrapper exposing one browser action as one tool."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        session: AdvancedBrowserSession,
    ) -> None:
        super().__init__(name=name, description=description)
        self._session = session

    @staticmethod
    def _format_message(observation: BrowserObservation) -> str:
        parts = [
            f"Action: {observation.action}",
            f"URL: {observation.url}",
            f"Title: {observation.title}",
        ]
        if observation.detail:
            parts.append(f"Detail: {observation.detail}")
        if observation.viewport_width and observation.viewport_height:
            parts.append(f"Viewport: {observation.viewport_width}x{observation.viewport_height}")
        parts.append(
            f"Interactive elements: {len(observation.interactive_elements)} "
            "(use selector when possible, otherwise inspect coordinates from these targets)"
        )
        for item in observation.interactive_elements[:15]:
            label = item.get("text") or item.get("aria_label") or item.get("placeholder") or item.get("tag") or ""
            parts.append(
                f"- {item.get('id')} <{item.get('tag')}> '{label[:80]}' "
                f"at ({item.get('x')}, {item.get('y')}) size={item.get('width')}x{item.get('height')}"
            )
        if observation.screenshot_base64:
            parts.append("[Screenshot captured and streamed to Browser Tab]")
        if observation.accessibility_tree_json:
            parts.append("--- Accessibility tree ---")
            parts.append(observation.accessibility_tree_json[:7000])
        if observation.page_errors:
            parts.append("--- Page errors ---")
            parts.extend(observation.page_errors[:10])
        if observation.network_failures:
            parts.append("--- Network failures ---")
            parts.extend(observation.network_failures[:10])
        if observation.console_messages:
            parts.append("--- Console ---")
            parts.extend(observation.console_messages[:20])
        parts.append(f"Action time: {observation.action_time_ms}ms")
        return "\n".join(parts)

    async def execute(self, input_data: Any) -> ToolResult:
        try:
            if self.name == "browser_navigate":
                url = self._get(input_data, "url", "")
                wait_until = self._get(input_data, "wait_until", "domcontentloaded")
                observation = await self._session.navigate(url, wait_until=wait_until)
            elif self.name == "browser_click":
                observation = await self._session.click(
                    selector=self._get(input_data, "selector"),
                    x=self._get(input_data, "x"),
                    y=self._get(input_data, "y"),
                )
            elif self.name == "browser_type":
                observation = await self._session.type_text(
                    selector=self._get(input_data, "selector", ""),
                    text=self._get(input_data, "text", ""),
                    clear_first=bool(self._get(input_data, "clear_first", False)),
                    submit=bool(self._get(input_data, "submit", False)),
                )
            elif self.name == "browser_scroll":
                observation = await self._session.scroll(
                    direction=str(self._get(input_data, "direction", "down")),
                    amount=int(self._get(input_data, "amount", 600)),
                )
            elif self.name == "browser_screenshot":
                observation = await self._session.screenshot(
                    full_page=bool(self._get(input_data, "full_page", False)),
                )
            else:
                raise ValueError(f"Unsupported advanced browser tool: {self.name}")
            return ToolResult(
                success=True,
                data=observation,
                message=self._format_message(observation),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                data=None,
                message=f"Error: {self.name} failed: {exc}",
                error=str(exc),
            )

    async def close(self) -> None:
        await self._session.close()

    @staticmethod
    def _get(input_data: Any, key: str, default: Any = None) -> Any:
        if isinstance(input_data, dict):
            return input_data.get(key, default)
        return getattr(input_data, key, default)


def create_advanced_browser_tools(
    *,
    headless: bool = True,
    window_width: int = 1440,
    window_height: int = 900,
    timeout_ms: int = 30_000,
    max_interactive_elements: int = 80,
    on_browser_event: Callable[[dict[str, Any]], Any] | None = None,
) -> list[AdvancedBrowserActionTool]:
    """Create the advanced browser tool suite backed by one shared session."""

    session = AdvancedBrowserSession(
        headless=headless,
        window_width=window_width,
        window_height=window_height,
        timeout_ms=timeout_ms,
        max_interactive_elements=max_interactive_elements,
        on_browser_event=on_browser_event,
    )
    return [
        AdvancedBrowserActionTool(
            name="browser_navigate",
            description=(
                "Open a URL in a persistent Chromium session, then return a screenshot, "
                "accessibility tree, and clickable element map."
            ),
            session=session,
        ),
        AdvancedBrowserActionTool(
            name="browser_click",
            description=(
                "Click an element using a CSS selector or raw x/y coordinates with "
                "human-like mouse movement. Returns a fresh screenshot and accessibility map."
            ),
            session=session,
        ),
        AdvancedBrowserActionTool(
            name="browser_type",
            description=(
                "Type text into an element with human-like keystroke pacing in the persistent "
                "browser session. Supports optional Enter submit after typing. Returns a fresh "
                "screenshot and accessibility map."
            ),
            session=session,
        ),
        AdvancedBrowserActionTool(
            name="browser_scroll",
            description=(
                "Scroll the page up, down, to top, or to bottom with gradual wheel motion. "
                "Returns a fresh screenshot and accessibility map."
            ),
            session=session,
        ),
        AdvancedBrowserActionTool(
            name="browser_screenshot",
            description=(
                "Capture the current page as a screenshot together with the accessibility tree "
                "and interactive element map so the agent can plan the next action."
            ),
            session=session,
        ),
    ]
