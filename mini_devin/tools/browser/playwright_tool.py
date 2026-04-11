"""
Playwright Browser Automation Tool for Mini-Devin

This module replaces Selenium with Microsoft Playwright for visual browser automation:
- Navigate pages and capture full-page screenshots
- Click, type, scroll, hover on any element
- Wait for elements, network idle, or URLs
- Execute JavaScript
- Extract text/HTML content
- Highlight elements for visual debugging
- Find all matching elements with bounding boxes
- Export pages as PDF
- Stream screenshots to frontend via WebSocket callbacks

Uses async Playwright (fully async/await compatible with the agent loop).
"""

import asyncio
import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from .base import BaseBrowserTool, ToolResult


class PlaywrightAction(str, Enum):
    """Supported Playwright browser actions."""
    NAVIGATE = "navigate"
    SCREENSHOT = "screenshot"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    SCROLL = "scroll"
    HOVER = "hover"
    WAIT_FOR = "wait_for"
    GET_TEXT = "get_text"
    GET_HTML = "get_html"
    EVALUATE = "evaluate"
    HIGHLIGHT = "highlight"
    FIND_ELEMENTS = "find_elements"
    PDF = "pdf"
    FILL = "fill"
    PRESS = "press"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    RELOAD = "reload"
    DEBUG_SNAPSHOT = "debug_snapshot"


@dataclass
class ElementBoundingBox:
    """Bounding box information for an element."""
    selector: str
    x: float
    y: float
    width: float
    height: float
    text: str = ""
    tag: str = ""
    is_visible: bool = True


@dataclass
class BrowserPageState:
    """State of the browser page after an action."""
    url: str
    title: str
    screenshot_base64: Optional[str] = None
    text: Optional[str] = None
    html: Optional[str] = None
    elements: list[ElementBoundingBox] = field(default_factory=list)
    evaluate_result: Any = None
    action_time_ms: int = 0
    dom_outline: Optional[str] = None


@dataclass
class PlaywrightResponse:
    """Response from a Playwright browser action."""
    success: bool
    action: PlaywrightAction
    page_state: Optional[BrowserPageState] = None
    error: Optional[str] = None
    pdf_base64: Optional[str] = None


class PlaywrightBrowserTool(BaseBrowserTool):
    """
    Visual browser automation tool powered by Microsoft Playwright.

    Capabilities:
    - Full-page and viewport screenshots (base64 PNG)
    - Element interaction: click, type, hover, fill, select, press
    - Smart waiting: network idle, selectors, URLs
    - JavaScript execution and evaluation
    - Visual debug: element highlight + screenshot
    - DOM extraction: text, HTML, element bounding boxes
    - PDF export
    - Live screenshot streaming to frontend Browser Tab

    Usage by the agent:
    - Use `navigate` to open any URL
    - Use `screenshot` to capture the current state
    - Use `click` with a CSS selector to interact with elements
    - Use `evaluate` to run JavaScript for complex interactions
    - Use `highlight` to visually confirm which element you're targeting
    """

    def __init__(
        self,
        headless: bool = True,
        window_width: int = 1280,
        window_height: int = 800,
        timeout_ms: int = 30000,
        screenshot_on_action: bool = True,
        on_browser_event: Optional[Callable] = None,
    ):
        super().__init__(
            name="browser_playwright",
            description=(
                "Headless Chromium (Playwright): navigate, click, type, screenshots, DOM/HTML, run JS. "
                "Captures DevTools-style signals — console (log/warn/error), uncaught page errors, "
                "XHR/fetch/document network lines, and a compact DOM outline — on navigate/reload/wait "
                "and on action=debug_snapshot."
            ),
        )
        self.headless = headless
        self.window_width = window_width
        self.window_height = window_height
        self.timeout_ms = timeout_ms
        self.screenshot_on_action = screenshot_on_action
        self.on_browser_event = on_browser_event  # Callback to stream to frontend

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False
        self._mode = "unknown"  # 'browserless', 'local', or 'unavailable'
        self._listeners_attached = False
        self._debug_console: list[str] = []
        self._debug_page_errors: list[str] = []
        self._debug_network: list[str] = []
        self._debug_request_failed: list[str] = []

    @staticmethod
    def _get_browserless_ws_url() -> Optional[str]:
        """
        Build Browserless WebSocket endpoint from environment.

        Supports:
          - BROWSERLESS_API_KEY  → wss://chrome.browserless.io?token=KEY
          - BROWSERLESS_WS_URL   → custom self-hosted endpoint
        """
        import os
        ws_url = os.getenv("BROWSERLESS_WS_URL")
        if ws_url:
            return ws_url
        api_key = os.getenv("BROWSERLESS_API_KEY")
        if api_key:
            return f"wss://chrome.browserless.io?token={api_key}"
        return None

    async def _ensure_initialized(self) -> None:
        """
        Lazily initialize browser on first use.

        Priority:
          1. Browserless.io remote Chromium (if BROWSERLESS_API_KEY / BROWSERLESS_WS_URL set)
          2. Local Chromium via Playwright
          3. Raise clear error if neither available
        """
        if self._initialized:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is required. Install with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )

        self._playwright_manager = async_playwright()
        self._playwright = await self._playwright_manager.start()

        # ── Option 1: Browserless remote Chromium ──────────────────────────
        ws_url = self._get_browserless_ws_url()
        if ws_url:
            try:
                self._browser = await self._playwright.chromium.connect_over_cdp(ws_url)
                self._context = await self._browser.new_context(
                    viewport={"width": self.window_width, "height": self.window_height},
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                )
                self._page = await self._context.new_page()
                self._page.set_default_timeout(self.timeout_ms)
                self._attach_debug_listeners()
                self._initialized = True
                self._mode = "browserless"
                return
            except Exception as e:
                # Browserless connection failed — try local fallback
                print(f"[Browser] Browserless connection failed ({e}), trying local Chromium...")

        # ── Option 2: Local Chromium ────────────────────────────────────────
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",
                ],
            )
            self._context = await self._browser.new_context(
                viewport={"width": self.window_width, "height": self.window_height},
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.timeout_ms)
            self._attach_debug_listeners()
            self._initialized = True
            self._mode = "local"
        except Exception as e:
            raise RuntimeError(
                f"Browser unavailable: {e}\n"
                "To enable browser automation:\n"
                "  • Set BROWSERLESS_API_KEY in .env (remote, no install needed)\n"
                "  • Or run: playwright install chromium (local)"
            )

    async def _take_screenshot(self, full_page: bool = False) -> Optional[str]:
        """Take a screenshot and return base64-encoded PNG."""
        if not self._page:
            return None
        try:
            screenshot_bytes = await self._page.screenshot(full_page=full_page)
            return base64.b64encode(screenshot_bytes).decode("utf-8")
        except Exception:
            return None

    async def _get_page_state(self, screenshot: bool = True, full_page: bool = False) -> BrowserPageState:
        """Gather current page state."""
        url = self._page.url
        title = await self._page.title()
        screenshot_b64 = None
        if screenshot and self.screenshot_on_action:
            screenshot_b64 = await self._take_screenshot(full_page=full_page)
        return BrowserPageState(url=url, title=title, screenshot_base64=screenshot_b64)

    def _clear_debug_buffers(self) -> None:
        self._debug_console.clear()
        self._debug_page_errors.clear()
        self._debug_network.clear()
        self._debug_request_failed.clear()

    def _attach_debug_listeners(self) -> None:
        """Wire Playwright page events (console, errors, network) for agent-visible diagnostics."""
        if self._listeners_attached or not self._page:
            return
        page = self._page

        def on_console(msg: Any) -> None:
            try:
                line = f"[{msg.type}] {msg.text}"[:4000]
                self._debug_console.append(line)
                if len(self._debug_console) > 250:
                    self._debug_console = self._debug_console[-250:]
            except Exception:
                pass

        def on_page_error(err: Any) -> None:
            try:
                self._debug_page_errors.append(str(err)[:4000])
                if len(self._debug_page_errors) > 60:
                    self._debug_page_errors = self._debug_page_errors[-60:]
            except Exception:
                pass

        def on_response(response: Any) -> None:
            try:
                req = response.request
                url = response.url
                if url.startswith(("data:", "blob:", "chrome-extension:")):
                    return
                rt = req.resource_type
                if rt not in ("document", "xhr", "fetch", "websocket"):
                    return
                status = response.status
                line = f"{req.method} {status} [{rt}] {url[:800]}"
                self._debug_network.append(line)
                if len(self._debug_network) > 150:
                    self._debug_network = self._debug_network[-150:]
            except Exception:
                pass

        def on_request_failed(request: Any) -> None:
            try:
                fail = request.failure
                ftxt = str(fail) if fail else "?"
                line = f"FAILED {request.method} {request.url[:600]} ({ftxt})"
                self._debug_request_failed.append(line)
                if len(self._debug_request_failed) > 50:
                    self._debug_request_failed = self._debug_request_failed[-50:]
            except Exception:
                pass

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)
        self._listeners_attached = True

    def _format_browser_diagnostics(self, max_chars: int = 16000) -> str:
        """Format captured console / network / errors for the LLM."""
        parts: list[str] = []
        if self._debug_page_errors:
            parts.append("--- Page errors (uncaught) ---")
            parts.extend(self._debug_page_errors[-25:])
        if self._debug_request_failed:
            parts.append("\n--- Failed network requests ---")
            parts.extend(self._debug_request_failed[-30:])
        if self._debug_console:
            parts.append("\n--- Console ---")
            # Prefer errors/warnings first if present
            errs = [l for l in self._debug_console if l.startswith("[error]") or l.startswith("[warning]")]
            rest = [l for l in self._debug_console if l not in errs]
            merged = errs[-40:] + rest[-(120 - min(40, len(errs))) :]
            parts.extend(merged[-140:])
        if self._debug_network:
            parts.append("\n--- Network (document / xhr / fetch / websocket) ---")
            parts.extend(self._debug_network[-90:])
        text = "\n".join(parts).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 80] + "\n… (truncated)"
        return text

    async def _get_dom_outline(self) -> str:
        """Compact JSON outline: headings + body child tags (DevTools Elements-lite)."""
        script = r"""() => {
  const pick = (sel, n) => Array.from(document.querySelectorAll(sel)).slice(0, n).map(e => ({
    t: e.tagName.toLowerCase(),
    txt: (e.innerText || '').trim().slice(0, 64).replace(/\s+/g, ' ')
  }));
  const kids = Array.from(document.body.children).slice(0, 32).map(n => {
    const id = n.id ? ('#' + n.id) : '';
    const c = n.className && typeof n.className === 'string'
      ? '.' + n.className.trim().split(/\s+/).slice(0, 2).join('.') : '';
    return n.tagName.toLowerCase() + id + c;
  });
  return JSON.stringify({
    url: location.href,
    title: document.title,
    headings: pick('h1,h2,h3', 12),
    bodyChildren: kids
  });
}"""
        try:
            raw = await self._page.evaluate(script)
            if isinstance(raw, dict):
                return json.dumps(raw, indent=2, ensure_ascii=False)[:9000]
            if isinstance(raw, str):
                return raw[:9000]
            return json.dumps(raw, indent=2, default=str, ensure_ascii=False)[:9000]
        except Exception as e:
            return f"(dom_outline error: {e})"

    async def _debug_snapshot(self, input_data: Any) -> PlaywrightResponse:
        """Wait briefly, capture DOM outline + screenshot + diagnostics (no navigation)."""
        settle_ms = min(int(self._get(input_data, "settle_ms", 400)), 5000)
        await asyncio.sleep(settle_ms / 1000.0)
        full_page = self._get(input_data, "full_page", False)
        outline = await self._get_dom_outline()
        page_state = await self._get_page_state(screenshot=True, full_page=full_page)
        page_state.dom_outline = outline
        return PlaywrightResponse(success=True, action=PlaywrightAction.DEBUG_SNAPSHOT, page_state=page_state)

    async def _fire_event(self, action: str, page_state: BrowserPageState) -> None:
        """Stream browser event to frontend via callback."""
        if self.on_browser_event:
            try:
                event_data = {
                    "event_type": action,
                    "url": page_state.url,
                    "screenshot_base64": page_state.screenshot_base64,
                }
                result = self.on_browser_event(event_data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Never crash the tool because of callback errors

    async def execute(self, input_data: Any) -> ToolResult:
        """Execute a browser action."""
        start_time = datetime.now(timezone.utc)

        action_str = (
            input_data.get("action") if isinstance(input_data, dict)
            else getattr(input_data, "action", "screenshot")
        )

        try:
            action = PlaywrightAction(action_str)
        except ValueError:
            return ToolResult(
                success=False,
                data=None,
                message=f"Unknown action: {action_str}",
                error=f"Unknown action: '{action_str}'. Supported: {[a.value for a in PlaywrightAction]}",
            )

        try:
            await self._ensure_initialized()

            # Dispatch to action handler
            dispatch = {
                PlaywrightAction.NAVIGATE: self._navigate,
                PlaywrightAction.SCREENSHOT: self._screenshot,
                PlaywrightAction.CLICK: self._click,
                PlaywrightAction.TYPE: self._type,
                PlaywrightAction.FILL: self._fill,
                PlaywrightAction.SELECT: self._select,
                PlaywrightAction.SCROLL: self._scroll,
                PlaywrightAction.HOVER: self._hover,
                PlaywrightAction.WAIT_FOR: self._wait_for,
                PlaywrightAction.GET_TEXT: self._get_text,
                PlaywrightAction.GET_HTML: self._get_html,
                PlaywrightAction.EVALUATE: self._evaluate,
                PlaywrightAction.HIGHLIGHT: self._highlight,
                PlaywrightAction.FIND_ELEMENTS: self._find_elements,
                PlaywrightAction.PDF: self._pdf,
                PlaywrightAction.PRESS: self._press,
                PlaywrightAction.GO_BACK: self._go_back,
                PlaywrightAction.GO_FORWARD: self._go_forward,
                PlaywrightAction.RELOAD: self._reload,
                PlaywrightAction.DEBUG_SNAPSHOT: self._debug_snapshot,
            }

            handler = dispatch.get(action)
            if not handler:
                raise ValueError(f"No handler for action: {action}")

            response = await handler(input_data)

            end_time = datetime.now(timezone.utc)
            elapsed_ms = int((end_time - start_time).total_seconds() * 1000)
            if response.page_state:
                response.page_state.action_time_ms = elapsed_ms

            # Fire frontend event
            if response.page_state:
                await self._fire_event(action.value, response.page_state)

            # Build human-readable output for LLM
            output_parts = []
            if response.page_state:
                output_parts.append(f"URL: {response.page_state.url}")
                output_parts.append(f"Title: {response.page_state.title}")
                if response.page_state.text:
                    output_parts.append(f"Text Content:\n{response.page_state.text[:3000]}")
                if response.page_state.html:
                    output_parts.append(f"HTML Content:\n{response.page_state.html[:3000]}")
                if response.page_state.evaluate_result is not None:
                    output_parts.append(f"Evaluate Result: {response.page_state.evaluate_result}")
                if response.page_state.elements:
                    output_parts.append(f"Found {len(response.page_state.elements)} elements:")
                    for el in response.page_state.elements[:20]:
                        output_parts.append(
                            f"  - <{el.tag}> '{el.text[:60]}' at ({el.x:.0f},{el.y:.0f}) "
                            f"size={el.width:.0f}x{el.height:.0f}"
                        )
                if response.page_state.screenshot_base64:
                    output_parts.append("[Screenshot captured and streamed to Browser Tab]")
                if response.page_state.dom_outline:
                    output_parts.append("--- DOM outline ---\n" + response.page_state.dom_outline[:8000])
                output_parts.append(f"Action time: {elapsed_ms}ms")

            diag = self._format_browser_diagnostics()
            if diag:
                output_parts.append("\n--- DevTools-style diagnostics (console / network / page errors) ---\n" + diag)

            message = f"browser_playwright '{action.value}' succeeded"
            output_text = "\n".join(output_parts) if output_parts else message

            return ToolResult(
                success=True,
                data=response,
                message=output_text,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=PlaywrightResponse(
                    success=False,
                    action=action if "action" in dir() else PlaywrightAction.SCREENSHOT,
                    error=str(e),
                ),
                message=f"browser_playwright '{action_str}' failed: {e}",
                error=str(e),
            )

    # ── Action Handlers ──────────────────────────────────────────────────────

    async def _navigate(self, input_data: Any) -> PlaywrightResponse:
        """Navigate to a URL and capture full-page screenshot."""
        url = self._get(input_data, "url", "")
        wait_until = self._get(input_data, "wait_until", "domcontentloaded")  # or "networkidle"
        full_page = self._get(input_data, "full_page", False)

        self._clear_debug_buffers()
        await self._page.goto(url, wait_until=wait_until)
        page_state = await self._get_page_state(screenshot=True, full_page=full_page)
        page_state.dom_outline = await self._get_dom_outline()
        return PlaywrightResponse(success=True, action=PlaywrightAction.NAVIGATE, page_state=page_state)

    async def _screenshot(self, input_data: Any) -> PlaywrightResponse:
        """Take a screenshot (viewport or full-page)."""
        full_page = self._get(input_data, "full_page", False)
        selector = self._get(input_data, "selector", None)

        if selector:
            # Screenshot of specific element
            element = await self._page.query_selector(selector)
            if element:
                screenshot_bytes = await element.screenshot()
                b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            else:
                raise ValueError(f"Element not found: {selector}")
        else:
            screenshot_bytes = await self._page.screenshot(full_page=full_page)
            b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        page_state = BrowserPageState(
            url=self._page.url,
            title=await self._page.title(),
            screenshot_base64=b64,
        )
        return PlaywrightResponse(success=True, action=PlaywrightAction.SCREENSHOT, page_state=page_state)

    async def _click(self, input_data: Any) -> PlaywrightResponse:
        """Click an element by CSS selector or coordinates."""
        selector = self._get(input_data, "selector", None)
        x = self._get(input_data, "x", None)
        y = self._get(input_data, "y", None)

        if selector:
            await self._page.click(selector)
        elif x is not None and y is not None:
            await self._page.mouse.click(float(x), float(y))
        else:
            raise ValueError("'click' requires either 'selector' or 'x'+'y' coordinates")

        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.CLICK, page_state=page_state)

    async def _type(self, input_data: Any) -> PlaywrightResponse:
        """Type text into an element (simulates real keystrokes)."""
        selector = self._get(input_data, "selector", "")
        text = self._get(input_data, "text", "")
        delay = self._get(input_data, "delay", 30)  # ms between keystrokes

        await self._page.type(selector, text, delay=delay)
        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.TYPE, page_state=page_state)

    async def _fill(self, input_data: Any) -> PlaywrightResponse:
        """Fill a form field instantly (faster than type for large text)."""
        selector = self._get(input_data, "selector", "")
        value = self._get(input_data, "value", "")

        await self._page.fill(selector, value)
        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.FILL, page_state=page_state)

    async def _select(self, input_data: Any) -> PlaywrightResponse:
        """Select an option from a <select> dropdown."""
        selector = self._get(input_data, "selector", "")
        value = self._get(input_data, "value", None)
        label = self._get(input_data, "label", None)

        if value:
            await self._page.select_option(selector, value=value)
        elif label:
            await self._page.select_option(selector, label=label)
        else:
            raise ValueError("'select' requires 'value' or 'label'")

        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.SELECT, page_state=page_state)

    async def _scroll(self, input_data: Any) -> PlaywrightResponse:
        """Scroll the page or an element."""
        direction = self._get(input_data, "direction", "down")
        amount = self._get(input_data, "amount", 500)

        scroll_map = {
            "down": f"window.scrollBy(0, {amount})",
            "up": f"window.scrollBy(0, -{amount})",
            "top": "window.scrollTo(0, 0)",
            "bottom": "window.scrollTo(0, document.body.scrollHeight)",
        }
        script = scroll_map.get(direction, f"window.scrollBy(0, {amount})")
        await self._page.evaluate(script)

        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.SCROLL, page_state=page_state)

    async def _hover(self, input_data: Any) -> PlaywrightResponse:
        """Hover over an element to trigger tooltips/dropdowns."""
        selector = self._get(input_data, "selector", "")
        await self._page.hover(selector)
        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.HOVER, page_state=page_state)

    async def _wait_for(self, input_data: Any) -> PlaywrightResponse:
        """Wait for a selector, URL change, or network idle."""
        selector = self._get(input_data, "selector", None)
        url_pattern = self._get(input_data, "url", None)
        network_idle = self._get(input_data, "network_idle", False)
        timeout = self._get(input_data, "timeout", self.timeout_ms)

        if selector:
            await self._page.wait_for_selector(selector, timeout=timeout)
        elif url_pattern:
            await self._page.wait_for_url(url_pattern, timeout=timeout)
        elif network_idle:
            await self._page.wait_for_load_state("networkidle", timeout=timeout)
        else:
            await self._page.wait_for_load_state("domcontentloaded", timeout=timeout)

        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.WAIT_FOR, page_state=page_state)

    async def _get_text(self, input_data: Any) -> PlaywrightResponse:
        """Extract text content from page or element."""
        selector = self._get(input_data, "selector", None)

        if selector:
            text = await self._page.inner_text(selector)
        else:
            text = await self._page.inner_text("body")

        page_state = await self._get_page_state()
        page_state.text = text
        return PlaywrightResponse(success=True, action=PlaywrightAction.GET_TEXT, page_state=page_state)

    async def _get_html(self, input_data: Any) -> PlaywrightResponse:
        """Get HTML content from page or element."""
        selector = self._get(input_data, "selector", None)

        if selector:
            html = await self._page.inner_html(selector)
        else:
            html = await self._page.content()

        page_state = await self._get_page_state(screenshot=False)
        page_state.html = html
        return PlaywrightResponse(success=True, action=PlaywrightAction.GET_HTML, page_state=page_state)

    async def _evaluate(self, input_data: Any) -> PlaywrightResponse:
        """Execute JavaScript in the browser context."""
        script = self._get(input_data, "script", "null")
        result = await self._page.evaluate(script)
        page_state = await self._get_page_state()
        page_state.evaluate_result = result
        return PlaywrightResponse(success=True, action=PlaywrightAction.EVALUATE, page_state=page_state)

    async def _highlight(self, input_data: Any) -> PlaywrightResponse:
        """Highlight an element with a red outline and take a screenshot — visual debug."""
        selector = self._get(input_data, "selector", "")

        # Inject highlight CSS and take screenshot
        safe_selector = selector.replace("'", "\\'")
        await self._page.evaluate(f"""
            (function() {{
                const el = document.querySelector('{safe_selector}');
                if (el) {{
                    el.style.outline = '3px solid red';
                    el.style.outlineOffset = '2px';
                    el.style.boxShadow = '0 0 10px rgba(255,0,0,0.5)';
                }}
            }})();
        """)
        # Small delay to render
        await asyncio.sleep(0.2)

        screenshot_bytes = await self._page.screenshot()
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        # Remove highlight after screenshot
        safe_selector = selector.replace("'", "\\'")
        await self._page.evaluate(f"""
            (function() {{
                const el = document.querySelector('{safe_selector}');
                if (el) {{
                    el.style.outline = '';
                    el.style.outlineOffset = '';
                    el.style.boxShadow = '';
                }}
            }})();
        """)

        page_state = BrowserPageState(
            url=self._page.url,
            title=await self._page.title(),
            screenshot_base64=b64,
        )
        return PlaywrightResponse(success=True, action=PlaywrightAction.HIGHLIGHT, page_state=page_state)

    async def _find_elements(self, input_data: Any) -> PlaywrightResponse:
        """Find all elements matching a selector and return their bounding boxes."""
        selector = self._get(input_data, "selector", "a")

        elements = await self._page.query_selector_all(selector)
        result_elements: list[ElementBoundingBox] = []

        for el in elements[:50]:  # Limit to 50 elements
            try:
                bb = await el.bounding_box()
                tag = await el.evaluate("el => el.tagName.toLowerCase()")
                text = (await el.inner_text()).strip()[:80]
                is_visible = await el.is_visible()

                if bb:
                    result_elements.append(ElementBoundingBox(
                        selector=selector,
                        x=bb["x"],
                        y=bb["y"],
                        width=bb["width"],
                        height=bb["height"],
                        text=text,
                        tag=tag,
                        is_visible=is_visible,
                    ))
            except Exception:
                continue

        page_state = await self._get_page_state(screenshot=True)
        page_state.elements = result_elements
        return PlaywrightResponse(success=True, action=PlaywrightAction.FIND_ELEMENTS, page_state=page_state)

    async def _pdf(self, input_data: Any) -> PlaywrightResponse:
        """Export the current page as PDF."""
        format_type = self._get(input_data, "format", "A4")

        pdf_bytes = await self._page.pdf(format=format_type)
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        page_state = await self._get_page_state(screenshot=True)
        return PlaywrightResponse(
            success=True,
            action=PlaywrightAction.PDF,
            page_state=page_state,
            pdf_base64=pdf_base64,
        )

    async def _press(self, input_data: Any) -> PlaywrightResponse:
        """Press a keyboard key (e.g. Enter, Tab, Escape)."""
        key = self._get(input_data, "key", "Enter")
        selector = self._get(input_data, "selector", None)

        if selector:
            await self._page.press(selector, key)
        else:
            await self._page.keyboard.press(key)

        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.PRESS, page_state=page_state)

    async def _go_back(self, input_data: Any) -> PlaywrightResponse:
        """Navigate back in browser history."""
        await self._page.go_back()
        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.GO_BACK, page_state=page_state)

    async def _go_forward(self, input_data: Any) -> PlaywrightResponse:
        """Navigate forward in browser history."""
        await self._page.go_forward()
        page_state = await self._get_page_state()
        return PlaywrightResponse(success=True, action=PlaywrightAction.GO_FORWARD, page_state=page_state)

    async def _reload(self, input_data: Any) -> PlaywrightResponse:
        """Reload the current page."""
        self._clear_debug_buffers()
        await self._page.reload()
        page_state = await self._get_page_state()
        page_state.dom_outline = await self._get_dom_outline()
        return PlaywrightResponse(success=True, action=PlaywrightAction.RELOAD, page_state=page_state)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _get(input_data: Any, key: str, default: Any = None) -> Any:
        """Get a parameter from dict or object input."""
        if isinstance(input_data, dict):
            return input_data.get(key, default)
        return getattr(input_data, key, default)

    async def close(self) -> None:
        """Clean up Playwright resources."""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        finally:
            self._initialized = False
            self._listeners_attached = False
            self._clear_debug_buffers()
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None


def create_playwright_tool(
    headless: bool = True,
    window_width: int = 1280,
    window_height: int = 800,
    timeout_ms: int = 30000,
    screenshot_on_action: bool = True,
    on_browser_event: Optional[Callable] = None,
) -> PlaywrightBrowserTool:
    """Create a Playwright browser tool instance."""
    return PlaywrightBrowserTool(
        headless=headless,
        window_width=window_width,
        window_height=window_height,
        timeout_ms=timeout_ms,
        screenshot_on_action=screenshot_on_action,
        on_browser_event=on_browser_event,
    )
