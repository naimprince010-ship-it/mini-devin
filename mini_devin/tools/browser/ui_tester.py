"""
Browser-Based UI Test Runner

Lets the agent (or API caller) define structured UI test suites and run them
against a live URL using Playwright.  Each test step can:
  - assert element exists / is visible
  - assert element contains expected text
  - click a button/link and assert the result
  - fill a form field
  - assert URL changed to expected value
  - take a screenshot (optionally compared for visual regression)
  - run arbitrary JavaScript and assert the result

Results include per-step pass/fail, timing, error messages, and screenshots.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .visual_regression import get_engine, DiffResult


class StepType(str, Enum):
    NAVIGATE = "navigate"
    ASSERT_ELEMENT = "assert_element"       # element exists and is visible
    ASSERT_TEXT = "assert_text"             # element contains text
    ASSERT_URL = "assert_url"               # current URL matches
    ASSERT_TITLE = "assert_title"           # page title matches
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    PRESS_KEY = "press_key"
    WAIT = "wait"                           # sleep N ms
    WAIT_FOR = "wait_for_selector"
    SCREENSHOT = "screenshot"              # capture (+ optional regression)
    EVALUATE = "evaluate"                  # JS assertion
    ASSERT_NO_JS_ERRORS = "assert_no_js_errors"
    HOVER = "hover"
    SCROLL = "scroll"


@dataclass
class TestStep:
    type: StepType
    # Common fields
    selector: Optional[str] = None
    url: Optional[str] = None
    text: Optional[str] = None            # expected text / value to type
    value: Optional[str] = None           # for fill / select
    script: Optional[str] = None          # JS for evaluate
    expected: Any = None                  # expected return value of evaluate
    timeout_ms: int = 10_000
    ms: int = 500                         # for wait
    screenshot_name: Optional[str] = None # for screenshot step
    set_baseline: bool = False            # set new baseline instead of compare
    threshold_percent: Optional[float] = None  # visual regression threshold
    description: Optional[str] = None


@dataclass
class StepResult:
    step_index: int
    step_type: str
    description: str
    passed: bool
    duration_ms: int
    error: Optional[str] = None
    screenshot_b64: Optional[str] = None   # thumbnail for report
    diff: Optional[Dict[str, Any]] = None  # visual regression diff info


@dataclass
class TestSuiteResult:
    suite_name: str
    url: str
    passed: bool
    total_steps: int
    passed_steps: int
    failed_steps: int
    duration_ms: int
    steps: List[StepResult] = field(default_factory=list)
    ran_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    js_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def summary(self) -> str:
        icon = "✅" if self.passed else "❌"
        return (
            f"{icon} {self.suite_name}: "
            f"{self.passed_steps}/{self.total_steps} steps passed "
            f"in {self.duration_ms}ms"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class UITestRunner:
    """Runs a sequence of UI test steps against a page using Playwright."""

    def __init__(
        self,
        working_dir: str = ".",
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        on_screenshot: Optional[Any] = None,  # callback(b64) for live preview
    ):
        self.working_dir = working_dir
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.on_screenshot = on_screenshot
        self._js_errors: List[str] = []

    async def run(
        self,
        suite_name: str,
        start_url: str,
        steps: List[TestStep],
        threshold_percent: float = 0.5,
    ) -> TestSuiteResult:
        """Execute the test suite and return a full report."""
        suite_start = time.monotonic()
        step_results: List[StepResult] = []
        self._js_errors = []

        try:
            from playwright.async_api import async_playwright  # type: ignore
        except ImportError:
            return TestSuiteResult(
                suite_name=suite_name,
                url=start_url,
                passed=False,
                total_steps=len(steps),
                passed_steps=0,
                failed_steps=len(steps),
                duration_ms=0,
                steps=[StepResult(
                    step_index=0,
                    step_type="setup",
                    description="Playwright not installed",
                    passed=False,
                    duration_ms=0,
                    error="playwright package not available; run: pip install playwright && playwright install chromium",
                )],
            )

        async with async_playwright() as pw:
            # Try Browserless first, fall back to local
            browser_ws = self._get_browserless_ws_url()
            try:
                if browser_ws:
                    browser = await pw.chromium.connect_over_cdp(browser_ws)
                else:
                    browser = await pw.chromium.launch(headless=self.headless)
            except Exception as e:
                browser = await pw.chromium.launch(headless=self.headless)

            context = await browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                ignore_https_errors=True,
            )
            page = await context.new_page()

            # Capture JS errors
            page.on("pageerror", lambda err: self._js_errors.append(str(err)))
            page.on("console", lambda msg: (
                self._js_errors.append(f"[{msg.type}] {msg.text}")
                if msg.type == "error" else None
            ))

            # Navigate to start URL
            try:
                await page.goto(start_url, wait_until="domcontentloaded", timeout=30_000)
            except Exception as e:
                await browser.close()
                return TestSuiteResult(
                    suite_name=suite_name,
                    url=start_url,
                    passed=False,
                    total_steps=len(steps),
                    passed_steps=0,
                    failed_steps=len(steps),
                    duration_ms=int((time.monotonic() - suite_start) * 1000),
                    steps=[StepResult(
                        step_index=0,
                        step_type="navigate",
                        description=f"Navigate to {start_url}",
                        passed=False,
                        duration_ms=0,
                        error=f"Failed to load page: {e}",
                    )],
                    js_errors=self._js_errors,
                )

            vr_engine = get_engine(self.working_dir)

            for idx, step in enumerate(steps):
                step_start = time.monotonic()
                desc = step.description or f"{step.type.value} step {idx + 1}"
                sr = await self._run_step(
                    page, idx, step, desc, vr_engine,
                    threshold_percent=threshold_percent,
                )
                step_results.append(sr)

                if not sr.passed and step.type not in (
                    StepType.SCREENSHOT, StepType.ASSERT_NO_JS_ERRORS
                ):
                    # Non-screenshot failures stop the suite
                    # (remaining steps marked as skipped)
                    for remaining_idx in range(idx + 1, len(steps)):
                        step_results.append(StepResult(
                            step_index=remaining_idx,
                            step_type=steps[remaining_idx].type.value,
                            description=steps[remaining_idx].description or f"step {remaining_idx + 1}",
                            passed=False,
                            duration_ms=0,
                            error="Skipped due to earlier failure",
                        ))
                    break

            await browser.close()

        passed_count = sum(1 for s in step_results if s.passed)
        total_ms = int((time.monotonic() - suite_start) * 1000)

        return TestSuiteResult(
            suite_name=suite_name,
            url=start_url,
            passed=passed_count == len(steps),
            total_steps=len(steps),
            passed_steps=passed_count,
            failed_steps=len(steps) - passed_count,
            duration_ms=total_ms,
            steps=step_results,
            js_errors=self._js_errors,
        )

    @staticmethod
    def _get_browserless_ws_url() -> Optional[str]:
        import os
        key = os.getenv("BROWSERLESS_API_KEY", "")
        ws = os.getenv("BROWSERLESS_WS_URL", "")
        if ws:
            return ws
        if key:
            return f"wss://chrome.browserless.io?token={key}"
        return None

    async def _take_screenshot_b64(self, page: Any) -> str:
        try:
            b = await page.screenshot(full_page=False)
            import base64
            return base64.b64encode(b).decode()
        except Exception:
            return ""

    async def _run_step(
        self,
        page: Any,
        idx: int,
        step: TestStep,
        desc: str,
        vr_engine: Any,
        threshold_percent: float,
    ) -> StepResult:
        t = time.monotonic()
        try:
            result = await self._execute_step(page, step, vr_engine, threshold_percent)
            elapsed = int((time.monotonic() - t) * 1000)
            if isinstance(result, StepResult):
                result.step_index = idx
                result.duration_ms = elapsed
                result.description = desc
                return result
            # result is (passed, error, screenshot_b64, diff)
            passed, error, sc_b64, diff = result
            return StepResult(
                step_index=idx,
                step_type=step.type.value,
                description=desc,
                passed=passed,
                duration_ms=elapsed,
                error=error,
                screenshot_b64=sc_b64,
                diff=diff,
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - t) * 1000)
            return StepResult(
                step_index=idx,
                step_type=step.type.value,
                description=desc,
                passed=False,
                duration_ms=elapsed,
                error=str(exc),
            )

    async def _execute_step(
        self,
        page: Any,
        step: TestStep,
        vr_engine: Any,
        threshold_percent: float,
    ):
        """Returns (passed, error, screenshot_b64, diff_dict) or a StepResult."""
        import base64

        stype = step.type

        if stype == StepType.NAVIGATE:
            await page.goto(step.url or "", wait_until="domcontentloaded", timeout=step.timeout_ms)
            return True, None, None, None

        elif stype == StepType.ASSERT_ELEMENT:
            loc = page.locator(step.selector)
            try:
                await loc.wait_for(state="visible", timeout=step.timeout_ms)
                return True, None, None, None
            except Exception as e:
                sc = await self._take_screenshot_b64(page)
                return False, f"Element '{step.selector}' not found/visible: {e}", sc, None

        elif stype == StepType.ASSERT_TEXT:
            loc = page.locator(step.selector)
            try:
                await loc.wait_for(state="visible", timeout=step.timeout_ms)
                actual = (await loc.text_content() or "").strip()
                expected = (step.text or "").strip()
                if expected.lower() in actual.lower():
                    return True, None, None, None
                sc = await self._take_screenshot_b64(page)
                return False, f"Text mismatch in '{step.selector}': expected '{expected}' in '{actual[:200]}'", sc, None
            except Exception as e:
                sc = await self._take_screenshot_b64(page)
                return False, str(e), sc, None

        elif stype == StepType.ASSERT_URL:
            current = page.url
            expected = step.url or ""
            passed = expected in current
            if not passed:
                sc = await self._take_screenshot_b64(page)
                return False, f"URL mismatch: expected '{expected}' in '{current}'", sc, None
            return True, None, None, None

        elif stype == StepType.ASSERT_TITLE:
            title = await page.title()
            expected = step.text or ""
            if expected.lower() in title.lower():
                return True, None, None, None
            sc = await self._take_screenshot_b64(page)
            return False, f"Title mismatch: expected '{expected}' in '{title}'", sc, None

        elif stype == StepType.CLICK:
            try:
                await page.locator(step.selector).click(timeout=step.timeout_ms)
                return True, None, None, None
            except Exception as e:
                sc = await self._take_screenshot_b64(page)
                return False, f"Click failed on '{step.selector}': {e}", sc, None

        elif stype == StepType.FILL:
            try:
                await page.locator(step.selector).fill(step.value or "", timeout=step.timeout_ms)
                return True, None, None, None
            except Exception as e:
                return False, f"Fill failed on '{step.selector}': {e}", None, None

        elif stype == StepType.SELECT:
            try:
                await page.locator(step.selector).select_option(step.value or "", timeout=step.timeout_ms)
                return True, None, None, None
            except Exception as e:
                return False, f"Select failed on '{step.selector}': {e}", None, None

        elif stype == StepType.PRESS_KEY:
            try:
                if step.selector:
                    await page.locator(step.selector).press(step.value or "Enter", timeout=step.timeout_ms)
                else:
                    await page.keyboard.press(step.value or "Enter")
                return True, None, None, None
            except Exception as e:
                return False, str(e), None, None

        elif stype == StepType.WAIT:
            await asyncio.sleep(step.ms / 1000)
            return True, None, None, None

        elif stype == StepType.WAIT_FOR:
            try:
                await page.locator(step.selector).wait_for(state="visible", timeout=step.timeout_ms)
                return True, None, None, None
            except Exception as e:
                sc = await self._take_screenshot_b64(page)
                return False, f"Timed out waiting for '{step.selector}': {e}", sc, None

        elif stype == StepType.HOVER:
            try:
                await page.locator(step.selector).hover(timeout=step.timeout_ms)
                return True, None, None, None
            except Exception as e:
                return False, str(e), None, None

        elif stype == StepType.SCROLL:
            try:
                if step.selector:
                    await page.locator(step.selector).scroll_into_view_if_needed(timeout=step.timeout_ms)
                else:
                    await page.evaluate("window.scrollBy(0, 500)")
                return True, None, None, None
            except Exception as e:
                return False, str(e), None, None

        elif stype == StepType.EVALUATE:
            try:
                result = await page.evaluate(step.script or "null")
                if step.expected is not None:
                    if result != step.expected:
                        return False, f"JS returned {result!r}, expected {step.expected!r}", None, None
                return True, None, None, None
            except Exception as e:
                return False, f"JS evaluation error: {e}", None, None

        elif stype == StepType.ASSERT_NO_JS_ERRORS:
            if self._js_errors:
                return False, f"JS errors found: {'; '.join(self._js_errors[:5])}", None, None
            return True, None, None, None

        elif stype == StepType.SCREENSHOT:
            sc_bytes = await page.screenshot(full_page=True)
            sc_b64 = base64.b64encode(sc_bytes).decode()

            if self.on_screenshot:
                try:
                    self.on_screenshot(sc_b64)
                except Exception:
                    pass

            name = step.screenshot_name or f"screenshot_{int(time.time())}"
            diff_dict: Optional[Dict[str, Any]] = None

            if step.set_baseline:
                vr_engine.save_screenshot(name, sc_bytes, set_as_baseline=True)
                return True, None, sc_b64, None
            else:
                try:
                    diff: DiffResult = vr_engine.compare(
                        name,
                        sc_bytes,
                        threshold_percent=step.threshold_percent or threshold_percent,
                    )
                    diff_dict = diff.to_dict()
                    passed = diff.passed
                    error = None if passed else (
                        f"Visual regression: {diff.changed_percent:.2f}% pixels changed "
                        f"(threshold {diff.threshold_percent}%)"
                    )
                    return passed, error, sc_b64, diff_dict
                except FileNotFoundError:
                    # No baseline yet — auto-set this as baseline
                    vr_engine.save_screenshot(name, sc_bytes, set_as_baseline=True)
                    return True, f"No baseline found — set '{name}' as new baseline", sc_b64, None

        return False, f"Unknown step type: {stype}", None, None


# ---------------------------------------------------------------------------
# Convenience: build test steps from a simple dict spec
# ---------------------------------------------------------------------------

def steps_from_spec(spec: List[Dict[str, Any]]) -> List[TestStep]:
    """
    Convert a list of dicts (from JSON/agent output) to TestStep objects.

    Example spec item:
      {"type": "click", "selector": "button#submit", "description": "Submit form"}
      {"type": "assert_text", "selector": "h1", "text": "Welcome"}
      {"type": "screenshot", "screenshot_name": "homepage", "set_baseline": false}
    """
    steps = []
    for item in spec:
        t = item.get("type", "")
        try:
            stype = StepType(t)
        except ValueError:
            continue
        steps.append(TestStep(
            type=stype,
            selector=item.get("selector"),
            url=item.get("url"),
            text=item.get("text"),
            value=item.get("value"),
            script=item.get("script"),
            expected=item.get("expected"),
            timeout_ms=int(item.get("timeout_ms", 10_000)),
            ms=int(item.get("ms", 500)),
            screenshot_name=item.get("screenshot_name"),
            set_baseline=bool(item.get("set_baseline", False)),
            threshold_percent=item.get("threshold_percent"),
            description=item.get("description"),
        ))
    return steps
