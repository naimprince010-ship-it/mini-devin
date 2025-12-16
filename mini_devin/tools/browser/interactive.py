"""
Browser Interactive Tool for Mini-Devin

This module implements interactive browser automation using Selenium:
- Navigate to pages
- Click elements
- Fill forms
- Extract content from JS-heavy pages
- Take screenshots

Note: Uses headless Chrome/Chromium for automation.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .base import BaseBrowserTool, ToolResult


class BrowserAction(str, Enum):
    """Supported browser actions."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    GET_TEXT = "get_text"
    GET_HTML = "get_html"
    WAIT = "wait"
    EXECUTE_JS = "execute_js"


@dataclass
class ElementInfo:
    """Information about a page element."""
    tag: str
    text: str
    attributes: dict[str, str] = field(default_factory=dict)
    is_visible: bool = True
    location: tuple[int, int] | None = None
    size: tuple[int, int] | None = None


@dataclass
class PageState:
    """Current state of the browser page."""
    url: str
    title: str
    html: str | None = None
    text: str | None = None
    screenshot_base64: str | None = None
    elements: list[ElementInfo] = field(default_factory=list)


@dataclass
class InteractiveResponse:
    """Response from an interactive browser action."""
    success: bool
    action: BrowserAction
    page_state: PageState | None = None
    error: str | None = None
    action_time_ms: int = 0


class BrowserInteractiveTool(BaseBrowserTool):
    """
    Interactive browser automation tool using Selenium.
    
    Features:
    - Navigate to URLs
    - Click elements by selector
    - Type text into inputs
    - Scroll pages
    - Take screenshots
    - Extract text/HTML content
    - Execute JavaScript
    
    Note: Runs in headless mode only.
    """
    
    def __init__(
        self,
        headless: bool = True,
        window_size: tuple[int, int] = (1920, 1080),
        timeout: int = 30,
        user_agent: str | None = None,
    ):
        super().__init__(
            name="browser_interactive",
            description="Interactive browser automation for JS-heavy pages",
        )
        self.headless = True  # Always headless per system requirements
        self.window_size = window_size
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        self._driver = None
        self._initialized = False
    
    def _init_driver(self) -> None:
        """Initialize the Selenium WebDriver."""
        if self._initialized:
            return
        
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument(f"--window-size={self.window_size[0]},{self.window_size[1]}")
            options.add_argument(f"--user-agent={self.user_agent}")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            
            self._driver = webdriver.Chrome(options=options)
            self._driver.set_page_load_timeout(self.timeout)
            self._driver.implicitly_wait(10)
            self._initialized = True
            
        except ImportError:
            raise ImportError(
                "Selenium is required for interactive browser. "
                "Install with: pip install selenium"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize browser: {e}")
    
    async def execute(self, input_data: Any) -> ToolResult:
        """Execute a browser action."""
        action = BrowserAction(
            input_data.action if hasattr(input_data, "action") else "navigate"
        )
        
        start_time = datetime.utcnow()
        
        try:
            # Initialize driver on first use
            if not self._initialized:
                self._init_driver()
            
            if action == BrowserAction.NAVIGATE:
                response = await self._navigate(input_data)
            elif action == BrowserAction.CLICK:
                response = await self._click(input_data)
            elif action == BrowserAction.TYPE:
                response = await self._type(input_data)
            elif action == BrowserAction.SCROLL:
                response = await self._scroll(input_data)
            elif action == BrowserAction.SCREENSHOT:
                response = await self._screenshot(input_data)
            elif action == BrowserAction.GET_TEXT:
                response = await self._get_text(input_data)
            elif action == BrowserAction.GET_HTML:
                response = await self._get_html(input_data)
            elif action == BrowserAction.WAIT:
                response = await self._wait(input_data)
            elif action == BrowserAction.EXECUTE_JS:
                response = await self._execute_js(input_data)
            else:
                response = InteractiveResponse(
                    success=False,
                    action=action,
                    error=f"Unknown action: {action}",
                )
            
            end_time = datetime.utcnow()
            response.action_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return ToolResult(
                success=response.success,
                data=response,
                message=f"Browser action '{action.value}' {'succeeded' if response.success else 'failed'}",
                error=response.error,
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=InteractiveResponse(
                    success=False,
                    action=action,
                    error=str(e),
                ),
                message=f"Browser action failed: {str(e)}",
                error=str(e),
            )
    
    async def _navigate(self, input_data: Any) -> InteractiveResponse:
        """Navigate to a URL."""
        url = getattr(input_data, "url", str(input_data))
        
        self._driver.get(url)
        
        page_state = self._get_page_state()
        
        return InteractiveResponse(
            success=True,
            action=BrowserAction.NAVIGATE,
            page_state=page_state,
        )
    
    async def _click(self, input_data: Any) -> InteractiveResponse:
        """Click an element."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        selector = getattr(input_data, "selector", "")
        selector_type = getattr(input_data, "selector_type", "css")
        
        by_type = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "name": By.NAME,
            "class": By.CLASS_NAME,
            "tag": By.TAG_NAME,
        }.get(selector_type, By.CSS_SELECTOR)
        
        wait = WebDriverWait(self._driver, self.timeout)
        element = wait.until(EC.element_to_be_clickable((by_type, selector)))
        element.click()
        
        page_state = self._get_page_state()
        
        return InteractiveResponse(
            success=True,
            action=BrowserAction.CLICK,
            page_state=page_state,
        )
    
    async def _type(self, input_data: Any) -> InteractiveResponse:
        """Type text into an element."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        selector = getattr(input_data, "selector", "")
        text = getattr(input_data, "text", "")
        clear_first = getattr(input_data, "clear_first", True)
        selector_type = getattr(input_data, "selector_type", "css")
        
        by_type = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "name": By.NAME,
        }.get(selector_type, By.CSS_SELECTOR)
        
        wait = WebDriverWait(self._driver, self.timeout)
        element = wait.until(EC.presence_of_element_located((by_type, selector)))
        
        if clear_first:
            element.clear()
        
        element.send_keys(text)
        
        page_state = self._get_page_state()
        
        return InteractiveResponse(
            success=True,
            action=BrowserAction.TYPE,
            page_state=page_state,
        )
    
    async def _scroll(self, input_data: Any) -> InteractiveResponse:
        """Scroll the page."""
        direction = getattr(input_data, "direction", "down")
        amount = getattr(input_data, "amount", 500)
        
        if direction == "down":
            self._driver.execute_script(f"window.scrollBy(0, {amount});")
        elif direction == "up":
            self._driver.execute_script(f"window.scrollBy(0, -{amount});")
        elif direction == "top":
            self._driver.execute_script("window.scrollTo(0, 0);")
        elif direction == "bottom":
            self._driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        page_state = self._get_page_state()
        
        return InteractiveResponse(
            success=True,
            action=BrowserAction.SCROLL,
            page_state=page_state,
        )
    
    async def _screenshot(self, input_data: Any) -> InteractiveResponse:
        """Take a screenshot."""
        screenshot_bytes = self._driver.get_screenshot_as_png()
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        
        page_state = self._get_page_state()
        page_state.screenshot_base64 = screenshot_base64
        
        return InteractiveResponse(
            success=True,
            action=BrowserAction.SCREENSHOT,
            page_state=page_state,
        )
    
    async def _get_text(self, input_data: Any) -> InteractiveResponse:
        """Get text content from page or element."""
        selector = getattr(input_data, "selector", None)
        
        if selector:
            from selenium.webdriver.common.by import By
            element = self._driver.find_element(By.CSS_SELECTOR, selector)
            text = element.text
        else:
            text = self._driver.find_element("tag name", "body").text
        
        page_state = self._get_page_state()
        page_state.text = text
        
        return InteractiveResponse(
            success=True,
            action=BrowserAction.GET_TEXT,
            page_state=page_state,
        )
    
    async def _get_html(self, input_data: Any) -> InteractiveResponse:
        """Get HTML content from page or element."""
        selector = getattr(input_data, "selector", None)
        
        if selector:
            from selenium.webdriver.common.by import By
            element = self._driver.find_element(By.CSS_SELECTOR, selector)
            html = element.get_attribute("outerHTML")
        else:
            html = self._driver.page_source
        
        page_state = self._get_page_state()
        page_state.html = html
        
        return InteractiveResponse(
            success=True,
            action=BrowserAction.GET_HTML,
            page_state=page_state,
        )
    
    async def _wait(self, input_data: Any) -> InteractiveResponse:
        """Wait for an element or condition."""
        import time
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        selector = getattr(input_data, "selector", None)
        seconds = getattr(input_data, "seconds", 1)
        condition = getattr(input_data, "condition", "presence")
        
        if selector:
            wait = WebDriverWait(self._driver, self.timeout)
            
            if condition == "presence":
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            elif condition == "visible":
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
            elif condition == "clickable":
                wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
        else:
            time.sleep(seconds)
        
        page_state = self._get_page_state()
        
        return InteractiveResponse(
            success=True,
            action=BrowserAction.WAIT,
            page_state=page_state,
        )
    
    async def _execute_js(self, input_data: Any) -> InteractiveResponse:
        """Execute JavaScript on the page."""
        script = getattr(input_data, "script", "")
        
        self._driver.execute_script(script)
        
        page_state = self._get_page_state()
        
        return InteractiveResponse(
            success=True,
            action=BrowserAction.EXECUTE_JS,
            page_state=page_state,
        )
    
    def _get_page_state(self) -> PageState:
        """Get current page state."""
        return PageState(
            url=self._driver.current_url,
            title=self._driver.title,
        )
    
    def close(self) -> None:
        """Close the browser."""
        if self._driver:
            self._driver.quit()
            self._driver = None
            self._initialized = False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def create_interactive_tool(
    window_size: tuple[int, int] = (1920, 1080),
    timeout: int = 30,
) -> BrowserInteractiveTool:
    """Create an interactive browser tool."""
    return BrowserInteractiveTool(
        window_size=window_size,
        timeout=timeout,
    )
