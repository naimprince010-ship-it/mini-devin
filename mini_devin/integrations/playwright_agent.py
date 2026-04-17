"""
Enhanced Playwright Agent for Plodder
Full web automation capabilities with intelligent interaction
"""

import asyncio
import json
import base64
from typing import Optional, List, Dict, Any, Union
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Locator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PlaywrightAgent:
    """Enhanced web automation agent with intelligent interaction capabilities"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.screenshots_dir = Path("./screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
    async def start(self, browser_type: str = "chromium") -> bool:
        """Start the browser and create context"""
        try:
            self.playwright = await async_playwright().start()
            
            if browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(headless=self.headless)
            elif browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(headless=self.headless)
            elif browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(headless=self.headless)
            else:
                raise ValueError(f"Unsupported browser type: {browser_type}")
            
            # Create context with enhanced settings
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                ignore_https_errors=True,
                permissions=["geolocation", "notifications"]
            )
            
            # Create page
            self.page = await self.context.new_page()
            
            # Set up console logging
            self.page.on("console", lambda msg: logger.info(f"Browser console: {msg.text}"))
            self.page.on("pageerror", lambda err: logger.error(f"Page error: {err}"))
            
            logger.info(f"Started {browser_type} browser (headless={self.headless})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            return False
    
    async def navigate_to(self, url: str) -> bool:
        """Navigate to a URL"""
        try:
            await self.page.goto(url, wait_until="networkidle")
            logger.info(f"Navigated to: {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return False
    
    async def take_screenshot(self, filename: Optional[str] = None) -> str:
        """Take a screenshot and return the path"""
        try:
            if not filename:
                timestamp = int(asyncio.get_event_loop().time())
                filename = f"screenshot_{timestamp}.png"
            
            screenshot_path = self.screenshots_dir / filename
            await self.page.screenshot(path=str(screenshot_path), full_page=True)
            
            logger.info(f"Screenshot saved: {screenshot_path}")
            return str(screenshot_path)
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return ""
    
    async def click_element(self, selector: str, wait_for: bool = True) -> bool:
        """Click an element by selector"""
        try:
            if wait_for:
                await self.page.wait_for_selector(selector, timeout=10000)
            
            await self.page.click(selector)
            logger.info(f"Clicked element: {selector}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to click element {selector}: {e}")
            return False
    
    async def type_text(self, selector: str, text: str, clear_first: bool = True) -> bool:
        """Type text into an element"""
        try:
            element = await self.page.wait_for_selector(selector, timeout=10000)
            
            if clear_first:
                await element.clear()
            
            await element.fill(text)
            logger.info(f"Typed text into {selector}: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to type text into {selector}: {e}")
            return False
    
    async def get_text(self, selector: str) -> str:
        """Get text content of an element"""
        try:
            element = await self.page.wait_for_selector(selector, timeout=10000)
            text = await element.text_content()
            logger.info(f"Retrieved text from {selector}: {text[:100]}...")
            return text or ""
            
        except Exception as e:
            logger.error(f"Failed to get text from {selector}: {e}")
            return ""
    
    async def get_attribute(self, selector: str, attribute: str) -> str:
        """Get attribute value of an element"""
        try:
            element = await self.page.wait_for_selector(selector, timeout=10000)
            value = await element.get_attribute(attribute)
            logger.info(f"Retrieved attribute {attribute} from {selector}: {value}")
            return value or ""
            
        except Exception as e:
            logger.error(f"Failed to get attribute {attribute} from {selector}: {e}")
            return ""
    
    async def wait_for_element(self, selector: str, timeout: int = 10000) -> bool:
        """Wait for an element to appear"""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            logger.info(f"Element found: {selector}")
            return True
            
        except Exception as e:
            logger.error(f"Element not found within timeout: {selector}")
            return False
    
    async def wait_for_navigation(self, timeout: int = 10000) -> bool:
        """Wait for navigation to complete"""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
            logger.info("Navigation completed")
            return True
            
        except Exception as e:
            logger.error(f"Navigation timeout: {e}")
            return False
    
    async def execute_javascript(self, script: str) -> Any:
        """Execute JavaScript in the page"""
        try:
            result = await self.page.evaluate(script)
            logger.info(f"Executed JavaScript: {script[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute JavaScript: {e}")
            return None
    
    async def scroll_to_element(self, selector: str) -> bool:
        """Scroll to an element"""
        try:
            element = await self.page.wait_for_selector(selector, timeout=10000)
            await element.scroll_into_view_if_needed()
            logger.info(f"Scrolled to element: {selector}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scroll to element {selector}: {e}")
            return False
    
    async def select_dropdown(self, selector: str, value: str) -> bool:
        """Select a dropdown option"""
        try:
            await self.page.select_option(selector, value)
            logger.info(f"Selected {value} from dropdown {selector}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to select from dropdown {selector}: {e}")
            return False
    
    async def upload_file(self, selector: str, file_path: str) -> bool:
        """Upload a file"""
        try:
            await self.page.set_input_files(selector, file_path)
            logger.info(f"Uploaded file {file_path} to {selector}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file to {selector}: {e}")
            return False
    
    async def handle_alert(self, accept: bool = True) -> bool:
        """Handle browser alert"""
        try:
            if accept:
                await self.page.on("dialog", lambda dialog: dialog.accept())
            else:
                await self.page.on("dialog", lambda dialog: dialog.dismiss())
            logger.info(f"Handled alert (accept={accept})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle alert: {e}")
            return False
    
    async def get_page_title(self) -> str:
        """Get the page title"""
        try:
            title = await self.page.title()
            logger.info(f"Page title: {title}")
            return title
            
        except Exception as e:
            logger.error(f"Failed to get page title: {e}")
            return ""
    
    async def get_current_url(self) -> str:
        """Get the current URL"""
        try:
            url = self.page.url
            logger.info(f"Current URL: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to get current URL: {e}")
            return ""
    
    async def switch_to_tab(self, index: int) -> bool:
        """Switch to a different tab"""
        try:
            pages = self.context.pages
            if 0 <= index < len(pages):
                await pages[index].bring_to_front()
                self.page = pages[index]
                logger.info(f"Switched to tab {index}")
                return True
            else:
                logger.error(f"Tab index {index} out of range")
                return False
                
        except Exception as e:
            logger.error(f"Failed to switch to tab {index}: {e}")
            return False
    
    async def open_new_tab(self, url: Optional[str] = None) -> bool:
        """Open a new tab"""
        try:
            new_page = await self.context.new_page()
            if url:
                await new_page.goto(url)
            self.page = new_page
            logger.info(f"Opened new tab: {url or 'about:blank'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open new tab: {e}")
            return False
    
    async def close_tab(self) -> bool:
        """Close current tab"""
        try:
            await self.page.close()
            # Switch to first available tab
            pages = self.context.pages
            if pages:
                self.page = pages[0]
                await self.page.bring_to_front()
            logger.info("Closed current tab")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close tab: {e}")
            return False
    
    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies"""
        try:
            cookies = await self.context.cookies()
            logger.info(f"Retrieved {len(cookies)} cookies")
            return cookies
            
        except Exception as e:
            logger.error(f"Failed to get cookies: {e}")
            return []
    
    async def set_cookies(self, cookies: List[Dict[str, Any]]) -> bool:
        """Set cookies"""
        try:
            await self.context.add_cookies(cookies)
            logger.info(f"Set {len(cookies)} cookies")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cookies: {e}")
            return False
    
    async def wait_and_click_with_retry(self, selector: str, max_retries: int = 3) -> bool:
        """Click element with retry logic"""
        for attempt in range(max_retries):
            try:
                await self.page.wait_for_selector(selector, timeout=5000)
                await self.page.click(selector)
                logger.info(f"Successfully clicked {selector} on attempt {attempt + 1}")
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {selector}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Failed to click {selector} after {max_retries} attempts")
                    return False
    
    async def smart_fill_form(self, form_data: Dict[str, str]) -> bool:
        """Smart form filling with multiple input types"""
        try:
            for selector, value in form_data.items():
                element = await self.page.wait_for_selector(selector, timeout=5000)
                tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                input_type = await element.get_attribute("type")
                
                if tag_name == "select":
                    await self.page.select_option(selector, value)
                elif tag_name == "input" and input_type in ["checkbox", "radio"]:
                    if value.lower() in ["true", "yes", "1", "checked"]:
                        await element.check()
                    else:
                        await element.uncheck()
                elif tag_name == "textarea" or (tag_name == "input" and input_type != "file"):
                    await element.fill(value)
                else:
                    logger.warning(f"Unsupported element type for {selector}: {tag_name}")
            
            logger.info("Successfully filled form")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fill form: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the browser and cleanup"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Browser stopped and cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Example usage and automation workflows
async def automated_login_workflow(
    agent: PlaywrightAgent,
    login_url: str,
    username_selector: str,
    password_selector: str,
    submit_selector: str,
    username: str,
    password: str
) -> bool:
    """Automated login workflow"""
    
    # Navigate to login page
    if not await agent.navigate_to(login_url):
        return False
    
    # Fill login form
    if not await agent.type_text(username_selector, username):
        return False
    
    if not await agent.type_text(password_selector, password):
        return False
    
    # Submit form
    if not await agent.click_element(submit_selector):
        return False
    
    # Wait for navigation
    await agent.wait_for_navigation()
    
    # Take screenshot for verification
    await agent.take_screenshot("login_result.png")
    
    return True

async def web_scraping_workflow(
    agent: PlaywrightAgent,
    url: str,
    data_selectors: Dict[str, str]
) -> Dict[str, str]:
    """Web scraping workflow"""
    
    # Navigate to page
    if not await agent.navigate_to(url):
        return {}
    
    # Extract data
    data = {}
    for key, selector in data_selectors.items():
        text = await agent.get_text(selector)
        data[key] = text
    
    # Take screenshot
    await agent.take_screenshot(f"scraping_{url.replace('https://', '').replace('/', '_')}.png")
    
    return data
