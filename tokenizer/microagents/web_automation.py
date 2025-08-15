"""
Web Automation Agent using Playwright + AX Tree

This microagent specializes in browser automation with accessibility tree access
for robust web interaction and testing.
"""

import asyncio
from typing import Dict, List, Optional, Any
from playwright.async_api import async_playwright, Page, Browser
import json
import logging

logger = logging.getLogger(__name__)


class WebAutomationAgent:
    """
    Web automation agent with Playwright and accessibility tree support.
    
    Features:
    - Browser automation across Chrome, Firefox, Safari
    - Accessibility tree navigation
    - Screenshot and PDF generation
    - Mobile device emulation
    - Network interception
    """
    
    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        self.headless = headless
        self.browser_type = browser_type
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        
    async def start(self):
        """Initialize the browser session."""
        try:
            self.playwright = await async_playwright().start()
            
            if self.browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(headless=self.headless)
            elif self.browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(headless=self.headless)
            elif self.browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(headless=self.headless)
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")
                
            self.page = await self.browser.new_page()
            logger.info(f"Started {self.browser_type} browser")
            
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise
            
    async def stop(self):
        """Clean up browser resources."""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Browser session closed")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
            
    async def navigate(self, url: str, wait_until: str = "networkidle") -> Dict[str, Any]:
        """
        Navigate to a URL and return page info.
        
        Args:
            url: Target URL
            wait_until: When to consider navigation finished
            
        Returns:
            Dict with page title, URL, and status
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
            
        try:
            response = await self.page.goto(url, wait_until=wait_until)
            title = await self.page.title()
            
            return {
                "url": self.page.url,
                "title": title,
                "status": response.status if response else None,
                "success": True
            }
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def get_accessibility_tree(self) -> Dict[str, Any]:
        """
        Get the accessibility tree of the current page.
        
        Returns:
            Dict containing the accessibility tree structure
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
            
        try:
            # Get accessibility tree snapshot
            snapshot = await self.page.accessibility.snapshot()
            return {
                "success": True,
                "tree": snapshot,
                "url": self.page.url
            }
        except Exception as e:
            logger.error(f"Failed to get accessibility tree: {e}")
            return {"success": False, "error": str(e)}
            
    async def find_accessible_elements(self, role: str = None, name: str = None) -> List[Dict]:
        """
        Find elements by accessibility attributes.
        
        Args:
            role: ARIA role (button, link, textbox, etc.)
            name: Accessible name
            
        Returns:
            List of matching elements with their properties
        """
        tree = await self.get_accessibility_tree()
        if not tree["success"]:
            return []
            
        def search_tree(node, matches):
            if isinstance(node, dict):
                node_role = node.get("role", "")
                node_name = node.get("name", "")
                
                role_match = not role or role.lower() in node_role.lower()
                name_match = not name or name.lower() in node_name.lower()
                
                if role_match and name_match:
                    matches.append({
                        "role": node_role,
                        "name": node_name,
                        "description": node.get("description", ""),
                        "value": node.get("value", "")
                    })
                    
                # Recursively search children
                for child in node.get("children", []):
                    search_tree(child, matches)
                    
        matches = []
        if tree.get("tree"):
            search_tree(tree["tree"], matches)
        return matches
        
    async def click_accessible_element(self, role: str, name: str) -> Dict[str, Any]:
        """
        Click an element identified by accessibility attributes.
        
        Args:
            role: ARIA role
            name: Accessible name
            
        Returns:
            Result of the click operation
        """
        try:
            # Use Playwright's accessibility-aware selectors
            selector = f'role={role}[name="{name}"]'
            await self.page.click(selector, timeout=10000)
            
            return {
                "success": True,
                "action": "click",
                "target": f"{role} with name '{name}'"
            }
        except Exception as e:
            logger.error(f"Failed to click accessible element: {e}")
            return {"success": False, "error": str(e)}
            
    async def fill_accessible_input(self, name: str, value: str) -> Dict[str, Any]:
        """
        Fill an input field identified by accessible name.
        
        Args:
            name: Accessible name of the input
            value: Value to fill
            
        Returns:
            Result of the fill operation
        """
        try:
            selector = f'role=textbox[name="{name}"]'
            await self.page.fill(selector, value, timeout=10000)
            
            return {
                "success": True,
                "action": "fill",
                "target": f"textbox with name '{name}'",
                "value": value
            }
        except Exception as e:
            logger.error(f"Failed to fill accessible input: {e}")
            return {"success": False, "error": str(e)}
            
    async def screenshot(self, path: str = None, full_page: bool = True) -> Dict[str, Any]:
        """
        Take a screenshot of the current page.
        
        Args:
            path: File path to save screenshot
            full_page: Whether to capture full page
            
        Returns:
            Screenshot result with path or base64 data
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
            
        try:
            if path:
                await self.page.screenshot(path=path, full_page=full_page)
                return {"success": True, "path": path}
            else:
                screenshot_bytes = await self.page.screenshot(full_page=full_page)
                import base64
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                return {"success": True, "data": screenshot_b64}
                
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def extract_page_data(self) -> Dict[str, Any]:
        """
        Extract comprehensive page data including accessibility info.
        
        Returns:
            Dict with page content, metadata, and accessibility tree
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
            
        try:
            # Get basic page info
            title = await self.page.title()
            url = self.page.url
            content = await self.page.content()
            
            # Get accessibility tree
            ax_tree = await self.get_accessibility_tree()
            
            # Get all interactive elements
            buttons = await self.find_accessible_elements(role="button")
            links = await self.find_accessible_elements(role="link")
            inputs = await self.find_accessible_elements(role="textbox")
            
            return {
                "success": True,
                "page": {
                    "title": title,
                    "url": url,
                    "content_length": len(content)
                },
                "accessibility": {
                    "tree": ax_tree.get("tree"),
                    "interactive_elements": {
                        "buttons": len(buttons),
                        "links": len(links),
                        "inputs": len(inputs)
                    }
                },
                "elements": {
                    "buttons": buttons[:10],  # Limit to first 10
                    "links": links[:10],
                    "inputs": inputs[:10]
                }
            }
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {"success": False, "error": str(e)}


# Convenience functions for quick usage
async def quick_screenshot(url: str, output_path: str = "screenshot.png") -> bool:
    """Take a quick screenshot of a URL."""
    async with WebAutomationAgent() as agent:
        await agent.navigate(url)
        result = await agent.screenshot(output_path)
        return result["success"]


async def quick_accessibility_audit(url: str) -> Dict[str, Any]:
    """Quick accessibility audit of a URL."""
    async with WebAutomationAgent() as agent:
        await agent.navigate(url)
        return await agent.extract_page_data()


if __name__ == "__main__":
    # Example usage
    async def main():
        async with WebAutomationAgent(headless=False) as agent:
            # Navigate to a page
            result = await agent.navigate("https://example.com")
            print(f"Navigation: {result}")
            
            # Get accessibility tree
            ax_tree = await agent.get_accessibility_tree()
            print(f"Accessibility tree nodes: {len(str(ax_tree))}")
            
            # Find buttons
            buttons = await agent.find_accessible_elements(role="button")
            print(f"Found {len(buttons)} buttons")
            
            # Extract comprehensive data
            data = await agent.extract_page_data()
            print(f"Page data extracted: {data['success']}")
            
    asyncio.run(main())
