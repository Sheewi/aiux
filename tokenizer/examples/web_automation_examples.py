#!/usr/bin/env python3
"""
Web Automation Agent example using Playwright + AX Tree.

This example demonstrates advanced web automation with accessibility tree support.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microagents.web_automation import WebAutomationAgent


async def accessibility_audit_example():
    """Example: Comprehensive accessibility audit of a webpage."""
    print("🔍 Accessibility Audit Example")
    print("-" * 40)
    
    async with WebAutomationAgent(headless=True) as agent:
        # Navigate to a test page
        await agent.navigate("https://www.w3.org/WAI/demos/bad/")
        
        # Get comprehensive page data with accessibility info
        page_data = await agent.extract_page_data()
        
        if page_data["success"]:
            print(f"Page: {page_data['page']['title']}")
            print(f"Interactive Elements Found:")
            print(f"  - Buttons: {len(page_data['elements']['buttons'])}")
            print(f"  - Links: {len(page_data['elements']['links'])}")
            print(f"  - Inputs: {len(page_data['elements']['inputs'])}")
            
            # Show some button details
            print("\nButton Details:")
            for button in page_data['elements']['buttons'][:3]:
                print(f"  - {button['name']} ({button['role']})")


async def form_automation_example():
    """Example: Automated form filling using accessibility selectors."""
    print("\n📝 Form Automation Example")
    print("-" * 40)
    
    async with WebAutomationAgent(headless=False) as agent:
        # Navigate to a form page
        await agent.navigate("https://httpbin.org/forms/post")
        
        # Fill form fields using accessibility attributes
        await agent.fill_accessible_input("Customer name", "John Doe")
        await agent.fill_accessible_input("Telephone", "555-1234")
        await agent.fill_accessible_input("Email address", "john@example.com")
        
        # Click submit button
        result = await agent.click_accessible_element("button", "Submit order")
        
        if result["success"]:
            print("✅ Form submitted successfully!")
        else:
            print(f"❌ Form submission failed: {result['error']}")


async def screenshot_comparison_example():
    """Example: Take screenshots for visual testing."""
    print("\n📸 Screenshot Comparison Example")
    print("-" * 40)
    
    async with WebAutomationAgent(headless=True) as agent:
        urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/forms/post"
        ]
        
        for i, url in enumerate(urls):
            await agent.navigate(url)
            result = await agent.screenshot(f"screenshot_{i+1}.png")
            
            if result["success"]:
                print(f"✅ Screenshot saved: {result['path']}")


if __name__ == "__main__":
    async def main():
        print("🌐 Web Automation Agent Examples")
        print("=" * 50)
        
        try:
            await accessibility_audit_example()
            await form_automation_example()
            await screenshot_comparison_example()
            
            print("\n🎉 All web automation examples completed!")
            
        except Exception as e:
            print(f"❌ Example failed: {e}")
    
    asyncio.run(main())
