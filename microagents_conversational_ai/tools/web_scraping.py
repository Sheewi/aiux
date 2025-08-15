"""
Web Scraping Tools - Advanced web content extraction and parsing
Provides intelligent web scraping with respect for robots.txt and rate limiting.
"""

import asyncio
import aiohttp
import logging
import re
import time
import urllib.robotparser
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin, urlparse
from datetime import datetime
import json

from .base_tool import (
    BaseTool, ToolMetadata, ToolRequest, ToolResponse,
    ToolType, ToolStatus, ToolCapability, create_tool_metadata
)

logger = logging.getLogger(__name__)

class WebScrapingTool(BaseTool):
    """Advanced web scraping tool with intelligent content extraction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        metadata = create_tool_metadata(
            tool_id="web_scraper",
            name="Web Scraping Tool",
            description="Intelligent web content extraction with support for various formats",
            tool_type=ToolType.WEB_SCRAPING,
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.BATCH_PROCESSING,
                ToolCapability.RATE_LIMITED,
                ToolCapability.NETWORK_DEPENDENT
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "urls": {"type": "array", "items": {"type": "string"}, "description": "Multiple URLs for batch processing"},
                    "selectors": {
                        "type": "object",
                        "description": "CSS selectors for specific content extraction",
                        "properties": {
                            "title": {"type": "string", "default": "title, h1"},
                            "content": {"type": "string", "default": "p, article, .content"},
                            "links": {"type": "string", "default": "a[href]"},
                            "images": {"type": "string", "default": "img[src]"}
                        }
                    },
                    "extract_type": {
                        "type": "string",
                        "enum": ["text", "html", "links", "images", "metadata", "all"],
                        "default": "text",
                        "description": "Type of content to extract"
                    },
                    "follow_redirects": {"type": "boolean", "default": True},
                    "respect_robots": {"type": "boolean", "default": True},
                    "timeout": {"type": "number", "default": 30},
                    "headers": {"type": "object", "description": "Custom HTTP headers"},
                    "clean_text": {"type": "boolean", "default": True, "description": "Clean and normalize text"}
                },
                "anyOf": [
                    {"required": ["url"]},
                    {"required": ["urls"]}
                ]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "links": {"type": "array"},
                    "images": {"type": "array"},
                    "metadata": {"type": "object"},
                    "status_code": {"type": "integer"},
                    "content_type": {"type": "string"},
                    "scraped_at": {"type": "string"}
                }
            },
            rate_limit=30,  # 30 requests per minute
            timeout=60.0,
            supported_formats=["html", "xml", "json"],
            tags=["scraping", "web", "extraction", "content", "html"]
        )
        
        super().__init__(metadata, config)
        
        # Default configuration
        self.default_headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; WebScrapingBot/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        self.robots_cache = {}  # Cache for robots.txt files
    
    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute web scraping request."""
        try:
            # Handle single URL or multiple URLs
            if "url" in request.parameters:
                result = await self._scrape_single_url(request)
            elif "urls" in request.parameters:
                result = await self._scrape_multiple_urls(request)
            else:
                return ToolResponse(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    status=ToolStatus.FAILED,
                    error="Either 'url' or 'urls' parameter is required"
                )
            
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.COMPLETED,
                result=result,
                metadata={
                    "scraping_timestamp": datetime.now().isoformat(),
                    "urls_processed": 1 if "url" in request.parameters else len(request.parameters.get("urls", []))
                }
            )
            
        except Exception as e:
            logger.error(f"Web scraping failed: {e}")
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _scrape_single_url(self, request: ToolRequest) -> Dict[str, Any]:
        """Scrape a single URL."""
        url = request.parameters["url"]
        extract_type = request.parameters.get("extract_type", "text")
        selectors = request.parameters.get("selectors", {})
        respect_robots = request.parameters.get("respect_robots", True)
        timeout = request.parameters.get("timeout", 30)
        custom_headers = request.parameters.get("headers", {})
        clean_text = request.parameters.get("clean_text", True)
        
        # Check robots.txt if required
        if respect_robots and not await self._can_fetch(url):
            raise Exception(f"Robots.txt disallows scraping {url}")
        
        # Prepare headers
        headers = {**self.default_headers, **custom_headers}
        
        # Fetch content
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                content = await response.text()
                content_type = response.headers.get('content-type', '')
        
        # Parse and extract content
        extracted_data = await self._extract_content(
            content, url, extract_type, selectors, clean_text
        )
        
        # Add response metadata
        extracted_data.update({
            "url": url,
            "status_code": response.status,
            "content_type": content_type,
            "scraped_at": datetime.now().isoformat()
        })
        
        return extracted_data
    
    async def _scrape_multiple_urls(self, request: ToolRequest) -> List[Dict[str, Any]]:
        """Scrape multiple URLs with batch processing."""
        urls = request.parameters["urls"]
        max_concurrent = min(5, len(urls))  # Limit concurrent requests
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str):
            async with semaphore:
                try:
                    # Create individual request for each URL
                    individual_request = ToolRequest(
                        request_id=f"{request.request_id}_{url}",
                        tool_id=request.tool_id,
                        action=request.action,
                        parameters={**request.parameters, "url": url}
                    )
                    # Remove 'urls' to avoid confusion
                    individual_request.parameters.pop('urls', None)
                    
                    result = await self._scrape_single_url(individual_request)
                    return result
                except Exception as e:
                    return {
                        "url": url,
                        "error": str(e),
                        "status_code": 0,
                        "scraped_at": datetime.now().isoformat()
                    }
        
        # Process all URLs concurrently
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _extract_content(self, html_content: str, base_url: str, 
                             extract_type: str, selectors: Dict[str, str],
                             clean_text: bool) -> Dict[str, Any]:
        """Extract specific content from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback to simple regex-based extraction
            return await self._extract_content_simple(html_content, extract_type, clean_text)
        
        soup = BeautifulSoup(html_content, 'html.parser')
        extracted = {}
        
        if extract_type in ["text", "all"]:
            # Extract title
            title_selector = selectors.get("title", "title, h1")
            title_elem = soup.select_one(title_selector)
            extracted["title"] = title_elem.get_text(strip=True) if title_elem else ""
            
            # Extract main content
            content_selector = selectors.get("content", "p, article, .content, main")
            content_elems = soup.select(content_selector)
            content_text = " ".join([elem.get_text(strip=True) for elem in content_elems])
            
            if clean_text:
                content_text = self._clean_text(content_text)
            
            extracted["content"] = content_text
        
        if extract_type in ["links", "all"]:
            # Extract links
            link_selector = selectors.get("links", "a[href]")
            link_elems = soup.select(link_selector)
            links = []
            for link in link_elems:
                href = link.get('href')
                if href:
                    absolute_url = urljoin(base_url, href)
                    links.append({
                        "url": absolute_url,
                        "text": link.get_text(strip=True),
                        "title": link.get('title', '')
                    })
            extracted["links"] = links
        
        if extract_type in ["images", "all"]:
            # Extract images
            img_selector = selectors.get("images", "img[src]")
            img_elems = soup.select(img_selector)
            images = []
            for img in img_elems:
                src = img.get('src')
                if src:
                    absolute_url = urljoin(base_url, src)
                    images.append({
                        "url": absolute_url,
                        "alt": img.get('alt', ''),
                        "title": img.get('title', '')
                    })
            extracted["images"] = images
        
        if extract_type in ["metadata", "all"]:
            # Extract metadata
            metadata = {}
            
            # Meta tags
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
                content = meta.get('content')
                if name and content:
                    metadata[name] = content
            
            # OpenGraph and Twitter Card data
            og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
            for tag in og_tags:
                prop = tag.get('property')
                content = tag.get('content')
                if prop and content:
                    metadata[prop] = content
            
            extracted["metadata"] = metadata
        
        if extract_type == "html":
            extracted["html"] = html_content
        
        return extracted
    
    async def _extract_content_simple(self, html_content: str, extract_type: str, 
                                    clean_text: bool) -> Dict[str, Any]:
        """Simple regex-based content extraction (fallback)."""
        extracted = {}
        
        if extract_type in ["text", "all"]:
            # Extract title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
            extracted["title"] = title_match.group(1) if title_match else ""
            
            # Extract text content (remove HTML tags)
            text_content = re.sub(r'<[^>]+>', ' ', html_content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            if clean_text:
                text_content = self._clean_text(text_content)
            
            extracted["content"] = text_content
        
        if extract_type in ["links", "all"]:
            # Extract links
            link_pattern = r'<a[^>]*href=["\'](.*?)["\'][^>]*>(.*?)</a>'
            links = []
            for match in re.finditer(link_pattern, html_content, re.IGNORECASE | re.DOTALL):
                url = match.group(1)
                text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                links.append({"url": url, "text": text})
            extracted["links"] = links
        
        if extract_type in ["images", "all"]:
            # Extract images
            img_pattern = r'<img[^>]*src=["\'](.*?)["\'][^>]*>'
            images = []
            for match in re.finditer(img_pattern, html_content, re.IGNORECASE):
                images.append({"url": match.group(1), "alt": "", "title": ""})
            extracted["images"] = images
        
        if extract_type == "html":
            extracted["html"] = html_content
        
        return extracted
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common web artifacts
        text = re.sub(r'JavaScript is disabled.*?enable JavaScript', '', text, flags=re.IGNORECASE)
        text = re.sub(r'This site requires JavaScript', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Skip to (main )?content', '', text, flags=re.IGNORECASE)
        
        # Remove navigation artifacts
        text = re.sub(r'\b(Home|About|Contact|Privacy|Terms)\b(?:\s+\|)?', '', text)
        
        return text.strip()
    
    async def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            # Check cache first
            if robots_url in self.robots_cache:
                rp = self.robots_cache[robots_url]
            else:
                # Fetch and parse robots.txt
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                
                try:
                    # Use aiohttp to fetch robots.txt
                    async with aiohttp.ClientSession() as session:
                        async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                robots_content = await response.text()
                                # Unfortunately, robotparser doesn't support async, so we'll use a simple approach
                                return not self._is_disallowed_by_robots(robots_content, url)
                            else:
                                # If robots.txt is not found, assume allowed
                                return True
                except:
                    # If we can't fetch robots.txt, assume allowed
                    return True
                
                self.robots_cache[robots_url] = rp
            
            return rp.can_fetch(self.default_headers['User-Agent'], url)
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True  # Default to allow if check fails
    
    def _is_disallowed_by_robots(self, robots_content: str, url: str) -> bool:
        """Simple robots.txt check."""
        lines = robots_content.lower().split('\n')
        user_agent_applies = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('user-agent:'):
                agent = line.split(':', 1)[1].strip()
                user_agent_applies = (agent == '*' or 'bot' in agent)
            elif user_agent_applies and line.startswith('disallow:'):
                disallowed_path = line.split(':', 1)[1].strip()
                if disallowed_path and urlparse(url).path.startswith(disallowed_path):
                    return True
        
        return False

class SitemapExtractorTool(BaseTool):
    """Tool for extracting and parsing website sitemaps."""
    
    def __init__(self, config: Dict[str, Any] = None):
        metadata = create_tool_metadata(
            tool_id="sitemap_extractor",
            name="Sitemap Extractor",
            description="Extract and parse XML sitemaps to discover website structure",
            tool_type=ToolType.WEB_SCRAPING,
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.NETWORK_DEPENDENT
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Website URL or direct sitemap URL"},
                    "discover_sitemaps": {"type": "boolean", "default": True, "description": "Auto-discover sitemaps"},
                    "max_urls": {"type": "integer", "default": 1000, "description": "Maximum URLs to extract"}
                },
                "required": ["url"]
            },
            tags=["sitemap", "xml", "discovery", "urls"]
        )
        
        super().__init__(metadata, config)
    
    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute sitemap extraction."""
        try:
            url = request.parameters["url"]
            discover_sitemaps = request.parameters.get("discover_sitemaps", True)
            max_urls = request.parameters.get("max_urls", 1000)
            
            sitemaps = []
            all_urls = []
            
            if discover_sitemaps:
                # Try to find sitemaps
                discovered_sitemaps = await self._discover_sitemaps(url)
                sitemaps.extend(discovered_sitemaps)
            else:
                # Treat URL as direct sitemap
                sitemaps.append(url)
            
            # Extract URLs from all sitemaps
            for sitemap_url in sitemaps:
                try:
                    urls = await self._extract_from_sitemap(sitemap_url, max_urls - len(all_urls))
                    all_urls.extend(urls)
                    
                    if len(all_urls) >= max_urls:
                        break
                except Exception as e:
                    logger.warning(f"Failed to extract from sitemap {sitemap_url}: {e}")
            
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.COMPLETED,
                result={
                    "sitemaps_found": sitemaps,
                    "urls_extracted": all_urls[:max_urls],
                    "total_urls": len(all_urls),
                    "extraction_timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _discover_sitemaps(self, base_url: str) -> List[str]:
        """Discover sitemap URLs from a website."""
        sitemaps = []
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Common sitemap locations
        common_paths = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemaps.xml",
            "/sitemap1.xml"
        ]
        
        async with aiohttp.ClientSession() as session:
            # Check common sitemap locations
            for path in common_paths:
                sitemap_url = base_domain + path
                try:
                    async with session.head(sitemap_url) as response:
                        if response.status == 200:
                            sitemaps.append(sitemap_url)
                except:
                    continue
            
            # Check robots.txt for sitemap declarations
            try:
                robots_url = base_domain + "/robots.txt"
                async with session.get(robots_url) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        for line in robots_content.split('\n'):
                            if line.lower().startswith('sitemap:'):
                                sitemap_url = line.split(':', 1)[1].strip()
                                if sitemap_url not in sitemaps:
                                    sitemaps.append(sitemap_url)
            except:
                pass
        
        return sitemaps
    
    async def _extract_from_sitemap(self, sitemap_url: str, max_urls: int) -> List[Dict[str, Any]]:
        """Extract URLs from a sitemap XML."""
        urls = []
        
        async with aiohttp.ClientSession() as session:
            async with session.get(sitemap_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch sitemap: HTTP {response.status}")
                
                content = await response.text()
        
        # Parse XML (simple regex approach for now)
        # In production, use xml.etree.ElementTree or lxml
        import re
        
        # Extract regular URLs
        url_pattern = r'<loc>(.*?)</loc>'
        lastmod_pattern = r'<lastmod>(.*?)</lastmod>'
        
        url_matches = re.findall(url_pattern, content)
        lastmod_matches = re.findall(lastmod_pattern, content)
        
        for i, url in enumerate(url_matches[:max_urls]):
            lastmod = lastmod_matches[i] if i < len(lastmod_matches) else None
            urls.append({
                "url": url.strip(),
                "lastmod": lastmod,
                "sitemap_source": sitemap_url
            })
        
        # Check for sitemap index (nested sitemaps)
        if '<sitemapindex' in content:
            sitemap_pattern = r'<loc>(.*?\.xml.*?)</loc>'
            nested_sitemaps = re.findall(sitemap_pattern, content)
            
            for nested_sitemap in nested_sitemaps:
                if len(urls) >= max_urls:
                    break
                try:
                    nested_urls = await self._extract_from_sitemap(
                        nested_sitemap.strip(), 
                        max_urls - len(urls)
                    )
                    urls.extend(nested_urls)
                except Exception as e:
                    logger.warning(f"Failed to extract from nested sitemap {nested_sitemap}: {e}")
        
        return urls

# Factory functions
def create_web_scraping_tool(**config) -> WebScrapingTool:
    """Create a web scraping tool instance."""
    return WebScrapingTool(config=config)

def create_sitemap_extractor_tool(**config) -> SitemapExtractorTool:
    """Create a sitemap extractor tool instance."""
    return SitemapExtractorTool(config=config)

# Example usage and testing
if __name__ == "__main__":
    async def demo():
        print("Web Scraping Tools Demo")
        print("=" * 40)
        
        # Create tools
        scraper = create_web_scraping_tool()
        sitemap_extractor = create_sitemap_extractor_tool()
        
        # Test web scraping
        scrape_request = ToolRequest(
            request_id="test_scrape_1",
            tool_id="web_scraper",
            action="scrape",
            parameters={
                "url": "https://httpbin.org/html",  # Test URL that returns HTML
                "extract_type": "all",
                "clean_text": True
            }
        )
        
        print("🕷️  Testing web scraping...")
        scrape_response = await scraper.execute(scrape_request)
        print(f"Status: {scrape_response.status.value}")
        
        if scrape_response.status == ToolStatus.COMPLETED:
            result = scrape_response.result
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Content length: {len(result.get('content', ''))}")
            print(f"Links found: {len(result.get('links', []))}")
            print(f"Images found: {len(result.get('images', []))}")
        
        # Test sitemap extraction
        sitemap_request = ToolRequest(
            request_id="test_sitemap_1",
            tool_id="sitemap_extractor",
            action="extract",
            parameters={
                "url": "https://example.com",
                "discover_sitemaps": True,
                "max_urls": 10
            }
        )
        
        print(f"\n🗺️  Testing sitemap extraction...")
        sitemap_response = await sitemap_extractor.execute(sitemap_request)
        print(f"Status: {sitemap_response.status.value}")
        
        if sitemap_response.status == ToolStatus.COMPLETED:
            result = sitemap_response.result
            print(f"Sitemaps found: {len(result.get('sitemaps_found', []))}")
            print(f"URLs extracted: {len(result.get('urls_extracted', []))}")
        
        print(f"\n✅ Web Scraping Tools Demo completed!")
    
    asyncio.run(demo())
