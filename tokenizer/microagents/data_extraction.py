"""
Data Extraction Agent using Scrapy + Diffbot

This microagent specializes in web scraping and structured data extraction
with intelligent content parsing.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from twisted.internet import defer
import re

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of a data extraction operation."""
    url: str
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None


class DiffbotClient:
    """Client for Diffbot API integration."""
    
    def __init__(self, api_token: str = None):
        self.api_token = api_token
        self.base_url = "https://api.diffbot.com/v3"
        
    async def extract_article(self, url: str) -> Dict[str, Any]:
        """Extract article content using Diffbot Article API."""
        if not self.api_token:
            return {"success": False, "error": "Diffbot API token not provided"}
            
        endpoint = f"{self.base_url}/article"
        params = {
            "token": self.api_token,
            "url": url,
            "format": "json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "data": data,
                            "source": "diffbot_article"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Diffbot API error: {response.status}"
                        }
        except Exception as e:
            logger.error(f"Diffbot article extraction failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def extract_product(self, url: str) -> Dict[str, Any]:
        """Extract product information using Diffbot Product API."""
        if not self.api_token:
            return {"success": False, "error": "Diffbot API token not provided"}
            
        endpoint = f"{self.base_url}/product"
        params = {
            "token": self.api_token,
            "url": url,
            "format": "json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "data": data,
                            "source": "diffbot_product"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Diffbot API error: {response.status}"
                        }
        except Exception as e:
            logger.error(f"Diffbot product extraction failed: {e}")
            return {"success": False, "error": str(e)}


class SmartSpider(scrapy.Spider):
    """Enhanced Scrapy spider with intelligent extraction."""
    
    name = "smart_spider"
    
    def __init__(self, start_urls=None, extraction_rules=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls or []
        self.extraction_rules = extraction_rules or {}
        self.results = []
        
    def parse(self, response):
        """Parse response using extraction rules."""
        try:
            extracted_data = {}
            
            # Apply extraction rules
            for field, rule in self.extraction_rules.items():
                if isinstance(rule, dict):
                    selector = rule.get("selector")
                    attribute = rule.get("attribute", "text")
                    multiple = rule.get("multiple", False)
                    
                    if selector:
                        if multiple:
                            if attribute == "text":
                                extracted_data[field] = response.css(selector).getall()
                            else:
                                extracted_data[field] = response.css(selector).attrib.getall(attribute)
                        else:
                            if attribute == "text":
                                extracted_data[field] = response.css(selector).get()
                            else:
                                element = response.css(selector).get()
                                if element:
                                    extracted_data[field] = response.css(selector).attrib.get(attribute)
                                    
            # Auto-extract common elements if no rules provided
            if not self.extraction_rules:
                extracted_data = self._auto_extract(response)
                
            result = ExtractionResult(
                url=response.url,
                success=True,
                data=extracted_data,
                metadata={
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "timestamp": response.meta.get("download_timestamp")
                }
            )
            
            self.results.append(result)
            yield result.__dict__
            
        except Exception as e:
            logger.error(f"Parsing failed for {response.url}: {e}")
            error_result = ExtractionResult(
                url=response.url,
                success=False,
                data={},
                metadata={"status_code": response.status},
                error=str(e)
            )
            self.results.append(error_result)
            yield error_result.__dict__
            
    def _auto_extract(self, response) -> Dict[str, Any]:
        """Automatically extract common page elements."""
        data = {}
        
        # Title
        title = response.css("title::text").get()
        if title:
            data["title"] = title.strip()
            
        # Meta description
        description = response.css('meta[name="description"]::attr(content)').get()
        if description:
            data["description"] = description.strip()
            
        # Headers
        headers = []
        for i in range(1, 7):
            h_tags = response.css(f"h{i}::text").getall()
            if h_tags:
                headers.extend([h.strip() for h in h_tags])
        if headers:
            data["headers"] = headers
            
        # Links
        links = response.css("a::attr(href)").getall()
        if links:
            # Convert relative URLs to absolute
            absolute_links = [urljoin(response.url, link) for link in links]
            data["links"] = absolute_links[:50]  # Limit to first 50
            
        # Images
        images = response.css("img::attr(src)").getall()
        if images:
            absolute_images = [urljoin(response.url, img) for img in images]
            data["images"] = absolute_images[:20]  # Limit to first 20
            
        # Text content (paragraphs)
        paragraphs = response.css("p::text").getall()
        if paragraphs:
            clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]
            data["text_content"] = clean_paragraphs
            
        return data


class DataExtractionAgent:
    """
    Data extraction agent combining Scrapy and Diffbot capabilities.
    
    Features:
    - Intelligent web scraping with Scrapy
    - Structured data extraction with Diffbot
    - Auto-detection of content types
    - Rate limiting and politeness
    - Multi-format output support
    """
    
    def __init__(self, diffbot_token: str = None, user_agent: str = None):
        self.diffbot_client = DiffbotClient(diffbot_token) if diffbot_token else None
        self.user_agent = user_agent or "DataExtractionAgent/1.0"
        self.settings = self._get_scrapy_settings()
        
    def _get_scrapy_settings(self) -> dict:
        """Get optimized Scrapy settings."""
        return {
            "USER_AGENT": self.user_agent,
            "ROBOTSTXT_OBEY": True,
            "DOWNLOAD_DELAY": 1,
            "RANDOMIZE_DOWNLOAD_DELAY": 0.5,
            "CONCURRENT_REQUESTS": 8,
            "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
            "AUTOTHROTTLE_ENABLED": True,
            "AUTOTHROTTLE_START_DELAY": 1,
            "AUTOTHROTTLE_MAX_DELAY": 10,
            "AUTOTHROTTLE_TARGET_CONCURRENCY": 2.0,
            "LOG_LEVEL": "WARNING"
        }
        
    async def extract_with_diffbot(self, url: str, content_type: str = "auto") -> ExtractionResult:
        """
        Extract data using Diffbot API.
        
        Args:
            url: Target URL
            content_type: 'article', 'product', or 'auto'
            
        Returns:
            ExtractionResult with structured data
        """
        if not self.diffbot_client:
            return ExtractionResult(
                url=url,
                success=False,
                data={},
                metadata={},
                error="Diffbot token not provided"
            )
            
        try:
            if content_type == "auto":
                # Try article first, then product
                result = await self.diffbot_client.extract_article(url)
                if not result["success"]:
                    result = await self.diffbot_client.extract_product(url)
            elif content_type == "article":
                result = await self.diffbot_client.extract_article(url)
            elif content_type == "product":
                result = await self.diffbot_client.extract_product(url)
            else:
                return ExtractionResult(
                    url=url,
                    success=False,
                    data={},
                    metadata={},
                    error=f"Unsupported content type: {content_type}"
                )
                
            return ExtractionResult(
                url=url,
                success=result["success"],
                data=result.get("data", {}),
                metadata={"source": result.get("source", "diffbot")},
                error=result.get("error")
            )
            
        except Exception as e:
            logger.error(f"Diffbot extraction failed: {e}")
            return ExtractionResult(
                url=url,
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
            
    def extract_with_scrapy(self, urls: List[str], 
                          extraction_rules: Dict[str, Any] = None) -> List[ExtractionResult]:
        """
        Extract data using Scrapy spider.
        
        Args:
            urls: List of URLs to scrape
            extraction_rules: Dict of field -> extraction rule mappings
            
        Returns:
            List of ExtractionResult objects
        """
        try:
            from twisted.internet import reactor
            
            settings = get_project_settings()
            settings.update(self.settings)
            
            runner = CrawlerRunner(settings)
            
            spider_kwargs = {
                "start_urls": urls,
                "extraction_rules": extraction_rules or {}
            }
            
            d = runner.crawl(SmartSpider, **spider_kwargs)
            d.addBoth(lambda _: reactor.stop())
            
            if not reactor.running:
                reactor.run()
                
            # Get results from spider
            spider = list(runner.crawlers)[0].spider
            return spider.results
            
        except Exception as e:
            logger.error(f"Scrapy extraction failed: {e}")
            return [ExtractionResult(
                url=url,
                success=False,
                data={},
                metadata={},
                error=str(e)
            ) for url in urls]
            
    async def extract_batch(self, urls: List[str], 
                          method: str = "auto",
                          extraction_rules: Dict[str, Any] = None) -> List[ExtractionResult]:
        """
        Extract data from multiple URLs using the best available method.
        
        Args:
            urls: List of URLs to extract from
            method: 'scrapy', 'diffbot', or 'auto'
            extraction_rules: Custom extraction rules for Scrapy
            
        Returns:
            List of ExtractionResult objects
        """
        results = []
        
        if method == "diffbot" and self.diffbot_client:
            for url in urls:
                result = await self.extract_with_diffbot(url)
                results.append(result)
        elif method == "scrapy":
            results = self.extract_with_scrapy(urls, extraction_rules)
        else:  # auto
            # Try Diffbot first for single URLs, Scrapy for multiple
            if len(urls) == 1 and self.diffbot_client:
                result = await self.extract_with_diffbot(urls[0])
                results.append(result)
            else:
                results = self.extract_with_scrapy(urls, extraction_rules)
                
        return results
        
    def extract_structured_data(self, html_content: str, url: str = "") -> Dict[str, Any]:
        """
        Extract structured data (JSON-LD, microdata, etc.) from HTML.
        
        Args:
            html_content: HTML content to parse
            url: Source URL for context
            
        Returns:
            Dict with extracted structured data
        """
        import json
        from scrapy import Selector
        
        try:
            selector = Selector(text=html_content)
            structured_data = {}
            
            # Extract JSON-LD
            json_ld_scripts = selector.css('script[type="application/ld+json"]::text').getall()
            if json_ld_scripts:
                json_ld_data = []
                for script in json_ld_scripts:
                    try:
                        data = json.loads(script.strip())
                        json_ld_data.append(data)
                    except json.JSONDecodeError:
                        continue
                structured_data["json_ld"] = json_ld_data
                
            # Extract Open Graph data
            og_data = {}
            og_tags = selector.css('meta[property^="og:"]')
            for tag in og_tags:
                property_name = tag.css("::attr(property)").get()
                content = tag.css("::attr(content)").get()
                if property_name and content:
                    og_data[property_name] = content
            if og_data:
                structured_data["open_graph"] = og_data
                
            # Extract Twitter Card data
            twitter_data = {}
            twitter_tags = selector.css('meta[name^="twitter:"]')
            for tag in twitter_tags:
                name = tag.css("::attr(name)").get()
                content = tag.css("::attr(content)").get()
                if name and content:
                    twitter_data[name] = content
            if twitter_data:
                structured_data["twitter_card"] = twitter_data
                
            # Extract schema.org microdata
            microdata_items = selector.css('[itemscope]')
            if microdata_items:
                microdata = []
                for item in microdata_items:
                    item_type = item.css("::attr(itemtype)").get()
                    item_data = {"@type": item_type} if item_type else {}
                    
                    # Extract properties
                    props = item.css('[itemprop]')
                    for prop in props:
                        prop_name = prop.css("::attr(itemprop)").get()
                        prop_value = prop.css("::attr(content)").get() or prop.css("::text").get()
                        if prop_name and prop_value:
                            item_data[prop_name] = prop_value.strip()
                            
                    if item_data:
                        microdata.append(item_data)
                structured_data["microdata"] = microdata
                
            return {
                "success": True,
                "url": url,
                "structured_data": structured_data
            }
            
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
            
    def create_extraction_rules(self, selectors: Dict[str, str]) -> Dict[str, Dict]:
        """
        Create extraction rules from CSS selectors.
        
        Args:
            selectors: Dict mapping field names to CSS selectors
            
        Returns:
            Dict of extraction rules for Scrapy
        """
        rules = {}
        for field, selector in selectors.items():
            rules[field] = {
                "selector": selector,
                "attribute": "text",
                "multiple": False
            }
        return rules


# Convenience functions for quick usage
async def quick_extract(url: str, diffbot_token: str = None) -> ExtractionResult:
    """Quick extraction from a single URL."""
    agent = DataExtractionAgent(diffbot_token=diffbot_token)
    results = await agent.extract_batch([url])
    return results[0] if results else ExtractionResult(
        url=url, success=False, data={}, metadata={}, error="No results"
    )


def quick_scrape(urls: List[str], selectors: Dict[str, str] = None) -> List[ExtractionResult]:
    """Quick scraping with optional custom selectors."""
    agent = DataExtractionAgent()
    extraction_rules = agent.create_extraction_rules(selectors) if selectors else None
    return agent.extract_with_scrapy(urls, extraction_rules)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize agent
        agent = DataExtractionAgent()
        
        # Extract from a single URL
        urls = ["https://example.com"]
        results = await agent.extract_batch(urls)
        
        for result in results:
            print(f"URL: {result.url}")
            print(f"Success: {result.success}")
            print(f"Data keys: {list(result.data.keys())}")
            if result.error:
                print(f"Error: {result.error}")
            print("-" * 50)
            
    asyncio.run(main())
