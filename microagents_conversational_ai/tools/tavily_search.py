"""
Tavily Search Tool - AI-powered web search and research
Provides intelligent web search capabilities with real-time data access.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_tool import (
    BaseTool, ToolMetadata, ToolRequest, ToolResponse, 
    ToolType, ToolStatus, ToolCapability, create_tool_metadata
)

logger = logging.getLogger(__name__)

class TavilySearchTool(BaseTool):
    """
    Tavily AI search tool for intelligent web research.
    Provides real-time search results with AI-powered analysis.
    """
    
    def __init__(self, api_key: str = None, config: Dict[str, Any] = None):
        # Initialize metadata
        metadata = create_tool_metadata(
            tool_id="tavily_search",
            name="Tavily Search",
            description="AI-powered web search and research tool with real-time data access",
            tool_type=ToolType.SEARCH,
            version="1.0.0",
            author="Tavily",
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.REAL_TIME,
                ToolCapability.CACHEABLE,
                ToolCapability.REQUIRES_AUTH,
                ToolCapability.RATE_LIMITED,
                ToolCapability.NETWORK_DEPENDENT
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "default": "basic",
                        "description": "Depth of search analysis"
                    },
                    "include_answer": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include AI-generated answer"
                    },
                    "include_raw_content": {
                        "type": "boolean", 
                        "default": False,
                        "description": "Include raw content from sources"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum number of results"
                    },
                    "include_images": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include image results"
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Domains to include in search"
                    },
                    "exclude_domains": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Domains to exclude from search"
                    }
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "query": {"type": "string"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "content": {"type": "string"},
                                "score": {"type": "number"},
                                "published_date": {"type": "string"}
                            }
                        }
                    },
                    "images": {"type": "array"},
                    "response_time": {"type": "number"}
                }
            },
            rate_limit=60,  # 60 requests per minute
            timeout=30.0,
            requires_api_key=True,
            supported_formats=["json"],
            tags=["search", "ai", "web", "research", "real-time"]
        )
        
        super().__init__(metadata, config)
        
        # Set up API key
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            logger.warning("Tavily API key not provided. Tool will use mock responses.")
        
        # Configuration
        self.base_url = "https://api.tavily.com"
        self.default_config = {
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 5,
            "include_images": False
        }
        
        # Update config with defaults
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute Tavily search request."""
        try:
            # Extract parameters
            query = request.parameters.get("query")
            search_depth = request.parameters.get("search_depth", self.config["search_depth"])
            include_answer = request.parameters.get("include_answer", self.config["include_answer"])
            include_raw_content = request.parameters.get("include_raw_content", self.config["include_raw_content"])
            max_results = request.parameters.get("max_results", self.config["max_results"])
            include_images = request.parameters.get("include_images", self.config["include_images"])
            include_domains = request.parameters.get("include_domains", [])
            exclude_domains = request.parameters.get("exclude_domains", [])
            
            if not query:
                return ToolResponse(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    status=ToolStatus.FAILED,
                    error="Query parameter is required"
                )
            
            # Perform search
            if self.api_key:
                result = await self._perform_real_search(
                    query=query,
                    search_depth=search_depth,
                    include_answer=include_answer,
                    include_raw_content=include_raw_content,
                    max_results=max_results,
                    include_images=include_images,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains
                )
            else:
                result = await self._perform_mock_search(query, max_results)
            
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.COMPLETED,
                result=result,
                metadata={
                    "query": query,
                    "search_depth": search_depth,
                    "results_count": len(result.get("results", [])),
                    "has_answer": bool(result.get("answer")),
                    "search_timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _perform_real_search(self, query: str, search_depth: str = "basic",
                                 include_answer: bool = True, include_raw_content: bool = False,
                                 max_results: int = 5, include_images: bool = False,
                                 include_domains: List[str] = None, 
                                 exclude_domains: List[str] = None) -> Dict[str, Any]:
        """Perform real Tavily API search."""
        import aiohttp
        
        # Prepare request payload
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "max_results": max_results,
            "include_images": include_images
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return self._format_search_result(result)
                else:
                    error_text = await response.text()
                    raise Exception(f"Tavily API error {response.status}: {error_text}")
    
    async def _perform_mock_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform mock search for testing without API key."""
        await asyncio.sleep(0.5)  # Simulate API delay
        
        # Generate mock results based on query
        mock_results = []
        for i in range(min(max_results, 3)):
            mock_results.append({
                "title": f"Search Result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "content": f"This is mock content for search result {i+1} related to '{query}'. "
                          f"It contains relevant information that would typically be found in web search results.",
                "score": 0.9 - (i * 0.1),
                "published_date": "2024-12-01"
            })
        
        return {
            "answer": f"Based on the search for '{query}', here are the key findings: "
                     f"This is a mock AI-generated answer that would normally provide "
                     f"a comprehensive summary of the search results.",
            "query": query,
            "results": mock_results,
            "images": [],
            "response_time": 0.5
        }
    
    def _format_search_result(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw Tavily API response."""
        formatted = {
            "answer": raw_result.get("answer", ""),
            "query": raw_result.get("query", ""),
            "results": [],
            "images": raw_result.get("images", []),
            "response_time": raw_result.get("response_time", 0)
        }
        
        # Format search results
        for result in raw_result.get("results", []):
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "published_date": result.get("published_date", "")
            }
            formatted["results"].append(formatted_result)
        
        return formatted

class TavilyNewsSearchTool(BaseTool):
    """Tavily news search tool for current events and news research."""
    
    def __init__(self, api_key: str = None, config: Dict[str, Any] = None):
        metadata = create_tool_metadata(
            tool_id="tavily_news",
            name="Tavily News Search",
            description="AI-powered news search for current events and breaking news",
            tool_type=ToolType.SEARCH,
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.REAL_TIME,
                ToolCapability.REQUIRES_AUTH,
                ToolCapability.RATE_LIMITED,
                ToolCapability.NETWORK_DEPENDENT
            ],
            input_schema={
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "News search query"},
                    "days": {"type": "integer", "default": 3, "description": "Number of days back to search"},
                    "max_results": {"type": "integer", "default": 10, "description": "Maximum results"}
                },
                "required": ["query"]
            },
            rate_limit=100,
            requires_api_key=True,
            tags=["news", "current-events", "breaking-news", "journalism"]
        )
        
        super().__init__(metadata, config)
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
    
    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute news search."""
        try:
            query = request.parameters.get("query")
            days = request.parameters.get("days", 3)
            max_results = request.parameters.get("max_results", 10)
            
            if self.api_key:
                result = await self._search_news(query, days, max_results)
            else:
                result = await self._mock_news_search(query, max_results)
            
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.COMPLETED,
                result=result,
                metadata={"search_type": "news", "days_back": days}
            )
            
        except Exception as e:
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _search_news(self, query: str, days: int, max_results: int) -> Dict[str, Any]:
        """Perform real news search via Tavily API."""
        import aiohttp
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "days": days,
            "topic": "news"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.tavily.com/search",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"News search failed: {response.status}")
    
    async def _mock_news_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Mock news search for testing."""
        await asyncio.sleep(0.3)
        
        news_results = []
        for i in range(min(max_results, 5)):
            news_results.append({
                "title": f"Breaking: {query} - Latest Update {i+1}",
                "url": f"https://news-example.com/story{i+1}",
                "content": f"Latest news regarding {query}. This is mock news content {i+1}.",
                "published_date": "2024-12-01",
                "source": f"News Source {i+1}",
                "score": 0.95 - (i * 0.05)
            })
        
        return {
            "query": query,
            "results": news_results,
            "total_results": len(news_results),
            "search_type": "news"
        }

# Factory functions for easy tool creation
def create_tavily_search_tool(api_key: str = None, **config) -> TavilySearchTool:
    """Create a Tavily search tool instance."""
    return TavilySearchTool(api_key=api_key, config=config)

def create_tavily_news_tool(api_key: str = None, **config) -> TavilyNewsSearchTool:
    """Create a Tavily news search tool instance.""" 
    return TavilyNewsSearchTool(api_key=api_key, config=config)

# Example usage and testing
if __name__ == "__main__":
    async def demo():
        print("Tavily Search Tools Demo")
        print("=" * 40)
        
        # Create tools
        search_tool = create_tavily_search_tool()
        news_tool = create_tavily_news_tool()
        
        # Test search tool
        search_request = ToolRequest(
            request_id="test_search_1",
            tool_id="tavily_search",
            action="search",
            parameters={
                "query": "artificial intelligence latest developments 2024",
                "max_results": 3,
                "include_answer": True
            }
        )
        
        print("🔍 Testing web search...")
        search_response = await search_tool.execute(search_request)
        print(f"Status: {search_response.status.value}")
        
        if search_response.status == ToolStatus.COMPLETED:
            result = search_response.result
            print(f"Answer: {result['answer'][:100]}...")
            print(f"Results: {len(result['results'])} found")
            for i, res in enumerate(result['results'][:2]):
                print(f"  {i+1}. {res['title']}")
        
        # Test news tool
        news_request = ToolRequest(
            request_id="test_news_1",
            tool_id="tavily_news", 
            action="search",
            parameters={
                "query": "technology breakthroughs",
                "days": 7,
                "max_results": 3
            }
        )
        
        print(f"\n📰 Testing news search...")
        news_response = await news_tool.execute(news_request)
        print(f"Status: {news_response.status.value}")
        
        if news_response.status == ToolStatus.COMPLETED:
            result = news_response.result
            print(f"News results: {len(result['results'])} found")
            for i, res in enumerate(result['results'][:2]):
                print(f"  {i+1}. {res['title']}")
        
        print(f"\n✅ Tavily Tools Demo completed!")
    
    asyncio.run(demo())
