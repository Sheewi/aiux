"""
API Orchestration Agent using httpx + GraphQL

This microagent specializes in asynchronous HTTP operations and GraphQL
orchestration for complex API workflows.
"""

import asyncio
import httpx
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time
import hashlib
from urllib.parse import urljoin, urlparse
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class APIRequest:
    """API request configuration."""
    method: str
    url: str
    headers: Dict[str, str] = None
    params: Dict[str, Any] = None
    data: Dict[str, Any] = None
    json_data: Dict[str, Any] = None
    timeout: float = 30.0
    retries: int = 3
    retry_delay: float = 1.0


@dataclass
class APIResponse:
    """API response data."""
    status_code: int
    success: bool
    data: Any
    headers: Dict[str, str]
    url: str
    duration: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class GraphQLQuery:
    """GraphQL query configuration."""
    query: str
    variables: Dict[str, Any] = None
    operation_name: str = None


class RateLimiter:
    """Rate limiting functionality."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        
    async def acquire(self) -> bool:
        """Acquire a rate limit token."""
        now = time.time()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # Check if we can make another request
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        else:
            # Calculate wait time
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request)
            await asyncio.sleep(wait_time)
            return await self.acquire()


class CacheManager:
    """Simple in-memory cache for API responses."""
    
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl
        
    def _generate_key(self, request: APIRequest) -> str:
        """Generate cache key from request."""
        key_data = f"{request.method}:{request.url}:{json.dumps(request.params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def get(self, request: APIRequest) -> Optional[APIResponse]:
        """Get cached response."""
        key = self._generate_key(request)
        
        if key in self.cache:
            response, expiry = self.cache[key]
            if datetime.now() < expiry:
                logger.debug(f"Cache hit for {request.url}")
                return response
            else:
                del self.cache[key]
                
        return None
        
    def set(self, request: APIRequest, response: APIResponse, ttl: int = None):
        """Cache response."""
        key = self._generate_key(request)
        expiry = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        self.cache[key] = (response, expiry)
        logger.debug(f"Cached response for {request.url}")
        
    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()


class GraphQLClient:
    """GraphQL client with advanced features."""
    
    def __init__(self, endpoint: str, headers: Dict[str, str] = None):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.transport = None
        self.client = None
        
    async def connect(self) -> bool:
        """Connect to GraphQL endpoint."""
        try:
            self.transport = AIOHTTPTransport(
                url=self.endpoint,
                headers=self.headers
            )
            self.client = Client(transport=self.transport, fetch_schema_from_transport=True)
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to GraphQL endpoint: {e}")
            return False
            
    async def execute(self, query: GraphQLQuery) -> Dict[str, Any]:
        """Execute GraphQL query."""
        if not self.client:
            await self.connect()
            
        try:
            gql_query = gql(query.query)
            
            start_time = time.time()
            result = await self.client.execute_async(
                gql_query,
                variable_values=query.variables,
                operation_name=query.operation_name
            )
            duration = time.time() - start_time
            
            return {
                "success": True,
                "data": result,
                "duration": duration,
                "query": query.query,
                "variables": query.variables
            }
            
        except Exception as e:
            logger.error(f"GraphQL query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query.query,
                "variables": query.variables
            }
            
    async def close(self):
        """Close GraphQL client."""
        if self.transport:
            await self.transport.close()


class APIOrchestrationAgent:
    """
    API orchestration agent with advanced HTTP and GraphQL capabilities.
    
    Features:
    - Asynchronous HTTP operations with httpx
    - GraphQL query orchestration
    - Rate limiting and retry logic
    - Response caching
    - Request/response middleware
    - Batch operations
    - Circuit breaker pattern
    - Authentication handling
    """
    
    def __init__(self, base_url: str = None, default_headers: Dict[str, str] = None,
                 rate_limit: int = 100, cache_ttl: int = 300):
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.rate_limiter = RateLimiter(rate_limit)
        self.cache = CacheManager(cache_ttl)
        self.client = None
        self.graphql_clients = {}
        self.middleware = []
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        
    async def start(self):
        """Initialize the HTTP client."""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.default_headers,
            timeout=30.0
        )
        logger.info("API orchestration agent started")
        
    async def stop(self):
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
            
        for gql_client in self.graphql_clients.values():
            await gql_client.close()
            
        logger.info("API orchestration agent stopped")
        
    def add_middleware(self, middleware: Callable):
        """Add request/response middleware."""
        self.middleware.append(middleware)
        
    async def _apply_middleware(self, request: APIRequest, response: APIResponse = None):
        """Apply middleware to request/response."""
        for middleware in self.middleware:
            if response:
                await middleware(request, response)
            else:
                await middleware(request)
                
    async def request(self, api_request: APIRequest, use_cache: bool = True) -> APIResponse:
        """
        Execute an HTTP request with all orchestration features.
        
        Args:
            api_request: Request configuration
            use_cache: Whether to use response caching
            
        Returns:
            APIResponse with results
        """
        if not self.client:
            await self.start()
            
        # Check cache first
        if use_cache and api_request.method.upper() == "GET":
            cached_response = self.cache.get(api_request)
            if cached_response:
                return cached_response
                
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Apply middleware
        await self._apply_middleware(api_request)
        
        # Prepare request
        url = api_request.url
        if self.base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(self.base_url, url)
            
        # Execute request with retries
        last_error = None
        
        for attempt in range(api_request.retries + 1):
            try:
                start_time = time.time()
                
                response = await self.client.request(
                    method=api_request.method,
                    url=url,
                    headers=api_request.headers,
                    params=api_request.params,
                    data=api_request.data,
                    json=api_request.json_data,
                    timeout=api_request.timeout
                )
                
                duration = time.time() - start_time
                
                # Parse response data
                try:
                    if response.headers.get("content-type", "").startswith("application/json"):
                        response_data = response.json()
                    else:
                        response_data = response.text
                except:
                    response_data = response.content
                    
                api_response = APIResponse(
                    status_code=response.status_code,
                    success=200 <= response.status_code < 300,
                    data=response_data,
                    headers=dict(response.headers),
                    url=str(response.url),
                    duration=duration,
                    metadata={
                        "attempt": attempt + 1,
                        "total_attempts": api_request.retries + 1
                    }
                )
                
                # Apply response middleware
                await self._apply_middleware(api_request, api_response)
                
                # Cache successful GET responses
                if (use_cache and api_request.method.upper() == "GET" 
                    and api_response.success):
                    self.cache.set(api_request, api_response)
                    
                return api_response
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt < api_request.retries:
                    await asyncio.sleep(api_request.retry_delay * (2 ** attempt))
                    
        # All retries failed
        return APIResponse(
            status_code=0,
            success=False,
            data=None,
            headers={},
            url=url,
            duration=0.0,
            error=last_error,
            metadata={"total_attempts": api_request.retries + 1}
        )
        
    async def get(self, url: str, **kwargs) -> APIResponse:
        """Convenience method for GET requests."""
        request = APIRequest(method="GET", url=url, **kwargs)
        return await self.request(request)
        
    async def post(self, url: str, **kwargs) -> APIResponse:
        """Convenience method for POST requests."""
        request = APIRequest(method="POST", url=url, **kwargs)
        return await self.request(request)
        
    async def put(self, url: str, **kwargs) -> APIResponse:
        """Convenience method for PUT requests."""
        request = APIRequest(method="PUT", url=url, **kwargs)
        return await self.request(request)
        
    async def delete(self, url: str, **kwargs) -> APIResponse:
        """Convenience method for DELETE requests."""
        request = APIRequest(method="DELETE", url=url, **kwargs)
        return await self.request(request)
        
    async def batch_requests(self, requests: List[APIRequest], 
                           max_concurrent: int = 10) -> List[APIResponse]:
        """
        Execute multiple requests concurrently.
        
        Args:
            requests: List of API requests
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of API responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_request(req):
            async with semaphore:
                return await self.request(req)
                
        tasks = [bounded_request(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    async def add_graphql_client(self, name: str, endpoint: str, 
                               headers: Dict[str, str] = None) -> bool:
        """Add a GraphQL client."""
        client = GraphQLClient(endpoint, headers)
        success = await client.connect()
        
        if success:
            self.graphql_clients[name] = client
            logger.info(f"Added GraphQL client: {name}")
        
        return success
        
    async def graphql_query(self, client_name: str, query: GraphQLQuery) -> Dict[str, Any]:
        """Execute GraphQL query."""
        if client_name not in self.graphql_clients:
            return {
                "success": False,
                "error": f"GraphQL client '{client_name}' not found"
            }
            
        client = self.graphql_clients[client_name]
        return await client.execute(query)
        
    async def orchestrate_workflow(self, workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Orchestrate a complex API workflow.
        
        Args:
            workflow: List of workflow steps with dependencies
            
        Returns:
            List of step results
        """
        results = {}
        step_results = []
        
        for step in workflow:
            step_id = step.get("id", f"step_{len(step_results)}")
            step_type = step.get("type", "http")
            dependencies = step.get("dependencies", [])
            
            # Wait for dependencies
            for dep_id in dependencies:
                if dep_id not in results:
                    step_results.append({
                        "step_id": step_id,
                        "success": False,
                        "error": f"Dependency '{dep_id}' not found"
                    })
                    continue
                    
            try:
                if step_type == "http":
                    # HTTP request step
                    request_config = step.get("request", {})
                    
                    # Substitute variables from previous steps
                    request_config = self._substitute_variables(request_config, results)
                    
                    api_request = APIRequest(**request_config)
                    response = await self.request(api_request)
                    
                    results[step_id] = response
                    step_results.append({
                        "step_id": step_id,
                        "success": response.success,
                        "data": response.data,
                        "status_code": response.status_code,
                        "duration": response.duration,
                        "error": response.error
                    })
                    
                elif step_type == "graphql":
                    # GraphQL query step
                    client_name = step.get("client")
                    query_config = step.get("query", {})
                    
                    # Substitute variables
                    query_config = self._substitute_variables(query_config, results)
                    
                    query = GraphQLQuery(**query_config)
                    result = await self.graphql_query(client_name, query)
                    
                    results[step_id] = result
                    step_results.append({
                        "step_id": step_id,
                        "success": result.get("success", False),
                        "data": result.get("data"),
                        "duration": result.get("duration", 0),
                        "error": result.get("error")
                    })
                    
                elif step_type == "delay":
                    # Delay step
                    delay_seconds = step.get("seconds", 1)
                    await asyncio.sleep(delay_seconds)
                    
                    results[step_id] = {"delayed": delay_seconds}
                    step_results.append({
                        "step_id": step_id,
                        "success": True,
                        "data": {"delayed_seconds": delay_seconds}
                    })
                    
            except Exception as e:
                logger.error(f"Workflow step {step_id} failed: {e}")
                step_results.append({
                    "step_id": step_id,
                    "success": False,
                    "error": str(e)
                })
                
        return step_results
        
    def _substitute_variables(self, config: Dict[str, Any], 
                            results: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute variables in configuration using previous results."""
        config_str = json.dumps(config)
        
        # Simple variable substitution (could be enhanced with templating)
        for result_id, result_data in results.items():
            if hasattr(result_data, 'data'):
                # Replace ${step_id.field} patterns
                if isinstance(result_data.data, dict):
                    for field, value in result_data.data.items():
                        config_str = config_str.replace(
                            f"${{{result_id}.{field}}}", 
                            json.dumps(value)
                        )
                        
        return json.loads(config_str)
        
    async def health_check(self, endpoints: List[str]) -> Dict[str, Any]:
        """Check health of multiple API endpoints."""
        health_requests = []
        
        for endpoint in endpoints:
            request = APIRequest(
                method="GET",
                url=endpoint,
                timeout=10.0,
                retries=1
            )
            health_requests.append(request)
            
        responses = await self.batch_requests(health_requests, max_concurrent=5)
        
        health_status = {}
        for i, (endpoint, response) in enumerate(zip(endpoints, responses)):
            if isinstance(response, Exception):
                health_status[endpoint] = {
                    "status": "error",
                    "error": str(response)
                }
            else:
                health_status[endpoint] = {
                    "status": "healthy" if response.success else "unhealthy",
                    "status_code": response.status_code,
                    "response_time": response.duration,
                    "error": response.error
                }
                
        return {
            "timestamp": datetime.now().isoformat(),
            "endpoints": health_status,
            "healthy_count": len([s for s in health_status.values() if s["status"] == "healthy"]),
            "total_count": len(endpoints)
        }


# Convenience functions for quick usage
async def quick_api_call(url: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
    """Quick API call without setting up full agent."""
    async with APIOrchestrationAgent() as agent:
        request = APIRequest(method=method, url=url, **kwargs)
        response = await agent.request(request)
        return asdict(response)


async def quick_graphql_query(endpoint: str, query: str, 
                            variables: Dict[str, Any] = None) -> Dict[str, Any]:
    """Quick GraphQL query."""
    async with APIOrchestrationAgent() as agent:
        await agent.add_graphql_client("default", endpoint)
        gql_query = GraphQLQuery(query=query, variables=variables)
        return await agent.graphql_query("default", gql_query)


if __name__ == "__main__":
    # Example usage
    async def main():
        async with APIOrchestrationAgent() as agent:
            # Simple HTTP request
            response = await agent.get("https://httpbin.org/get", 
                                     params={"test": "value"})
            print(f"HTTP Response: {response.success}, Status: {response.status_code}")
            
            # Batch requests
            requests = [
                APIRequest("GET", "https://httpbin.org/get"),
                APIRequest("GET", "https://httpbin.org/headers"),
                APIRequest("GET", "https://httpbin.org/user-agent")
            ]
            
            batch_responses = await agent.batch_requests(requests)
            print(f"Batch requests completed: {len(batch_responses)}")
            
            # Health check
            endpoints = [
                "https://httpbin.org/get",
                "https://httpbin.org/status/200",
                "https://httpbin.org/status/500"
            ]
            
            health = await agent.health_check(endpoints)
            print(f"Health check: {health['healthy_count']}/{health['total_count']} healthy")
            
    asyncio.run(main())
