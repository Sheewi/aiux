"""
External API Integration Hub
Centralized management for all external API integrations

This module provides:
- Unified API client management
- Authentication handling
- Rate limiting and retry logic
- Response caching
- Error handling and monitoring
- API health checks
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import hashlib
import os

logger = logging.getLogger(__name__)

class APIProvider(Enum):
    """Supported API providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GITHUB = "github"
    FIGMA = "figma"
    SLACK = "slack"
    NOTION = "notion"
    AIRTABLE = "airtable"
    STRIPE = "stripe"
    CUSTOM = "custom"

class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

@dataclass
class APICredentials:
    """API credentials storage"""
    provider: APIProvider
    api_key: str
    secret_key: str = ""
    base_url: str = ""
    additional_headers: Dict[str, str] = field(default_factory=dict)
    rate_limit: int = 100  # requests per minute
    timeout: int = 30

@dataclass
class APIRequest:
    """API request configuration"""
    provider: APIProvider
    endpoint: str
    method: HTTPMethod = HTTPMethod.GET
    data: Optional[Dict] = None
    params: Optional[Dict] = None
    headers: Optional[Dict] = None
    timeout: Optional[int] = None
    cache_ttl: int = 0  # seconds, 0 = no cache

@dataclass
class APIResponse:
    """API response wrapper"""
    status_code: int
    data: Any
    headers: Dict[str, str]
    request_id: str
    provider: APIProvider
    cached: bool = False
    response_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self) -> bool:
        """Check if request can be made"""
        now = time.time()
        
        # Remove old requests outside time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def time_until_available(self) -> float:
        """Time until next request can be made"""
        if len(self.requests) < self.max_requests:
            return 0.0
        
        oldest_request = min(self.requests)
        return self.time_window - (time.time() - oldest_request)

class APICache:
    """Simple in-memory cache for API responses"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def _generate_key(self, request: APIRequest) -> str:
        """Generate cache key for request"""
        key_data = f"{request.provider.value}:{request.endpoint}:{request.method.value}"
        if request.params:
            key_data += f":{json.dumps(request.params, sort_keys=True)}"
        if request.data:
            key_data += f":{json.dumps(request.data, sort_keys=True)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, request: APIRequest) -> Optional[APIResponse]:
        """Get cached response"""
        if request.cache_ttl <= 0:
            return None
        
        key = self._generate_key(request)
        if key in self.cache:
            cached_response, cached_time = self.cache[key]
            
            if time.time() - cached_time < request.cache_ttl:
                self.access_times[key] = time.time()
                cached_response.cached = True
                return cached_response
            else:
                # Expired
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        return None
    
    def set(self, request: APIRequest, response: APIResponse):
        """Cache response"""
        if request.cache_ttl <= 0:
            return
        
        # Cleanup if cache is full
        if len(self.cache) >= self.max_size:
            self._cleanup_cache()
        
        key = self._generate_key(request)
        self.cache[key] = (response, time.time())
        self.access_times[key] = time.time()
    
    def _cleanup_cache(self):
        """Remove least recently used items"""
        if not self.access_times:
            return
        
        # Remove 25% of least recently used items
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        items_to_remove = len(sorted_items) // 4
        
        for key, _ in sorted_items[:items_to_remove]:
            if key in self.cache:
                del self.cache[key]
            del self.access_times[key]

class ExternalAPIHub:
    """Central hub for external API integrations"""
    
    def __init__(self, workspace_path: str = "/media/r/Workspace"):
        self.workspace_path = workspace_path
        self.credentials = {}
        self.rate_limiters = {}
        self.cache = APICache()
        self.session = None
        self.logger = logging.getLogger("external_api_hub")
        
        # API configurations
        self.api_configs = {
            APIProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1',
                'auth_header': 'Authorization',
                'auth_prefix': 'Bearer ',
                'default_headers': {'Content-Type': 'application/json'}
            },
            APIProvider.ANTHROPIC: {
                'base_url': 'https://api.anthropic.com/v1',
                'auth_header': 'x-api-key',
                'auth_prefix': '',
                'default_headers': {'Content-Type': 'application/json'}
            },
            APIProvider.GITHUB: {
                'base_url': 'https://api.github.com',
                'auth_header': 'Authorization',
                'auth_prefix': 'token ',
                'default_headers': {'Accept': 'application/vnd.github.v3+json'}
            },
            APIProvider.FIGMA: {
                'base_url': 'https://api.figma.com/v1',
                'auth_header': 'X-Figma-Token',
                'auth_prefix': '',
                'default_headers': {}
            },
            APIProvider.SLACK: {
                'base_url': 'https://slack.com/api',
                'auth_header': 'Authorization',
                'auth_prefix': 'Bearer ',
                'default_headers': {'Content-Type': 'application/json'}
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize API hub"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Load credentials from environment or config
            await self._load_credentials()
            
            self.logger.info("External API hub initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"API hub initialization failed: {e}")
            return False
    
    async def _load_credentials(self):
        """Load API credentials from environment"""
        try:
            # Load from environment variables
            env_mappings = {
                APIProvider.OPENAI: 'OPENAI_API_KEY',
                APIProvider.ANTHROPIC: 'ANTHROPIC_API_KEY',
                APIProvider.GITHUB: 'GITHUB_TOKEN',
                APIProvider.FIGMA: 'FIGMA_TOKEN',
                APIProvider.SLACK: 'SLACK_TOKEN'
            }
            
            for provider, env_var in env_mappings.items():
                api_key = os.getenv(env_var)
                if api_key:
                    config = self.api_configs.get(provider, {})
                    credentials = APICredentials(
                        provider=provider,
                        api_key=api_key,
                        base_url=config.get('base_url', ''),
                        additional_headers=config.get('default_headers', {}),
                        rate_limit=100,
                        timeout=30
                    )
                    await self.add_credentials(credentials)
            
            # Load from config file if exists
            config_path = os.path.join(self.workspace_path, 'config', 'api_credentials.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    for provider_name, cred_data in config_data.items():
                        try:
                            provider = APIProvider(provider_name)
                            credentials = APICredentials(
                                provider=provider,
                                api_key=cred_data.get('api_key', ''),
                                secret_key=cred_data.get('secret_key', ''),
                                base_url=cred_data.get('base_url', ''),
                                additional_headers=cred_data.get('headers', {}),
                                rate_limit=cred_data.get('rate_limit', 100),
                                timeout=cred_data.get('timeout', 30)
                            )
                            await self.add_credentials(credentials)
                        except ValueError:
                            self.logger.warning(f"Unknown API provider: {provider_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {e}")
    
    async def add_credentials(self, credentials: APICredentials) -> bool:
        """Add API credentials"""
        try:
            self.credentials[credentials.provider] = credentials
            
            # Initialize rate limiter
            self.rate_limiters[credentials.provider] = RateLimiter(
                max_requests=credentials.rate_limit,
                time_window=60
            )
            
            self.logger.info(f"Added credentials for {credentials.provider.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add credentials: {e}")
            return False
    
    async def make_request(self, request: APIRequest) -> APIResponse:
        """Make API request"""
        try:
            request_id = f"req_{int(time.time())}_{hash(str(request)) % 10000}"
            start_time = time.time()
            
            # Check cache first
            cached_response = self.cache.get(request)
            if cached_response:
                return cached_response
            
            # Check credentials
            if request.provider not in self.credentials:
                raise ValueError(f"No credentials found for {request.provider.value}")
            
            credentials = self.credentials[request.provider]
            
            # Check rate limit
            rate_limiter = self.rate_limiters.get(request.provider)
            if rate_limiter and not await rate_limiter.acquire():
                wait_time = rate_limiter.time_until_available()
                raise Exception(f"Rate limit exceeded. Wait {wait_time:.1f} seconds")
            
            # Build request
            url = self._build_url(credentials, request.endpoint)
            headers = self._build_headers(credentials, request.headers)
            timeout = request.timeout or credentials.timeout
            
            # Make HTTP request
            async with self.session.request(
                method=request.method.value,
                url=url,
                json=request.data if request.method != HTTPMethod.GET else None,
                params=request.params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                response_data = await self._parse_response(response)
                response_time = time.time() - start_time
                
                api_response = APIResponse(
                    status_code=response.status,
                    data=response_data,
                    headers=dict(response.headers),
                    request_id=request_id,
                    provider=request.provider,
                    response_time=response_time
                )
                
                # Cache successful responses
                if response.status < 400:
                    self.cache.set(request, api_response)
                
                return api_response
                
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            return APIResponse(
                status_code=500,
                data={'error': str(e)},
                headers={},
                request_id=request_id,
                provider=request.provider,
                response_time=time.time() - start_time
            )
    
    def _build_url(self, credentials: APICredentials, endpoint: str) -> str:
        """Build complete URL"""
        base_url = credentials.base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        return f"{base_url}/{endpoint}"
    
    def _build_headers(self, credentials: APICredentials, request_headers: Optional[Dict] = None) -> Dict[str, str]:
        """Build request headers with authentication"""
        headers = credentials.additional_headers.copy()
        
        if request_headers:
            headers.update(request_headers)
        
        # Add authentication
        config = self.api_configs.get(credentials.provider, {})
        auth_header = config.get('auth_header')
        auth_prefix = config.get('auth_prefix', '')
        
        if auth_header and credentials.api_key:
            headers[auth_header] = f"{auth_prefix}{credentials.api_key}"
        
        return headers
    
    async def _parse_response(self, response) -> Any:
        """Parse HTTP response"""
        try:
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                return await response.json()
            elif 'text/' in content_type:
                return await response.text()
            else:
                return await response.read()
                
        except Exception as e:
            self.logger.error(f"Response parsing failed: {e}")
            return {'error': 'Failed to parse response'}
    
    # Convenience methods for common APIs
    
    async def openai_chat_completion(self, messages: List[Dict], model: str = "gpt-3.5-turbo") -> APIResponse:
        """OpenAI chat completion"""
        request = APIRequest(
            provider=APIProvider.OPENAI,
            endpoint="chat/completions",
            method=HTTPMethod.POST,
            data={
                "model": model,
                "messages": messages,
                "temperature": 0.7
            },
            cache_ttl=300  # 5 minutes
        )
        return await self.make_request(request)
    
    async def github_get_repo(self, owner: str, repo: str) -> APIResponse:
        """Get GitHub repository info"""
        request = APIRequest(
            provider=APIProvider.GITHUB,
            endpoint=f"repos/{owner}/{repo}",
            method=HTTPMethod.GET,
            cache_ttl=3600  # 1 hour
        )
        return await self.make_request(request)
    
    async def figma_get_file(self, file_key: str) -> APIResponse:
        """Get Figma file"""
        request = APIRequest(
            provider=APIProvider.FIGMA,
            endpoint=f"files/{file_key}",
            method=HTTPMethod.GET,
            cache_ttl=1800  # 30 minutes
        )
        return await self.make_request(request)
    
    async def slack_send_message(self, channel: str, text: str) -> APIResponse:
        """Send Slack message"""
        request = APIRequest(
            provider=APIProvider.SLACK,
            endpoint="chat.postMessage",
            method=HTTPMethod.POST,
            data={
                "channel": channel,
                "text": text
            }
        )
        return await self.make_request(request)
    
    async def health_check(self, provider: APIProvider) -> Dict[str, Any]:
        """Check API health"""
        try:
            if provider not in self.credentials:
                return {'status': 'error', 'message': 'No credentials configured'}
            
            # Simple health check endpoints
            health_endpoints = {
                APIProvider.OPENAI: "models",
                APIProvider.GITHUB: "user",
                APIProvider.FIGMA: "me",
                APIProvider.SLACK: "auth.test"
            }
            
            endpoint = health_endpoints.get(provider)
            if not endpoint:
                return {'status': 'unknown', 'message': 'No health check endpoint'}
            
            request = APIRequest(
                provider=provider,
                endpoint=endpoint,
                method=HTTPMethod.GET,
                timeout=10
            )
            
            response = await self.make_request(request)
            
            if response.status_code < 400:
                return {
                    'status': 'healthy',
                    'response_time': response.response_time,
                    'timestamp': response.timestamp
                }
            else:
                return {
                    'status': 'unhealthy',
                    'status_code': response.status_code,
                    'error': response.data
                }
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def get_all_health_status(self) -> Dict[str, Any]:
        """Get health status for all configured APIs"""
        health_status = {}
        
        for provider in self.credentials.keys():
            health_status[provider.value] = await self.health_check(provider)
        
        return {
            'timestamp': time.time(),
            'total_apis': len(self.credentials),
            'health_checks': health_status
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get API hub status"""
        return {
            'initialized': bool(self.session),
            'configured_apis': [p.value for p in self.credentials.keys()],
            'cache_size': len(self.cache.cache),
            'rate_limiters': len(self.rate_limiters),
            'workspace_path': self.workspace_path,
            'capabilities': [
                'multi_provider_support',
                'authentication_management',
                'rate_limiting',
                'response_caching',
                'health_monitoring',
                'error_handling'
            ]
        }
    
    async def close(self):
        """Close API hub and cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.logger.info("API hub closed")

# Global API hub instance
api_hub = ExternalAPIHub()

async def initialize_api_hub() -> bool:
    """Initialize API hub"""
    return await api_hub.initialize()

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("🌐 External API Hub Demo")
        print("=" * 50)
        
        success = await api_hub.initialize()
        print(f"Initialization: {'✅ Success' if success else '❌ Failed'}")
        
        if success:
            # Demo health checks
            health_status = await api_hub.get_all_health_status()
            print(f"Health checks: {len(health_status.get('health_checks', {}))}")
            
            # Demo status
            status = api_hub.get_status()
            print(f"Configured APIs: {status['configured_apis']}")
            print(f"Status: {status}")
            
            # Cleanup
            await api_hub.close()
        
        print("Demo complete!")
    
    asyncio.run(demo())
