"""
API Client Tools - Unified interface for external API interactions
Provides standardized access to various APIs with authentication and rate limiting.
"""

import asyncio
import aiohttp
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import base64
import hashlib
import hmac
from urllib.parse import urlencode, quote

from .base_tool import (
    BaseTool, ToolMetadata, ToolRequest, ToolResponse,
    ToolType, ToolStatus, ToolCapability, create_tool_metadata
)

logger = logging.getLogger(__name__)

class RestApiTool(BaseTool):
    """Generic REST API client tool with authentication support."""
    
    def __init__(self, config: Dict[str, Any] = None):
        metadata = create_tool_metadata(
            tool_id="rest_api_client",
            name="REST API Client",
            description="Generic REST API client with support for various authentication methods",
            tool_type=ToolType.API_CLIENT,
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.RATE_LIMITED,
                ToolCapability.REQUIRES_AUTH,
                ToolCapability.NETWORK_DEPENDENT
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "API endpoint URL"},
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                        "default": "GET",
                        "description": "HTTP method"
                    },
                    "headers": {"type": "object", "description": "HTTP headers"},
                    "params": {"type": "object", "description": "Query parameters"},
                    "data": {"type": "object", "description": "Request body data"},
                    "json": {"type": "object", "description": "JSON request body"},
                    "auth": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["basic", "bearer", "api_key", "oauth", "custom"]
                            },
                            "credentials": {"type": "object"}
                        },
                        "description": "Authentication configuration"
                    },
                    "timeout": {"type": "number", "default": 30},
                    "follow_redirects": {"type": "boolean", "default": True},
                    "verify_ssl": {"type": "boolean", "default": True}
                },
                "required": ["url"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status_code": {"type": "integer"},
                    "headers": {"type": "object"},
                    "data": {"type": ["object", "array", "string"]},
                    "response_time": {"type": "number"},
                    "content_type": {"type": "string"},
                    "content_length": {"type": "integer"}
                }
            },
            rate_limit=120,  # 120 requests per minute
            timeout=60.0,
            tags=["api", "rest", "http", "client", "integration"]
        )
        
        super().__init__(metadata, config)
    
    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute REST API request."""
        try:
            url = request.parameters["url"]
            method = request.parameters.get("method", "GET").upper()
            headers = request.parameters.get("headers", {})
            params = request.parameters.get("params", {})
            data = request.parameters.get("data")
            json_data = request.parameters.get("json")
            auth_config = request.parameters.get("auth")
            timeout = request.parameters.get("timeout", 30)
            follow_redirects = request.parameters.get("follow_redirects", True)
            verify_ssl = request.parameters.get("verify_ssl", True)
            
            start_time = time.time()
            
            # Prepare authentication
            if auth_config:
                headers = await self._add_authentication(headers, auth_config, method, url)
            
            # Prepare request parameters
            kwargs = {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "timeout": aiohttp.ClientTimeout(total=timeout),
                "allow_redirects": follow_redirects,
                "ssl": verify_ssl
            }
            
            # Add request body
            if json_data:
                kwargs["json"] = json_data
            elif data:
                kwargs["data"] = data
            
            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.request(**kwargs) as response:
                    response_time = time.time() - start_time
                    
                    # Read response
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/json' in content_type:
                        response_data = await response.json()
                    else:
                        response_data = await response.text()
                    
                    result = {
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "data": response_data,
                        "response_time": response_time,
                        "content_type": content_type,
                        "content_length": len(str(response_data))
                    }
                    
                    return ToolResponse(
                        request_id=request.request_id,
                        tool_id=request.tool_id,
                        status=ToolStatus.COMPLETED,
                        result=result,
                        metadata={
                            "method": method,
                            "url": url,
                            "success": 200 <= response.status < 300
                        }
                    )
            
        except Exception as e:
            logger.error(f"REST API request failed: {e}")
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _add_authentication(self, headers: Dict[str, str], 
                                auth_config: Dict[str, Any],
                                method: str, url: str) -> Dict[str, str]:
        """Add authentication headers based on auth type."""
        auth_type = auth_config.get("type")
        credentials = auth_config.get("credentials", {})
        headers = headers.copy()
        
        if auth_type == "basic":
            username = credentials.get("username")
            password = credentials.get("password")
            if username and password:
                auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {auth_string}"
        
        elif auth_type == "bearer":
            token = credentials.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif auth_type == "api_key":
            api_key = credentials.get("api_key")
            key_name = credentials.get("key_name", "X-API-Key")
            location = credentials.get("location", "header")  # header or query
            
            if api_key:
                if location == "header":
                    headers[key_name] = api_key
                # For query parameters, would need to modify URL
        
        elif auth_type == "oauth":
            # OAuth 1.0a implementation would go here
            # This is a simplified version
            token = credentials.get("access_token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif auth_type == "custom":
            # Custom headers
            custom_headers = credentials.get("headers", {})
            headers.update(custom_headers)
        
        return headers

class GraphQLTool(BaseTool):
    """GraphQL API client tool."""
    
    def __init__(self, config: Dict[str, Any] = None):
        metadata = create_tool_metadata(
            tool_id="graphql_client",
            name="GraphQL Client",
            description="GraphQL API client with query and mutation support",
            tool_type=ToolType.API_CLIENT,
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.RATE_LIMITED,
                ToolCapability.NETWORK_DEPENDENT
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "GraphQL endpoint URL"},
                    "query": {"type": "string", "description": "GraphQL query or mutation"},
                    "variables": {"type": "object", "description": "Query variables"},
                    "operation_name": {"type": "string", "description": "Operation name"},
                    "headers": {"type": "object", "description": "HTTP headers"},
                    "auth": {"type": "object", "description": "Authentication config"}
                },
                "required": ["endpoint", "query"]
            },
            tags=["graphql", "api", "query", "mutation"]
        )
        
        super().__init__(metadata, config)
    
    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute GraphQL request."""
        try:
            endpoint = request.parameters["endpoint"]
            query = request.parameters["query"]
            variables = request.parameters.get("variables", {})
            operation_name = request.parameters.get("operation_name")
            headers = request.parameters.get("headers", {})
            auth_config = request.parameters.get("auth")
            
            # Prepare request payload
            payload = {
                "query": query,
                "variables": variables
            }
            
            if operation_name:
                payload["operationName"] = operation_name
            
            # Set default headers
            headers.setdefault("Content-Type", "application/json")
            
            # Add authentication if provided
            if auth_config:
                rest_tool = RestApiTool()
                headers = await rest_tool._add_authentication(headers, auth_config, "POST", endpoint)
            
            start_time = time.time()
            
            # Make GraphQL request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers
                ) as response:
                    response_time = time.time() - start_time
                    response_data = await response.json()
                    
                    # Check for GraphQL errors
                    if "errors" in response_data:
                        error_messages = [error.get("message", str(error)) for error in response_data["errors"]]
                        return ToolResponse(
                            request_id=request.request_id,
                            tool_id=request.tool_id,
                            status=ToolStatus.FAILED,
                            error=f"GraphQL errors: {'; '.join(error_messages)}",
                            result=response_data
                        )
                    
                    return ToolResponse(
                        request_id=request.request_id,
                        tool_id=request.tool_id,
                        status=ToolStatus.COMPLETED,
                        result={
                            "data": response_data.get("data"),
                            "extensions": response_data.get("extensions"),
                            "response_time": response_time,
                            "status_code": response.status
                        },
                        metadata={
                            "operation_type": "mutation" if query.strip().startswith("mutation") else "query",
                            "has_variables": bool(variables)
                        }
                    )
            
        except Exception as e:
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=str(e)
            )

class WebhookTool(BaseTool):
    """Tool for sending webhook notifications."""
    
    def __init__(self, config: Dict[str, Any] = None):
        metadata = create_tool_metadata(
            tool_id="webhook_sender",
            name="Webhook Sender",
            description="Send HTTP webhook notifications with various payload formats",
            tool_type=ToolType.COMMUNICATION,
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.RATE_LIMITED,
                ToolCapability.NETWORK_DEPENDENT
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Webhook URL"},
                    "payload": {"type": "object", "description": "Webhook payload"},
                    "method": {"type": "string", "enum": ["POST", "PUT"], "default": "POST"},
                    "headers": {"type": "object", "description": "Custom headers"},
                    "secret": {"type": "string", "description": "Secret for signature generation"},
                    "signature_header": {"type": "string", "default": "X-Signature", "description": "Header name for signature"},
                    "timeout": {"type": "number", "default": 30},
                    "retry_count": {"type": "integer", "default": 3}
                },
                "required": ["url", "payload"]
            },
            tags=["webhook", "notification", "http", "integration"]
        )
        
        super().__init__(metadata, config)
    
    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute webhook request."""
        try:
            url = request.parameters["url"]
            payload = request.parameters["payload"]
            method = request.parameters.get("method", "POST")
            headers = request.parameters.get("headers", {})
            secret = request.parameters.get("secret")
            signature_header = request.parameters.get("signature_header", "X-Signature")
            timeout = request.parameters.get("timeout", 30)
            retry_count = request.parameters.get("retry_count", 3)
            
            # Set default headers
            headers.setdefault("Content-Type", "application/json")
            headers.setdefault("User-Agent", "WebhookTool/1.0")
            
            # Add timestamp
            payload["timestamp"] = datetime.now().isoformat()
            
            # Generate signature if secret provided
            if secret:
                payload_str = json.dumps(payload, separators=(',', ':'))
                signature = hmac.new(
                    secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers[signature_header] = f"sha256={signature}"
            
            # Attempt to send webhook with retries
            last_error = None
            
            for attempt in range(retry_count + 1):
                try:
                    start_time = time.time()
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.request(
                            method=method,
                            url=url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=timeout)
                        ) as response:
                            response_time = time.time() - start_time
                            response_text = await response.text()
                            
                            return ToolResponse(
                                request_id=request.request_id,
                                tool_id=request.tool_id,
                                status=ToolStatus.COMPLETED,
                                result={
                                    "status_code": response.status,
                                    "response": response_text,
                                    "response_time": response_time,
                                    "attempt": attempt + 1,
                                    "success": 200 <= response.status < 300
                                },
                                metadata={
                                    "webhook_url": url,
                                    "method": method,
                                    "has_signature": bool(secret)
                                }
                            )
                
                except Exception as e:
                    last_error = e
                    if attempt < retry_count:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        break
            
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=f"Webhook failed after {retry_count + 1} attempts: {last_error}"
            )
            
        except Exception as e:
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=str(e)
            )

class OpenAITool(BaseTool):
    """OpenAI API client tool."""
    
    def __init__(self, api_key: str = None, config: Dict[str, Any] = None):
        metadata = create_tool_metadata(
            tool_id="openai_client",
            name="OpenAI API Client",
            description="Client for OpenAI API services (Chat, Completions, Embeddings)",
            tool_type=ToolType.AI_MODEL,
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.REQUIRES_AUTH,
                ToolCapability.RATE_LIMITED,
                ToolCapability.NETWORK_DEPENDENT
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["chat", "completions", "embeddings", "images"],
                        "description": "OpenAI service to use"
                    },
                    "model": {"type": "string", "default": "gpt-3.5-turbo", "description": "Model to use"},
                    "messages": {"type": "array", "description": "Chat messages"},
                    "prompt": {"type": "string", "description": "Completion prompt"},
                    "text": {"type": "string", "description": "Text for embeddings"},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens"},
                    "temperature": {"type": "number", "description": "Temperature for randomness"},
                    "top_p": {"type": "number", "description": "Top-p sampling parameter"}
                },
                "required": ["service"]
            },
            rate_limit=60,
            requires_api_key=True,
            tags=["openai", "ai", "llm", "chat", "embeddings"]
        )
        
        super().__init__(metadata, config)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
    
    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute OpenAI API request."""
        try:
            if not self.api_key:
                return ToolResponse(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    status=ToolStatus.FAILED,
                    error="OpenAI API key not provided"
                )
            
            service = request.parameters["service"]
            
            if service == "chat":
                result = await self._chat_completion(request.parameters)
            elif service == "completions":
                result = await self._text_completion(request.parameters)
            elif service == "embeddings":
                result = await self._embeddings(request.parameters)
            else:
                return ToolResponse(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    status=ToolStatus.FAILED,
                    error=f"Unsupported service: {service}"
                )
            
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.COMPLETED,
                result=result,
                metadata={"service": service, "model": request.parameters.get("model")}
            )
            
        except Exception as e:
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _chat_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute chat completion."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": params.get("model", "gpt-3.5-turbo"),
            "messages": params.get("messages", [])
        }
        
        # Add optional parameters
        for key in ["max_tokens", "temperature", "top_p"]:
            if key in params:
                payload[key] = params[key]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                return await response.json()
    
    async def _text_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text completion."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": params.get("model", "gpt-3.5-turbo-instruct"),
            "prompt": params.get("prompt", "")
        }
        
        for key in ["max_tokens", "temperature", "top_p"]:
            if key in params:
                payload[key] = params[key]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/completions",
                json=payload,
                headers=headers
            ) as response:
                return await response.json()
    
    async def _embeddings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": params.get("model", "text-embedding-ada-002"),
            "input": params.get("text", "")
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers
            ) as response:
                return await response.json()

# Factory functions
def create_rest_api_tool(**config) -> RestApiTool:
    """Create a REST API client tool."""
    return RestApiTool(config=config)

def create_graphql_tool(**config) -> GraphQLTool:
    """Create a GraphQL client tool."""
    return GraphQLTool(config=config)

def create_webhook_tool(**config) -> WebhookTool:
    """Create a webhook sender tool."""
    return WebhookTool(config=config)

def create_openai_tool(api_key: str = None, **config) -> OpenAITool:
    """Create an OpenAI API client tool."""
    return OpenAITool(api_key=api_key, config=config)

# Example usage and testing
if __name__ == "__main__":
    async def demo():
        print("API Client Tools Demo")
        print("=" * 40)
        
        # Create tools
        rest_client = create_rest_api_tool()
        webhook_tool = create_webhook_tool()
        
        # Test REST API client
        api_request = ToolRequest(
            request_id="test_api_1",
            tool_id="rest_api_client",
            action="request",
            parameters={
                "url": "https://httpbin.org/json",
                "method": "GET",
                "headers": {"Accept": "application/json"}
            }
        )
        
        print("🌐 Testing REST API client...")
        api_response = await rest_client.execute(api_request)
        print(f"Status: {api_response.status.value}")
        
        if api_response.status == ToolStatus.COMPLETED:
            result = api_response.result
            print(f"HTTP Status: {result['status_code']}")
            print(f"Response time: {result['response_time']:.3f}s")
            print(f"Content type: {result['content_type']}")
        
        # Test webhook sender
        webhook_request = ToolRequest(
            request_id="test_webhook_1",
            tool_id="webhook_sender",
            action="send",
            parameters={
                "url": "https://httpbin.org/post",
                "payload": {
                    "event": "test_event",
                    "data": {"message": "Hello from webhook tool!"}
                },
                "headers": {"X-Custom-Header": "test"}
            }
        )
        
        print(f"\n🔗 Testing webhook sender...")
        webhook_response = await webhook_tool.execute(webhook_request)
        print(f"Status: {webhook_response.status.value}")
        
        if webhook_response.status == ToolStatus.COMPLETED:
            result = webhook_response.result
            print(f"Webhook Status: {result['status_code']}")
            print(f"Success: {result['success']}")
        
        print(f"\n✅ API Client Tools Demo completed!")
    
    asyncio.run(demo())
