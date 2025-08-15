"""
Tools System - Base Tool Framework
Provides foundation for all tool implementations and integrations.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ToolType(Enum):
    """Types of tools available in the system."""
    SEARCH = "search"
    WEB_SCRAPING = "web_scraping"
    API_CLIENT = "api_client"
    FILE_OPERATIONS = "file_operations"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    SECURITY = "security"
    HARDWARE = "hardware"
    AI_MODEL = "ai_model"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    CUSTOM = "custom"

class ToolStatus(Enum):
    """Tool execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class ToolCapability(Enum):
    """Tool capabilities and features."""
    ASYNC_EXECUTION = "async_execution"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME = "real_time"
    STATEFUL = "stateful"
    CACHEABLE = "cacheable"
    REQUIRES_AUTH = "requires_auth"
    RATE_LIMITED = "rate_limited"
    HARDWARE_DEPENDENT = "hardware_dependent"
    NETWORK_DEPENDENT = "network_dependent"

@dataclass
class ToolMetadata:
    """Comprehensive metadata for tools."""
    tool_id: str
    name: str
    description: str
    tool_type: ToolType
    version: str
    author: str
    capabilities: List[ToolCapability]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str] = None
    rate_limit: Optional[int] = None  # requests per minute
    timeout: float = 30.0  # seconds
    requires_api_key: bool = False
    supported_formats: List[str] = None
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.supported_formats is None:
            self.supported_formats = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class ToolRequest:
    """Request structure for tool execution."""
    request_id: str
    tool_id: str
    action: str
    parameters: Dict[str, Any]
    context: Dict[str, Any] = None
    priority: int = 1
    timeout: Optional[float] = None
    callback_url: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ToolResponse:
    """Response structure from tool execution."""
    request_id: str
    tool_id: str
    status: ToolStatus
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.completed_at is None:
            self.completed_at = datetime.now()

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, metadata: ToolMetadata, config: Dict[str, Any] = None):
        self.metadata = metadata
        self.config = config or {}
        self.status = ToolStatus.IDLE
        self.last_used = None
        self.usage_count = 0
        self.error_count = 0
        self._rate_limiter = None
        
        # Initialize rate limiter if needed
        if metadata.rate_limit:
            self._rate_limiter = RateLimiter(metadata.rate_limit)
    
    @abstractmethod
    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute the tool with given request."""
        pass
    
    async def validate_request(self, request: ToolRequest) -> bool:
        """Validate request against tool's input schema."""
        try:
            # Basic validation - can be extended with JSON schema validation
            required_params = self.metadata.input_schema.get('required', [])
            for param in required_params:
                if param not in request.parameters:
                    raise ValueError(f"Missing required parameter: {param}")
            
            return True
        except Exception as e:
            logger.error(f"Request validation failed for {self.metadata.tool_id}: {e}")
            return False
    
    async def check_rate_limit(self) -> bool:
        """Check if rate limit allows execution."""
        if self._rate_limiter:
            return await self._rate_limiter.can_proceed()
        return True
    
    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return self.metadata
    
    async def pre_execute(self, request: ToolRequest) -> bool:
        """Pre-execution checks and setup."""
        # Validate request
        if not await self.validate_request(request):
            return False
        
        # Check rate limits
        if not await self.check_rate_limit():
            return False
        
        # Update status
        self.status = ToolStatus.RUNNING
        return True
    
    async def post_execute(self, request: ToolRequest, response: ToolResponse):
        """Post-execution cleanup and logging."""
        self.status = ToolStatus.IDLE
        self.last_used = datetime.now()
        self.usage_count += 1
        
        if response.status == ToolStatus.FAILED:
            self.error_count += 1
        
        # Log execution
        logger.info(f"Tool {self.metadata.tool_id} executed: {response.status.value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            'tool_id': self.metadata.tool_id,
            'status': self.status.value,
            'usage_count': self.usage_count,
            'error_count': self.error_count,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'error_rate': self.error_count / max(self.usage_count, 1)
        }

class RateLimiter:
    """Simple rate limiter for tools."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    async def can_proceed(self) -> bool:
        """Check if request can proceed based on rate limit."""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        # Check if we can make another request
        if len(self.requests) < self.requests_per_minute:
            self.requests.append(now)
            return True
        
        return False

class ToolExecutor:
    """Manages tool execution with async capabilities."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.request_history: List[ToolRequest] = []
        self.response_history: List[ToolResponse] = []
        
    def register_tool(self, tool: BaseTool):
        """Register a tool for execution."""
        self.tools[tool.metadata.tool_id] = tool
        logger.info(f"Registered tool: {tool.metadata.tool_id}")
    
    def unregister_tool(self, tool_id: str):
        """Unregister a tool."""
        if tool_id in self.tools:
            del self.tools[tool_id]
            logger.info(f"Unregistered tool: {tool_id}")
    
    async def execute_tool(self, request: ToolRequest) -> ToolResponse:
        """Execute a tool request."""
        start_time = time.time()
        
        # Check if tool exists
        if request.tool_id not in self.tools:
            return ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=f"Tool not found: {request.tool_id}",
                execution_time=time.time() - start_time
            )
        
        tool = self.tools[request.tool_id]
        
        try:
            # Pre-execution checks
            if not await tool.pre_execute(request):
                return ToolResponse(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    status=ToolStatus.FAILED,
                    error="Pre-execution checks failed",
                    execution_time=time.time() - start_time
                )
            
            # Execute tool
            timeout = request.timeout or tool.metadata.timeout
            response = await asyncio.wait_for(
                tool.execute(request),
                timeout=timeout
            )
            
            # Update execution time
            response.execution_time = time.time() - start_time
            
            # Post-execution
            await tool.post_execute(request, response)
            
            # Store in history
            self.request_history.append(request)
            self.response_history.append(response)
            
            return response
            
        except asyncio.TimeoutError:
            response = ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.TIMEOUT,
                error=f"Tool execution timed out after {timeout}s",
                execution_time=time.time() - start_time
            )
            await tool.post_execute(request, response)
            return response
            
        except Exception as e:
            response = ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
            await tool.post_execute(request, response)
            return response
    
    async def execute_tool_async(self, request: ToolRequest) -> str:
        """Execute tool asynchronously and return task ID."""
        task_id = str(uuid.uuid4())
        
        async def execute_and_store():
            response = await self.execute_tool(request)
            # Store response for later retrieval
            return response
        
        task = asyncio.create_task(execute_and_store())
        self.active_requests[task_id] = task
        
        return task_id
    
    async def get_task_result(self, task_id: str) -> Optional[ToolResponse]:
        """Get result of async task."""
        if task_id not in self.active_requests:
            return None
        
        task = self.active_requests[task_id]
        
        if task.done():
            result = await task
            del self.active_requests[task_id]
            return result
        
        return None  # Still running
    
    def get_tool_metadata(self, tool_id: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool."""
        if tool_id in self.tools:
            return self.tools[tool_id].metadata
        return None
    
    def list_tools(self) -> Dict[str, ToolMetadata]:
        """List all registered tools."""
        return {tool_id: tool.metadata for tool_id, tool in self.tools.items()}
    
    def get_tools_by_type(self, tool_type: ToolType) -> Dict[str, ToolMetadata]:
        """Get tools of specific type."""
        return {
            tool_id: tool.metadata 
            for tool_id, tool in self.tools.items()
            if tool.metadata.tool_type == tool_type
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide tool execution statistics."""
        return {
            'total_tools': len(self.tools),
            'active_requests': len(self.active_requests),
            'total_requests': len(self.request_history),
            'total_responses': len(self.response_history),
            'tool_stats': {tool_id: tool.get_stats() for tool_id, tool in self.tools.items()}
        }

# Utility functions for creating common tool types
def create_tool_metadata(tool_id: str, name: str, description: str, 
                        tool_type: ToolType, **kwargs) -> ToolMetadata:
    """Create tool metadata with common defaults."""
    # Set defaults that can be overridden by kwargs
    defaults = {
        'version': "1.0.0",
        'author': "System",
        'capabilities': [],
        'input_schema': {},
        'output_schema': {}
    }
    
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    
    return ToolMetadata(
        tool_id=tool_id,
        name=name,
        description=description,
        tool_type=tool_type,
        **defaults
    )

def create_tool_request(tool_id: str, action: str, parameters: Dict[str, Any], 
                       **kwargs) -> ToolRequest:
    """Create a tool request with common defaults."""
    return ToolRequest(
        request_id=str(uuid.uuid4()),
        tool_id=tool_id,
        action=action,
        parameters=parameters,
        **kwargs
    )

# Example usage and testing
if __name__ == "__main__":
    async def demo():
        print("Tools System - Base Framework Demo")
        print("=" * 50)
        
        # Create tool executor
        executor = ToolExecutor()
        
        # Example tool implementation
        class EchoTool(BaseTool):
            async def execute(self, request: ToolRequest) -> ToolResponse:
                await asyncio.sleep(0.1)  # Simulate work
                
                return ToolResponse(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    status=ToolStatus.COMPLETED,
                    result={"echo": request.parameters.get("message", "Hello!")},
                    metadata={"processed": True}
                )
        
        # Create and register echo tool
        echo_metadata = create_tool_metadata(
            "echo_tool",
            "Echo Tool",
            "Simple echo tool for testing",
            ToolType.CUSTOM,
            input_schema={"properties": {"message": {"type": "string"}}},
            output_schema={"properties": {"echo": {"type": "string"}}}
        )
        
        echo_tool = EchoTool(echo_metadata)
        executor.register_tool(echo_tool)
        
        # Test tool execution
        request = create_tool_request(
            "echo_tool",
            "echo",
            {"message": "Hello, Tools System!"}
        )
        
        print(f"Executing tool request...")
        response = await executor.execute_tool(request)
        
        print(f"Status: {response.status.value}")
        print(f"Result: {response.result}")
        print(f"Execution time: {response.execution_time:.3f}s")
        
        # Show system stats
        stats = executor.get_system_stats()
        print(f"\nSystem stats:")
        print(f"Total tools: {stats['total_tools']}")
        print(f"Total requests: {stats['total_requests']}")
        
        print("\n✅ Base Tools Framework Demo completed!")
    
    asyncio.run(demo())
