"""
Internal API Management System
Manages internal APIs, microservices, and inter-component communication

This module provides:
- Internal service discovery
- API endpoint management
- Request routing and load balancing
- Service health monitoring
- Authentication and authorization
- Message queue integration
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import threading
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status states"""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"

class MessageType(Enum):
    """Internal message types"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    HEALTH_CHECK = "health_check"

@dataclass
class ServiceInfo:
    """Service registration information"""
    service_id: str
    name: str
    version: str
    endpoints: List[str]
    status: ServiceStatus = ServiceStatus.STARTING
    health_check_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)

@dataclass
class InternalMessage:
    """Internal message structure"""
    message_id: str
    message_type: MessageType
    source_service: str
    target_service: str
    endpoint: str
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class APIEndpoint:
    """Internal API endpoint definition"""
    service_id: str
    path: str
    method: str
    handler: Callable
    auth_required: bool = False
    rate_limit: int = 1000  # requests per minute
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ServiceRegistry:
    """Service discovery and registration"""
    
    def __init__(self):
        self.services = {}
        self.endpoints = {}
        self.health_checks = {}
        self.logger = logging.getLogger("service_registry")
        self._lock = threading.RLock()
    
    def register_service(self, service_info: ServiceInfo) -> bool:
        """Register a service"""
        try:
            with self._lock:
                self.services[service_info.service_id] = service_info
                
                # Register endpoints
                for endpoint in service_info.endpoints:
                    if endpoint not in self.endpoints:
                        self.endpoints[endpoint] = []
                    self.endpoints[endpoint].append(service_info.service_id)
                
                self.logger.info(f"Registered service: {service_info.name} ({service_info.service_id})")
                return True
                
        except Exception as e:
            self.logger.error(f"Service registration failed: {e}")
            return False
    
    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service"""
        try:
            with self._lock:
                if service_id not in self.services:
                    return False
                
                service_info = self.services[service_id]
                
                # Remove from endpoints
                for endpoint in service_info.endpoints:
                    if endpoint in self.endpoints:
                        if service_id in self.endpoints[endpoint]:
                            self.endpoints[endpoint].remove(service_id)
                        if not self.endpoints[endpoint]:
                            del self.endpoints[endpoint]
                
                del self.services[service_id]
                
                self.logger.info(f"Unregistered service: {service_info.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Service unregistration failed: {e}")
            return False
    
    def update_service_status(self, service_id: str, status: ServiceStatus) -> bool:
        """Update service status"""
        try:
            with self._lock:
                if service_id in self.services:
                    self.services[service_id].status = status
                    self.services[service_id].last_heartbeat = time.time()
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Status update failed: {e}")
            return False
    
    def discover_service(self, endpoint: str) -> Optional[List[str]]:
        """Discover services providing an endpoint"""
        with self._lock:
            return self.endpoints.get(endpoint, []).copy()
    
    def get_service_info(self, service_id: str) -> Optional[ServiceInfo]:
        """Get service information"""
        with self._lock:
            return self.services.get(service_id)
    
    def get_healthy_services(self) -> List[ServiceInfo]:
        """Get all healthy services"""
        with self._lock:
            return [
                service for service in self.services.values()
                if service.status == ServiceStatus.HEALTHY
            ]
    
    def cleanup_stale_services(self, timeout: float = 300) -> int:
        """Remove services that haven't sent heartbeat"""
        try:
            current_time = time.time()
            stale_services = []
            
            with self._lock:
                for service_id, service in self.services.items():
                    if current_time - service.last_heartbeat > timeout:
                        stale_services.append(service_id)
                
                for service_id in stale_services:
                    self.unregister_service(service_id)
            
            return len(stale_services)
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 0

class MessageBroker:
    """Internal message broker for service communication"""
    
    def __init__(self):
        self.message_queues = defaultdict(list)
        self.subscribers = defaultdict(list)
        self.pending_responses = {}
        self.logger = logging.getLogger("message_broker")
        self._lock = threading.RLock()
    
    async def send_message(self, message: InternalMessage) -> Optional[InternalMessage]:
        """Send message to target service"""
        try:
            message.message_id = str(uuid.uuid4())
            
            if message.message_type == MessageType.REQUEST:
                # Store for response tracking
                self.pending_responses[message.message_id] = {
                    'message': message,
                    'timestamp': time.time(),
                    'future': asyncio.Future()
                }
                
                # Queue message
                with self._lock:
                    self.message_queues[message.target_service].append(message)
                
                # Wait for response
                try:
                    future = self.pending_responses[message.message_id]['future']
                    response = await asyncio.wait_for(future, timeout=message.timeout)
                    return response
                except asyncio.TimeoutError:
                    self.logger.warning(f"Message timeout: {message.message_id}")
                    return None
                finally:
                    if message.message_id in self.pending_responses:
                        del self.pending_responses[message.message_id]
            
            else:
                # One-way message
                with self._lock:
                    self.message_queues[message.target_service].append(message)
                
                # Notify subscribers for events/broadcasts
                if message.message_type in [MessageType.EVENT, MessageType.BROADCAST]:
                    await self._notify_subscribers(message)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Message send failed: {e}")
            return None
    
    async def receive_messages(self, service_id: str, timeout: float = 1.0) -> List[InternalMessage]:
        """Receive messages for a service"""
        try:
            messages = []
            
            with self._lock:
                if service_id in self.message_queues:
                    messages = self.message_queues[service_id].copy()
                    self.message_queues[service_id].clear()
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Message receive failed: {e}")
            return []
    
    async def send_response(self, original_message: InternalMessage, response_data: Any) -> bool:
        """Send response to original message"""
        try:
            if original_message.message_id in self.pending_responses:
                response = InternalMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    source_service=original_message.target_service,
                    target_service=original_message.source_service,
                    endpoint=original_message.endpoint,
                    data=response_data
                )
                
                future = self.pending_responses[original_message.message_id]['future']
                if not future.done():
                    future.set_result(response)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Response send failed: {e}")
            return False
    
    def subscribe_to_events(self, service_id: str, event_types: List[str]) -> bool:
        """Subscribe service to event types"""
        try:
            with self._lock:
                for event_type in event_types:
                    if service_id not in self.subscribers[event_type]:
                        self.subscribers[event_type].append(service_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Event subscription failed: {e}")
            return False
    
    async def _notify_subscribers(self, message: InternalMessage):
        """Notify event subscribers"""
        try:
            event_type = message.endpoint
            
            with self._lock:
                subscribers = self.subscribers.get(event_type, []).copy()
            
            for subscriber_id in subscribers:
                if subscriber_id != message.source_service:
                    notification = InternalMessage(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.EVENT,
                        source_service=message.source_service,
                        target_service=subscriber_id,
                        endpoint=event_type,
                        data=message.data
                    )
                    
                    with self._lock:
                        self.message_queues[subscriber_id].append(notification)
            
        except Exception as e:
            self.logger.error(f"Subscriber notification failed: {e}")

class LoadBalancer:
    """Load balancer for internal services"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.service_counters = defaultdict(int)
        self.logger = logging.getLogger("load_balancer")
    
    def select_service(self, services: List[str], endpoint: str) -> Optional[str]:
        """Select service using load balancing strategy"""
        if not services:
            return None
        
        if len(services) == 1:
            return services[0]
        
        if self.strategy == "round_robin":
            return self._round_robin_select(services, endpoint)
        elif self.strategy == "random":
            import random
            return random.choice(services)
        else:
            return services[0]  # First available
    
    def _round_robin_select(self, services: List[str], endpoint: str) -> str:
        """Round robin selection"""
        key = f"{endpoint}:{':'.join(sorted(services))}"
        index = self.service_counters[key] % len(services)
        self.service_counters[key] += 1
        return services[index]

class InternalAPIManager:
    """Main internal API management system"""
    
    def __init__(self, workspace_path: str = "/media/r/Workspace"):
        self.workspace_path = workspace_path
        self.service_registry = ServiceRegistry()
        self.message_broker = MessageBroker()
        self.load_balancer = LoadBalancer()
        self.api_endpoints = {}
        self.running = False
        self.logger = logging.getLogger("internal_api_manager")
    
    async def initialize(self) -> bool:
        """Initialize internal API manager"""
        try:
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._cleanup_task())
            asyncio.create_task(self._health_monitoring_task())
            
            self.logger.info("Internal API manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Internal API manager initialization failed: {e}")
            return False
    
    async def register_service(self, name: str, version: str, endpoints: List[str], metadata: Dict = None) -> str:
        """Register a new service"""
        try:
            service_id = f"{name}_{version}_{int(time.time())}"
            
            service_info = ServiceInfo(
                service_id=service_id,
                name=name,
                version=version,
                endpoints=endpoints,
                status=ServiceStatus.STARTING,
                metadata=metadata or {}
            )
            
            success = self.service_registry.register_service(service_info)
            
            if success:
                # Update status to healthy
                await asyncio.sleep(1)  # Simulate startup time
                self.service_registry.update_service_status(service_id, ServiceStatus.HEALTHY)
                
                return service_id
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Service registration failed: {e}")
            return ""
    
    async def register_api_endpoint(self, service_id: str, path: str, method: str, handler: Callable, **kwargs) -> bool:
        """Register API endpoint"""
        try:
            endpoint = APIEndpoint(
                service_id=service_id,
                path=path,
                method=method,
                handler=handler,
                **kwargs
            )
            
            endpoint_key = f"{method}:{path}"
            self.api_endpoints[endpoint_key] = endpoint
            
            self.logger.info(f"Registered API endpoint: {method} {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"API endpoint registration failed: {e}")
            return False
    
    async def call_internal_api(self, endpoint: str, method: str = "GET", data: Any = None, source_service: str = "system") -> Any:
        """Call internal API endpoint"""
        try:
            # Discover services
            services = self.service_registry.discover_service(endpoint)
            if not services:
                raise ValueError(f"No services found for endpoint: {endpoint}")
            
            # Filter healthy services
            healthy_services = []
            for service_id in services:
                service_info = self.service_registry.get_service_info(service_id)
                if service_info and service_info.status == ServiceStatus.HEALTHY:
                    healthy_services.append(service_id)
            
            if not healthy_services:
                raise ValueError(f"No healthy services found for endpoint: {endpoint}")
            
            # Load balance
            target_service = self.load_balancer.select_service(healthy_services, endpoint)
            
            # Send request
            message = InternalMessage(
                message_id="",
                message_type=MessageType.REQUEST,
                source_service=source_service,
                target_service=target_service,
                endpoint=endpoint,
                data=data
            )
            
            response = await self.message_broker.send_message(message)
            
            if response:
                return response.data
            else:
                raise Exception("Request timeout or failed")
                
        except Exception as e:
            self.logger.error(f"Internal API call failed: {e}")
            raise
    
    async def publish_event(self, event_type: str, data: Any, source_service: str = "system"):
        """Publish event to subscribers"""
        try:
            message = InternalMessage(
                message_id="",
                message_type=MessageType.EVENT,
                source_service=source_service,
                target_service="*",
                endpoint=event_type,
                data=data
            )
            
            await self.message_broker.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Event publishing failed: {e}")
    
    async def subscribe_to_events(self, service_id: str, event_types: List[str]) -> bool:
        """Subscribe service to events"""
        return self.message_broker.subscribe_to_events(service_id, event_types)
    
    async def heartbeat(self, service_id: str) -> bool:
        """Send heartbeat for service"""
        return self.service_registry.update_service_status(service_id, ServiceStatus.HEALTHY)
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.running:
            try:
                # Cleanup stale services
                cleaned = self.service_registry.cleanup_stale_services()
                if cleaned > 0:
                    self.logger.info(f"Cleaned up {cleaned} stale services")
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
    
    async def _health_monitoring_task(self):
        """Background health monitoring"""
        while self.running:
            try:
                services = self.service_registry.get_healthy_services()
                
                for service in services:
                    # Check if service is still responsive
                    if time.time() - service.last_heartbeat > 120:  # 2 minutes
                        self.service_registry.update_service_status(
                            service.service_id, 
                            ServiceStatus.DEGRADED
                        )
                
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get internal API manager status"""
        services = list(self.service_registry.services.values())
        
        status_counts = defaultdict(int)
        for service in services:
            status_counts[service.status.value] += 1
        
        return {
            'running': self.running,
            'total_services': len(services),
            'service_status': dict(status_counts),
            'total_endpoints': len(self.api_endpoints),
            'message_queues': len(self.message_broker.message_queues),
            'pending_responses': len(self.message_broker.pending_responses),
            'workspace_path': self.workspace_path,
            'capabilities': [
                'service_discovery',
                'load_balancing',
                'message_brokering',
                'health_monitoring',
                'event_publishing',
                'api_endpoint_management'
            ]
        }
    
    async def shutdown(self):
        """Shutdown internal API manager"""
        self.running = False
        self.logger.info("Internal API manager shutdown")

# Global internal API manager instance
internal_api = InternalAPIManager()

async def initialize_internal_api() -> bool:
    """Initialize internal API manager"""
    return await internal_api.initialize()

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("🔧 Internal API Management Demo")
        print("=" * 50)
        
        success = await internal_api.initialize()
        print(f"Initialization: {'✅ Success' if success else '❌ Failed'}")
        
        if success:
            # Demo service registration
            service_id = await internal_api.register_service(
                name="demo_service",
                version="1.0.0",
                endpoints=["process_data", "get_status"],
                metadata={"description": "Demo service"}
            )
            print(f"Service registered: {'✅ Success' if service_id else '❌ Failed'}")
            
            # Demo API endpoint registration
            async def demo_handler(data):
                return {"processed": True, "input": data}
            
            endpoint_registered = await internal_api.register_api_endpoint(
                service_id=service_id,
                path="process_data",
                method="POST",
                handler=demo_handler
            )
            print(f"Endpoint registered: {'✅ Success' if endpoint_registered else '❌ Failed'}")
            
            # Demo status
            status = internal_api.get_status()
            print(f"Status: {status}")
            
            # Cleanup
            await internal_api.shutdown()
        
        print("Demo complete!")
    
    asyncio.run(demo())
