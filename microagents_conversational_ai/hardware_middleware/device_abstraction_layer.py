"""
Device Abstraction Layer - Universal hardware interface with AI integration
Provides a unified interface for all hardware devices with AI orchestration.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class DeviceStatus(Enum):
    """Device connection and operational status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class DeviceCapability(Enum):
    """Standard device capabilities"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    MONITOR = "monitor"
    CONFIGURE = "configure"
    AI_PROCESSING = "ai_processing"
    DATA_STORAGE = "data_storage"
    REAL_TIME = "real_time"

@dataclass
class DeviceCommand:
    """Standardized device command structure"""
    command_id: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    priority: int = 1
    requires_ai: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceResponse:
    """Standardized device response structure"""
    command_id: str
    status: str  # 'success', 'error', 'timeout', 'pending'
    data: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    ai_processed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class AbstractDevice(ABC):
    """Abstract base class for all hardware devices"""
    
    def __init__(self, device_id: str, device_name: str, device_type: str):
        self.device_id = device_id
        self.device_name = device_name
        self.device_type = device_type
        self.status = DeviceStatus.DISCONNECTED
        self.capabilities: List[DeviceCapability] = []
        self.metadata: Dict[str, Any] = {}
        
        # AI integration
        self.ai_orchestrator = None
        self.ai_enabled = False
        
        # Command tracking
        self.active_commands: Dict[str, DeviceCommand] = {}
        self.command_history: List[DeviceResponse] = []
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'status_changed': [],
            'data_received': [],
            'error_occurred': [],
            'command_completed': []
        }
        
        logger.info(f"Created abstract device: {device_name} ({device_type})")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the device"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the device"""
        pass
    
    @abstractmethod
    async def execute_command(self, command: DeviceCommand) -> DeviceResponse:
        """Execute a command on the device"""
        pass
    
    @abstractmethod
    async def read_data(self) -> Any:
        """Read data from the device"""
        pass
    
    @abstractmethod
    async def write_data(self, data: Any) -> bool:
        """Write data to the device"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        pass
    
    def add_capability(self, capability: DeviceCapability):
        """Add a capability to the device"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def has_capability(self, capability: DeviceCapability) -> bool:
        """Check if device has a specific capability"""
        return capability in self.capabilities
    
    def set_ai_orchestrator(self, ai_orchestrator):
        """Set AI orchestrator for enhanced capabilities"""
        self.ai_orchestrator = ai_orchestrator
        self.ai_enabled = ai_orchestrator is not None
        logger.info(f"AI orchestrator {'enabled' if self.ai_enabled else 'disabled'} for {self.device_name}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for device events"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Any):
        """Emit an event to all registered handlers"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self, event_type, data)
                else:
                    handler(self, event_type, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def _change_status(self, new_status: DeviceStatus):
        """Change device status and emit event"""
        old_status = self.status
        self.status = new_status
        logger.info(f"Device {self.device_name} status changed: {old_status.value} -> {new_status.value}")
        await self._emit_event('status_changed', {'old_status': old_status, 'new_status': new_status})
    
    async def _process_with_ai(self, command: DeviceCommand) -> Optional[Dict[str, Any]]:
        """Process command with AI orchestrator if available"""
        if not self.ai_enabled or not command.requires_ai:
            return None
        
        try:
            ai_prompt = [
                f"Device command analysis for {self.device_name}:",
                f"Command: {command.action}",
                f"Parameters: {json.dumps(command.parameters, indent=2)}",
                f"Device capabilities: {[cap.value for cap in self.capabilities]}",
                "Optimize this command execution and suggest improvements."
            ]
            
            ai_result = await self.ai_orchestrator.process_conversation(
                ai_prompt,
                session_id=f"device_command_{self.device_id}_{command.command_id}"
            )
            
            if ai_result.get('status') == 'success':
                return {
                    'ai_optimizations': ai_result.get('response', {}).get('insights', []),
                    'suggested_parameters': ai_result.get('response', {}).get('recommendations', []),
                    'risk_assessment': ai_result.get('response', {}).get('summary', '')
                }
        
        except Exception as e:
            logger.error(f"Error in AI processing for device {self.device_name}: {e}")
        
        return None

class USBDevice(AbstractDevice):
    """USB device implementation"""
    
    def __init__(self, device_id: str, device_name: str, vendor_id: str = "", product_id: str = ""):
        super().__init__(device_id, device_name, "usb")
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.usb_handle = None
        
        # Add standard USB capabilities
        self.add_capability(DeviceCapability.READ)
        self.add_capability(DeviceCapability.WRITE)
        self.add_capability(DeviceCapability.CONFIGURE)
    
    async def connect(self) -> bool:
        """Connect to USB device"""
        try:
            await self._change_status(DeviceStatus.CONNECTING)
            
            # Simulate USB connection (in production would use actual USB libraries)
            await asyncio.sleep(0.5)
            
            # Check if device exists
            if self._check_usb_device_exists():
                await self._change_status(DeviceStatus.CONNECTED)
                logger.info(f"Connected to USB device: {self.device_name}")
                return True
            else:
                await self._change_status(DeviceStatus.ERROR)
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to USB device {self.device_name}: {e}")
            await self._change_status(DeviceStatus.ERROR)
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from USB device"""
        try:
            if self.usb_handle:
                # Close USB handle
                self.usb_handle = None
            
            await self._change_status(DeviceStatus.DISCONNECTED)
            logger.info(f"Disconnected from USB device: {self.device_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting USB device {self.device_name}: {e}")
            return False
    
    def _check_usb_device_exists(self) -> bool:
        """Check if USB device exists (simulation)"""
        # In production, would check actual USB device availability
        return True
    
    async def execute_command(self, command: DeviceCommand) -> DeviceResponse:
        """Execute command on USB device"""
        start_time = time.time()
        
        try:
            # Process with AI if requested
            ai_result = await self._process_with_ai(command)
            
            # Track active command
            self.active_commands[command.command_id] = command
            
            # Simulate command execution
            if command.action == "read":
                data = await self._usb_read(command.parameters)
            elif command.action == "write":
                data = await self._usb_write(command.parameters)
            elif command.action == "configure":
                data = await self._usb_configure(command.parameters)
            else:
                raise ValueError(f"Unknown USB command: {command.action}")
            
            # Create response
            response = DeviceResponse(
                command_id=command.command_id,
                status="success",
                data=data,
                execution_time=time.time() - start_time,
                ai_processed=ai_result is not None,
                metadata={'ai_result': ai_result} if ai_result else {}
            )
            
            # Cleanup
            del self.active_commands[command.command_id]
            self.command_history.append(response)
            
            await self._emit_event('command_completed', response)
            return response
            
        except Exception as e:
            error_response = DeviceResponse(
                command_id=command.command_id,
                status="error",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
            
            if command.command_id in self.active_commands:
                del self.active_commands[command.command_id]
            
            await self._emit_event('error_occurred', error_response)
            return error_response
    
    async def _usb_read(self, parameters: Dict[str, Any]) -> Any:
        """Read from USB device"""
        # Simulate USB read operation
        await asyncio.sleep(0.1)
        return {"usb_data": "simulated_read_data", "bytes_read": 1024}
    
    async def _usb_write(self, parameters: Dict[str, Any]) -> Any:
        """Write to USB device"""
        # Simulate USB write operation
        data = parameters.get('data', b'')
        await asyncio.sleep(0.1)
        return {"bytes_written": len(data) if isinstance(data, (bytes, str)) else 0}
    
    async def _usb_configure(self, parameters: Dict[str, Any]) -> Any:
        """Configure USB device"""
        # Simulate USB configuration
        await asyncio.sleep(0.2)
        return {"configuration": "applied", "parameters": parameters}
    
    async def read_data(self) -> Any:
        """Read data from USB device"""
        command = DeviceCommand(
            command_id=f"read_{int(time.time() * 1000)}",
            action="read"
        )
        response = await self.execute_command(command)
        return response.data if response.status == "success" else None
    
    async def write_data(self, data: Any) -> bool:
        """Write data to USB device"""
        command = DeviceCommand(
            command_id=f"write_{int(time.time() * 1000)}",
            action="write",
            parameters={"data": data}
        )
        response = await self.execute_command(command)
        return response.status == "success"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get USB device status"""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "status": self.status.value,
            "vendor_id": self.vendor_id,
            "product_id": self.product_id,
            "capabilities": [cap.value for cap in self.capabilities],
            "active_commands": len(self.active_commands),
            "ai_enabled": self.ai_enabled
        }

class NetworkDevice(AbstractDevice):
    """Network device implementation"""
    
    def __init__(self, device_id: str, device_name: str, ip_address: str = "", port: int = 80):
        super().__init__(device_id, device_name, "network")
        self.ip_address = ip_address
        self.port = port
        self.connection = None
        
        # Add network capabilities
        self.add_capability(DeviceCapability.READ)
        self.add_capability(DeviceCapability.WRITE)
        self.add_capability(DeviceCapability.MONITOR)
        self.add_capability(DeviceCapability.REAL_TIME)
    
    async def connect(self) -> bool:
        """Connect to network device"""
        try:
            await self._change_status(DeviceStatus.CONNECTING)
            
            # Simulate network connection
            await asyncio.sleep(1.0)
            
            if self._check_network_connectivity():
                await self._change_status(DeviceStatus.CONNECTED)
                logger.info(f"Connected to network device: {self.device_name} at {self.ip_address}:{self.port}")
                return True
            else:
                await self._change_status(DeviceStatus.ERROR)
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to network device {self.device_name}: {e}")
            await self._change_status(DeviceStatus.ERROR)
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from network device"""
        try:
            if self.connection:
                self.connection = None
            
            await self._change_status(DeviceStatus.DISCONNECTED)
            logger.info(f"Disconnected from network device: {self.device_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting network device {self.device_name}: {e}")
            return False
    
    def _check_network_connectivity(self) -> bool:
        """Check network connectivity (simulation)"""
        # In production, would ping or test actual connection
        return bool(self.ip_address)
    
    async def execute_command(self, command: DeviceCommand) -> DeviceResponse:
        """Execute command on network device"""
        start_time = time.time()
        
        try:
            # Process with AI if requested
            ai_result = await self._process_with_ai(command)
            
            # Track active command
            self.active_commands[command.command_id] = command
            
            # Simulate network command execution
            if command.action == "ping":
                data = await self._network_ping(command.parameters)
            elif command.action == "request":
                data = await self._network_request(command.parameters)
            elif command.action == "monitor":
                data = await self._network_monitor(command.parameters)
            else:
                raise ValueError(f"Unknown network command: {command.action}")
            
            # Create response
            response = DeviceResponse(
                command_id=command.command_id,
                status="success",
                data=data,
                execution_time=time.time() - start_time,
                ai_processed=ai_result is not None,
                metadata={'ai_result': ai_result} if ai_result else {}
            )
            
            # Cleanup
            del self.active_commands[command.command_id]
            self.command_history.append(response)
            
            await self._emit_event('command_completed', response)
            return response
            
        except Exception as e:
            error_response = DeviceResponse(
                command_id=command.command_id,
                status="error",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
            
            if command.command_id in self.active_commands:
                del self.active_commands[command.command_id]
            
            await self._emit_event('error_occurred', error_response)
            return error_response
    
    async def _network_ping(self, parameters: Dict[str, Any]) -> Any:
        """Ping network device"""
        await asyncio.sleep(0.1)
        return {"ping_success": True, "latency_ms": 15, "packet_loss": 0}
    
    async def _network_request(self, parameters: Dict[str, Any]) -> Any:
        """Make network request"""
        await asyncio.sleep(0.3)
        return {"response_code": 200, "data": "simulated_response", "headers": {}}
    
    async def _network_monitor(self, parameters: Dict[str, Any]) -> Any:
        """Monitor network device"""
        await asyncio.sleep(0.5)
        return {"bandwidth_usage": "10.5 Mbps", "active_connections": 42, "status": "healthy"}
    
    async def read_data(self) -> Any:
        """Read data from network device"""
        command = DeviceCommand(
            command_id=f"read_{int(time.time() * 1000)}",
            action="request",
            parameters={"method": "GET", "endpoint": "/status"}
        )
        response = await self.execute_command(command)
        return response.data if response.status == "success" else None
    
    async def write_data(self, data: Any) -> bool:
        """Send data to network device"""
        command = DeviceCommand(
            command_id=f"write_{int(time.time() * 1000)}",
            action="request",
            parameters={"method": "POST", "data": data}
        )
        response = await self.execute_command(command)
        return response.status == "success"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get network device status"""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "status": self.status.value,
            "ip_address": self.ip_address,
            "port": self.port,
            "capabilities": [cap.value for cap in self.capabilities],
            "active_commands": len(self.active_commands),
            "ai_enabled": self.ai_enabled
        }

class DeviceAbstractionLayer:
    """Main device abstraction layer with AI orchestration"""
    
    def __init__(self, ai_orchestrator=None):
        self.ai_orchestrator = ai_orchestrator
        self.devices: Dict[str, AbstractDevice] = {}
        self.device_factories: Dict[str, Callable] = {
            'usb': self._create_usb_device,
            'network': self._create_network_device
        }
        
        logger.info(f"Device Abstraction Layer initialized (AI integration: {ai_orchestrator is not None})")
    
    def register_device_factory(self, device_type: str, factory: Callable):
        """Register a device factory for a specific type"""
        self.device_factories[device_type] = factory
    
    async def create_device(self, device_type: str, device_id: str, device_name: str, **kwargs) -> Optional[AbstractDevice]:
        """Create a device of the specified type"""
        try:
            if device_type not in self.device_factories:
                logger.error(f"Unknown device type: {device_type}")
                return None
            
            device = self.device_factories[device_type](device_id, device_name, **kwargs)
            
            # Set AI orchestrator if available
            if self.ai_orchestrator:
                device.set_ai_orchestrator(self.ai_orchestrator)
            
            self.devices[device_id] = device
            logger.info(f"Created device: {device_name} ({device_type})")
            return device
            
        except Exception as e:
            logger.error(f"Error creating device {device_name}: {e}")
            return None
    
    def _create_usb_device(self, device_id: str, device_name: str, **kwargs) -> USBDevice:
        """Create USB device"""
        return USBDevice(
            device_id=device_id,
            device_name=device_name,
            vendor_id=kwargs.get('vendor_id', ''),
            product_id=kwargs.get('product_id', '')
        )
    
    def _create_network_device(self, device_id: str, device_name: str, **kwargs) -> NetworkDevice:
        """Create network device"""
        return NetworkDevice(
            device_id=device_id,
            device_name=device_name,
            ip_address=kwargs.get('ip_address', ''),
            port=kwargs.get('port', 80)
        )
    
    def get_device(self, device_id: str) -> Optional[AbstractDevice]:
        """Get device by ID"""
        return self.devices.get(device_id)
    
    def get_all_devices(self) -> Dict[str, AbstractDevice]:
        """Get all registered devices"""
        return self.devices.copy()
    
    def get_devices_by_type(self, device_type: str) -> List[AbstractDevice]:
        """Get devices by type"""
        return [device for device in self.devices.values() if device.device_type == device_type]
    
    def get_devices_by_status(self, status: DeviceStatus) -> List[AbstractDevice]:
        """Get devices by status"""
        return [device for device in self.devices.values() if device.status == status]
    
    async def connect_all_devices(self) -> Dict[str, bool]:
        """Connect all registered devices"""
        results = {}
        for device_id, device in self.devices.items():
            try:
                results[device_id] = await device.connect()
            except Exception as e:
                logger.error(f"Error connecting device {device_id}: {e}")
                results[device_id] = False
        return results
    
    async def disconnect_all_devices(self) -> Dict[str, bool]:
        """Disconnect all registered devices"""
        results = {}
        for device_id, device in self.devices.items():
            try:
                results[device_id] = await device.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting device {device_id}: {e}")
                results[device_id] = False
        return results
    
    async def execute_command_on_device(self, device_id: str, command: DeviceCommand) -> Optional[DeviceResponse]:
        """Execute command on specific device"""
        device = self.get_device(device_id)
        if not device:
            logger.error(f"Device not found: {device_id}")
            return None
        
        return await device.execute_command(command)
    
    def remove_device(self, device_id: str) -> bool:
        """Remove device from registry"""
        if device_id in self.devices:
            del self.devices[device_id]
            logger.info(f"Removed device: {device_id}")
            return True
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status_counts = {}
        for device in self.devices.values():
            status = device.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_devices': len(self.devices),
            'status_breakdown': status_counts,
            'ai_integration_enabled': self.ai_orchestrator is not None,
            'device_types': list(set(device.device_type for device in self.devices.values()))
        }