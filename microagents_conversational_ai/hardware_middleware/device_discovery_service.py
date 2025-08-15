"""
Device Discovery Service - Enhanced version with Universal AI System integration
Auto-detection and hotplug monitoring with AI orchestration capabilities.
"""

import time
import logging
import asyncio
from typing import Dict, List, Callable, Optional, Set, Any
import json
from dataclasses import dataclass
from enum import Enum

# Optional imports with graceful fallback
try:
    import pyudev
    HAS_PYUDEV = True
except ImportError:
    HAS_PYUDEV = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Enhanced device type classification"""
    USB = "usb"
    SERIAL = "serial"
    NETWORK = "network"
    BLUETOOTH = "bluetooth"
    AI_ACCELERATOR = "ai_accelerator"
    STORAGE = "storage"
    DISPLAY = "display"
    AUDIO = "audio"
    CAMERA = "camera"
    SENSOR = "sensor"
    UNKNOWN = "unknown"

@dataclass
class EnhancedDeviceInfo:
    """Enhanced device information with AI system integration"""
    device_id: str
    device_type: DeviceType
    name: str
    vendor: str = "Unknown"
    model: str = "Unknown"
    capabilities: List[str] = None
    ai_compatible: bool = False
    connection_status: str = "disconnected"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.metadata is None:
            self.metadata = {}

class DeviceDiscoveryEvent:
    """Enhanced device discovery event with AI system integration"""
    
    def __init__(self, event_type: str, device_info: EnhancedDeviceInfo, timestamp: float = None):
        self.event_type = event_type  # 'connected', 'disconnected', 'changed', 'ai_ready'
        self.device_info = device_info
        self.timestamp = timestamp or time.time()
        self.processed_by_ai = False
    
    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type,
            'device_info': {
                'device_id': self.device_info.device_id,
                'device_type': self.device_info.device_type.value,
                'name': self.device_info.name,
                'vendor': self.device_info.vendor,
                'model': self.device_info.model,
                'capabilities': self.device_info.capabilities,
                'ai_compatible': self.device_info.ai_compatible,
                'connection_status': self.device_info.connection_status,
                'metadata': self.device_info.metadata
            },
            'timestamp': self.timestamp,
            'processed_by_ai': self.processed_by_ai
        }

class EnhancedDeviceDiscoveryService:
    """Enhanced Device Discovery Service with Universal AI System integration"""
    
    def __init__(self, device_manager=None, message_bus=None, ai_orchestrator=None):
        self.device_manager = device_manager
        self.message_bus = message_bus
        self.ai_orchestrator = ai_orchestrator  # Integration with Universal AI System
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'connected': [],
            'disconnected': [],
            'changed': [],
            'ai_ready': []
        }
        
        # Discovered devices registry
        self.discovered_devices: Dict[str, EnhancedDeviceInfo] = {}
        
        # AI integration state
        self.ai_integration_enabled = ai_orchestrator is not None
        self.ai_device_analysis_queue = asyncio.Queue() if self.ai_integration_enabled else None
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task = None
        self._ai_analysis_task = None
        
        logger.info(f"Enhanced Device Discovery Service initialized (AI integration: {self.ai_integration_enabled})")
    
    def add_event_handler(self, event_type: str, handler: Callable[[DeviceDiscoveryEvent], None]):
        """Add event handler for device discovery events"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def remove_event_handler(self, event_type: str, handler: Callable[[DeviceDiscoveryEvent], None]):
        """Remove event handler"""
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def start(self):
        """Start device discovery monitoring with AI integration"""
        if self._monitoring:
            logger.warning("Device discovery already running")
            return
        
        self._monitoring = True
        logger.info("Starting enhanced device discovery service...")
        
        # Start device monitoring
        self._monitor_task = asyncio.create_task(self._monitor_devices())
        
        # Start AI analysis if enabled
        if self.ai_integration_enabled:
            self._ai_analysis_task = asyncio.create_task(self._ai_device_analysis())
        
        # Initial device scan
        await self._perform_initial_scan()
    
    async def stop(self):
        """Stop device discovery monitoring"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        logger.info("Stopping device discovery service...")
        
        # Cancel monitoring tasks
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._ai_analysis_task:
            self._ai_analysis_task.cancel()
            try:
                await self._ai_analysis_task
            except asyncio.CancelledError:
                pass
    
    async def _perform_initial_scan(self):
        """Perform initial device scan"""
        logger.info("Performing initial device scan...")
        
        try:
            # Scan USB devices
            usb_devices = self._scan_usb_devices()
            for device in usb_devices:
                await self._handle_device_event('connected', device)
            
            # Scan network devices
            network_devices = self._scan_network_devices()
            for device in network_devices:
                await self._handle_device_event('connected', device)
            
            # Scan storage devices
            storage_devices = self._scan_storage_devices()
            for device in storage_devices:
                await self._handle_device_event('connected', device)
            
            logger.info(f"Initial scan complete. Found {len(self.discovered_devices)} devices")
            
        except Exception as e:
            logger.error(f"Error during initial device scan: {e}")
    
    def _scan_usb_devices(self) -> List[EnhancedDeviceInfo]:
        """Scan for USB devices"""
        devices = []
        
        if not HAS_PYUDEV:
            logger.warning("pyudev not available - USB scanning limited")
            return devices
        
        try:
            context = pyudev.Context()
            for device in context.list_devices(subsystem='usb'):
                if device.device_type == 'usb_device':
                    device_info = EnhancedDeviceInfo(
                        device_id=f"usb_{device.sys_name}",
                        device_type=DeviceType.USB,
                        name=device.get('ID_MODEL', 'Unknown USB Device'),
                        vendor=device.get('ID_VENDOR', 'Unknown'),
                        model=device.get('ID_MODEL', 'Unknown'),
                        capabilities=['usb_communication'],
                        ai_compatible=self._check_ai_compatibility(device),
                        connection_status="connected",
                        metadata={
                            'sys_path': device.sys_path,
                            'vendor_id': device.get('ID_VENDOR_ID'),
                            'product_id': device.get('ID_MODEL_ID')
                        }
                    )
                    devices.append(device_info)
        except Exception as e:
            logger.error(f"Error scanning USB devices: {e}")
        
        return devices
    
    def _scan_network_devices(self) -> List[EnhancedDeviceInfo]:
        """Scan for network devices"""
        devices = []
        
        if not HAS_PSUTIL:
            logger.warning("psutil not available - network scanning limited")
            return devices
        
        try:
            network_interfaces = psutil.net_if_addrs()
            for interface_name, addresses in network_interfaces.items():
                if interface_name != 'lo':  # Skip loopback
                    device_info = EnhancedDeviceInfo(
                        device_id=f"network_{interface_name}",
                        device_type=DeviceType.NETWORK,
                        name=f"Network Interface {interface_name}",
                        capabilities=['network_communication'],
                        ai_compatible=True,  # Network devices can be AI compatible
                        connection_status="connected" if psutil.net_if_stats()[interface_name].isup else "disconnected",
                        metadata={
                            'interface_name': interface_name,
                            'addresses': [addr.address for addr in addresses]
                        }
                    )
                    devices.append(device_info)
        except Exception as e:
            logger.error(f"Error scanning network devices: {e}")
        
        return devices
    
    def _scan_storage_devices(self) -> List[EnhancedDeviceInfo]:
        """Scan for storage devices"""
        devices = []
        
        if not HAS_PSUTIL:
            return devices
        
        try:
            disk_partitions = psutil.disk_partitions()
            for partition in disk_partitions:
                device_info = EnhancedDeviceInfo(
                    device_id=f"storage_{partition.device.replace('/', '_')}",
                    device_type=DeviceType.STORAGE,
                    name=f"Storage {partition.device}",
                    capabilities=['data_storage'],
                    ai_compatible=True,  # Storage can be used for AI data
                    connection_status="connected",
                    metadata={
                        'device_path': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype
                    }
                )
                devices.append(device_info)
        except Exception as e:
            logger.error(f"Error scanning storage devices: {e}")
        
        return devices
    
    def _check_ai_compatibility(self, device) -> bool:
        """Check if device is AI compatible"""
        # Simple heuristics for AI compatibility
        ai_keywords = ['gpu', 'tpu', 'neural', 'ai', 'accelerator', 'cuda', 'opencl']
        device_name = str(device.get('ID_MODEL', '')).lower()
        vendor_name = str(device.get('ID_VENDOR', '')).lower()
        
        return any(keyword in device_name or keyword in vendor_name for keyword in ai_keywords)
    
    async def _monitor_devices(self):
        """Monitor for device changes"""
        while self._monitoring:
            try:
                # Simple monitoring - in production would use udev monitoring
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                # Check for device changes
                current_devices = {}
                
                # Re-scan devices
                usb_devices = self._scan_usb_devices()
                network_devices = self._scan_network_devices()
                storage_devices = self._scan_storage_devices()
                
                all_current_devices = usb_devices + network_devices + storage_devices
                
                for device in all_current_devices:
                    current_devices[device.device_id] = device
                
                # Check for new devices
                for device_id, device_info in current_devices.items():
                    if device_id not in self.discovered_devices:
                        await self._handle_device_event('connected', device_info)
                
                # Check for removed devices
                for device_id in list(self.discovered_devices.keys()):
                    if device_id not in current_devices:
                        await self._handle_device_event('disconnected', self.discovered_devices[device_id])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in device monitoring: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_device_event(self, event_type: str, device_info: EnhancedDeviceInfo):
        """Handle device discovery events"""
        try:
            # Update registry
            if event_type == 'connected':
                self.discovered_devices[device_info.device_id] = device_info
                logger.info(f"Device connected: {device_info.name} ({device_info.device_type.value})")
            elif event_type == 'disconnected':
                if device_info.device_id in self.discovered_devices:
                    del self.discovered_devices[device_info.device_id]
                logger.info(f"Device disconnected: {device_info.name}")
            
            # Create event
            event = DeviceDiscoveryEvent(event_type, device_info)
            
            # Queue for AI analysis if enabled
            if self.ai_integration_enabled and device_info.ai_compatible:
                await self.ai_device_analysis_queue.put(event)
            
            # Notify event handlers
            for handler in self.event_handlers.get(event_type, []):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
            
            # Publish to message bus if available
            if self.message_bus:
                await self.message_bus.publish(f"device.{event_type}", event.to_dict())
            
        except Exception as e:
            logger.error(f"Error handling device event: {e}")
    
    async def _ai_device_analysis(self):
        """AI analysis of discovered devices"""
        if not self.ai_integration_enabled:
            return
        
        logger.info("Starting AI device analysis task...")
        
        while self._monitoring:
            try:
                # Wait for device events to analyze
                event = await asyncio.wait_for(self.ai_device_analysis_queue.get(), timeout=10.0)
                
                # Analyze device with AI orchestrator
                analysis_result = await self._analyze_device_with_ai(event)
                
                if analysis_result:
                    # Update device info with AI insights
                    device_info = event.device_info
                    device_info.metadata['ai_analysis'] = analysis_result
                    
                    # Mark as processed
                    event.processed_by_ai = True
                    
                    # Trigger AI ready event if device has interesting capabilities
                    if analysis_result.get('interesting_capabilities'):
                        await self._handle_device_event('ai_ready', device_info)
                
            except asyncio.TimeoutError:
                continue  # No events to process
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in AI device analysis: {e}")
    
    async def _analyze_device_with_ai(self, event: DeviceDiscoveryEvent) -> Dict[str, Any]:
        """Analyze device capabilities using AI orchestrator"""
        try:
            if not self.ai_orchestrator:
                return {}
            
            # Create analysis prompt for the AI system
            analysis_prompt = [
                f"Analyze this newly discovered device for AI/automation potential:",
                f"Device: {event.device_info.name}",
                f"Type: {event.device_info.device_type.value}",
                f"Vendor: {event.device_info.vendor}",
                f"Model: {event.device_info.model}",
                f"Current capabilities: {', '.join(event.device_info.capabilities)}",
                f"AI Compatible: {event.device_info.ai_compatible}",
                "What automation or AI integration opportunities does this device present?"
            ]
            
            # Process with AI orchestrator
            ai_result = await self.ai_orchestrator.process_conversation(
                analysis_prompt, 
                session_id=f"device_analysis_{event.device_info.device_id}"
            )
            
            if ai_result.get('status') == 'success':
                return {
                    'analysis_timestamp': time.time(),
                    'ai_recommendations': ai_result.get('response', {}).get('insights', []),
                    'automation_potential': ai_result.get('response', {}).get('summary', ''),
                    'interesting_capabilities': len(ai_result.get('response', {}).get('insights', [])) > 0,
                    'orchestration_plan': ai_result.get('orchestration_plan', {})
                }
            
        except Exception as e:
            logger.error(f"Error in AI device analysis: {e}")
        
        return {}
    
    def get_all_devices(self) -> Dict[str, EnhancedDeviceInfo]:
        """Get all discovered devices"""
        return self.discovered_devices.copy()
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[EnhancedDeviceInfo]:
        """Get devices by type"""
        return [device for device in self.discovered_devices.values() if device.device_type == device_type]
    
    def get_ai_compatible_devices(self) -> List[EnhancedDeviceInfo]:
        """Get AI compatible devices"""
        return [device for device in self.discovered_devices.values() if device.ai_compatible]
    
    def get_device_by_id(self, device_id: str) -> Optional[EnhancedDeviceInfo]:
        """Get device by ID"""
        return self.discovered_devices.get(device_id)

# Compatibility alias for existing code
DeviceDiscoveryService = EnhancedDeviceDiscoveryService
DeviceEvent = DeviceDiscoveryEvent