"""
Hardware Middleware - Main Integration Module
Provides unified interface for all hardware middleware components.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .device_abstraction_layer import (
    DeviceAbstractionLayer, DeviceMetadata, DeviceType, DeviceStatus,
    create_camera_device, create_audio_device, create_usb_device
)
from .communication_bus import (
    CommunicationBus, Message, MessageType, MessagePriority,
    create_command_message, create_response_message, create_event_message
)
from .device_discovery_service import (
    DeviceDiscoveryService, DiscoveredDevice, DeviceEvent
)

logger = logging.getLogger(__name__)

class HardwareMiddleware:
    """
    Main hardware middleware interface that integrates all components.
    Provides a unified API for device management and communication.
    """
    
    def __init__(self, bus_id: str = None):
        # Initialize core components
        self.communication_bus = CommunicationBus(bus_id)
        self.device_abstraction_layer = DeviceAbstractionLayer()
        self.device_discovery_service = DeviceDiscoveryService(self.communication_bus)
        
        # State tracking
        self.is_running = False
        self.auto_register_devices = True
        self.device_callbacks: Dict[str, List[Callable]] = {
            'device_discovered': [],
            'device_connected': [],
            'device_disconnected': [],
            'device_error': []
        }
        
        # Setup device discovery callbacks
        self.device_discovery_service.add_device_callback(self._handle_device_event)
        
        logger.info("Hardware Middleware initialized")
    
    async def start(self):
        """Start all middleware components."""
        if self.is_running:
            return
        
        try:
            # Start communication bus
            await self.communication_bus.start()
            
            # Start device discovery
            await self.device_discovery_service.start()
            
            self.is_running = True
            logger.info("Hardware Middleware started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Hardware Middleware: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop all middleware components."""
        if not self.is_running:
            return
        
        try:
            # Stop device discovery
            await self.device_discovery_service.stop()
            
            # Disconnect all devices
            await self._disconnect_all_devices()
            
            # Stop communication bus
            await self.communication_bus.stop()
            
            self.is_running = False
            logger.info("Hardware Middleware stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Hardware Middleware: {e}")
    
    # Device Management API
    
    async def get_all_devices(self) -> Dict[str, DeviceMetadata]:
        """Get all registered devices."""
        return await self.device_abstraction_layer.get_all_devices()
    
    async def get_devices_by_type(self, device_type: DeviceType) -> Dict[str, DeviceMetadata]:
        """Get devices of a specific type."""
        return await self.device_abstraction_layer.get_devices_by_type(device_type)
    
    async def get_device_status(self, device_id: str) -> Optional[DeviceStatus]:
        """Get the status of a specific device."""
        return await self.device_abstraction_layer.get_device_status(device_id)
    
    async def connect_device(self, device_id: str) -> bool:
        """Connect to a specific device."""
        success = await self.device_abstraction_layer.connect_device(device_id)
        
        if success:
            await self._notify_device_event('device_connected', device_id)
            
            # Send connection event to bus
            event_msg = create_event_message(
                source="hardware_middleware",
                event_type="device_connected",
                data={'device_id': device_id}
            )
            await self.communication_bus.publish(event_msg)
        
        return success
    
    async def disconnect_device(self, device_id: str) -> bool:
        """Disconnect from a specific device."""
        success = await self.device_abstraction_layer.disconnect_device(device_id)
        
        if success:
            await self._notify_device_event('device_disconnected', device_id)
            
            # Send disconnection event to bus
            event_msg = create_event_message(
                source="hardware_middleware",
                event_type="device_disconnected",
                data={'device_id': device_id}
            )
            await self.communication_bus.publish(event_msg)
        
        return success
    
    async def execute_device_command(self, device_id: str, action: str, 
                                   parameters: Dict[str, Any] = None) -> Any:
        """Execute a command on a device and return the response."""
        if parameters is None:
            parameters = {}
        
        try:
            response = await self.device_abstraction_layer.execute_command(
                device_id, action, parameters
            )
            
            # Send command execution event
            event_msg = create_event_message(
                source="hardware_middleware",
                event_type="command_executed",
                data={
                    'device_id': device_id,
                    'action': action,
                    'success': response.success,
                    'execution_time': response.execution_time
                }
            )
            await self.communication_bus.publish(event_msg)
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing command on {device_id}: {e}")
            
            # Send error event
            event_msg = create_event_message(
                source="hardware_middleware",
                event_type="command_error",
                data={
                    'device_id': device_id,
                    'action': action,
                    'error': str(e)
                }
            )
            await self.communication_bus.publish(event_msg)
            
            raise
    
    # High-level convenience methods
    
    async def capture_image(self, camera_id: str = None, resolution: str = "1920x1080", 
                          format_type: str = "jpg") -> Any:
        """Capture an image from a camera."""
        if not camera_id:
            # Find first available camera
            cameras = await self.get_devices_by_type(DeviceType.CAMERA)
            if not cameras:
                raise ValueError("No cameras available")
            camera_id = list(cameras.keys())[0]
        
        return await self.execute_device_command(camera_id, "capture_image", {
            'resolution': resolution,
            'format': format_type
        })
    
    async def record_video(self, camera_id: str = None, duration: int = 10,
                         resolution: str = "1920x1080", fps: int = 30) -> Any:
        """Record video from a camera."""
        if not camera_id:
            cameras = await self.get_devices_by_type(DeviceType.CAMERA)
            if not cameras:
                raise ValueError("No cameras available")
            camera_id = list(cameras.keys())[0]
        
        return await self.execute_device_command(camera_id, "record_video", {
            'duration': duration,
            'resolution': resolution,
            'fps': fps
        })
    
    async def record_audio(self, audio_device_id: str = None, duration: int = 10,
                         sample_rate: int = 44100, format_type: str = "wav") -> Any:
        """Record audio from a microphone."""
        if not audio_device_id:
            audio_devices = await self.get_devices_by_type(DeviceType.AUDIO)
            if not audio_devices:
                raise ValueError("No audio devices available")
            audio_device_id = list(audio_devices.keys())[0]
        
        return await self.execute_device_command(audio_device_id, "record_audio", {
            'duration': duration,
            'sample_rate': sample_rate,
            'format': format_type
        })
    
    async def play_audio(self, audio_device_id: str = None, filename: str = None) -> Any:
        """Play audio through a speaker."""
        if not audio_device_id:
            audio_devices = await self.get_devices_by_type(DeviceType.AUDIO)
            if not audio_devices:
                raise ValueError("No audio devices available")
            audio_device_id = list(audio_devices.keys())[0]
        
        if not filename:
            raise ValueError("Filename required for audio playback")
        
        return await self.execute_device_command(audio_device_id, "play_audio", {
            'filename': filename
        })
    
    # Communication Bus API
    
    async def subscribe_to_events(self, topic_pattern: str, 
                                callback: Callable[[Message], None]) -> bool:
        """Subscribe to events on the communication bus."""
        return await self.communication_bus.subscribe(
            subscriber_id="hardware_middleware_client",
            topic_pattern=topic_pattern,
            callback=callback,
            message_types={MessageType.EVENT}
        )
    
    async def publish_event(self, event_type: str, data: Any = None) -> bool:
        """Publish an event to the communication bus."""
        event_msg = create_event_message(
            source="hardware_middleware",
            event_type=event_type,
            data=data
        )
        return await self.communication_bus.publish(event_msg)
    
    # Device Discovery API
    
    async def scan_for_devices(self) -> Dict[str, DiscoveredDevice]:
        """Manually trigger device discovery scan."""
        return await self.device_discovery_service.scan_all_devices()
    
    async def get_discovered_devices(self) -> Dict[str, DiscoveredDevice]:
        """Get all discovered devices."""
        return await self.device_discovery_service.get_all_devices()
    
    async def get_discovered_devices_by_type(self, device_type: DeviceType) -> List[DiscoveredDevice]:
        """Get discovered devices of a specific type."""
        return await self.device_discovery_service.get_devices_by_type(device_type)
    
    # Configuration and Callbacks
    
    def add_device_callback(self, event_type: str, callback: Callable[[str], None]):
        """Add callback for device events."""
        if event_type in self.device_callbacks:
            self.device_callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown device event type: {event_type}")
    
    def remove_device_callback(self, event_type: str, callback: Callable[[str], None]):
        """Remove device event callback."""
        if event_type in self.device_callbacks and callback in self.device_callbacks[event_type]:
            self.device_callbacks[event_type].remove(callback)
    
    def set_auto_register_devices(self, enabled: bool):
        """Enable/disable automatic device registration."""
        self.auto_register_devices = enabled
        logger.info(f"Auto device registration: {'enabled' if enabled else 'disabled'}")
    
    # Status and Health Monitoring
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        dal_health = await self.device_abstraction_layer.health_check()
        bus_stats = await self.communication_bus.get_stats()
        discovery_stats = await self.device_discovery_service.get_discovery_stats()
        
        return {
            'middleware_running': self.is_running,
            'timestamp': datetime.now().isoformat(),
            'device_abstraction_layer': dal_health,
            'communication_bus': bus_stats,
            'device_discovery': discovery_stats
        }
    
    async def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data from all devices."""
        telemetry = {}
        
        devices = await self.get_all_devices()
        for device_id in devices.keys():
            try:
                device_telemetry = await self.device_abstraction_layer.get_device_telemetry(device_id)
                if device_telemetry:
                    telemetry[device_id] = device_telemetry
            except Exception as e:
                logger.warning(f"Failed to get telemetry for {device_id}: {e}")
        
        return telemetry
    
    # Private methods
    
    async def _handle_device_event(self, device: DiscoveredDevice, event: DeviceEvent):
        """Handle device discovery events."""
        if event == DeviceEvent.DISCOVERED and self.auto_register_devices:
            # Automatically register newly discovered devices
            success = self.device_abstraction_layer.register_device(device.device_metadata)
            if success:
                logger.info(f"Auto-registered device: {device.device_id}")
                await self._notify_device_event('device_discovered', device.device_id)
        
        elif event == DeviceEvent.LOST:
            # Handle lost devices
            await self.disconnect_device(device.device_id)
            logger.info(f"Device lost: {device.device_id}")
    
    async def _notify_device_event(self, event_type: str, device_id: str):
        """Notify registered callbacks about device events."""
        if event_type in self.device_callbacks:
            for callback in self.device_callbacks[event_type]:
                try:
                    callback(device_id)
                except Exception as e:
                    logger.error(f"Error in device callback: {e}")
    
    async def _disconnect_all_devices(self):
        """Disconnect from all devices."""
        devices = await self.get_all_devices()
        for device_id in devices.keys():
            try:
                await self.disconnect_device(device_id)
            except Exception as e:
                logger.warning(f"Error disconnecting {device_id}: {e}")

# Factory function for easy initialization
def create_hardware_middleware(bus_id: str = None, auto_start: bool = True) -> HardwareMiddleware:
    """Create and optionally start hardware middleware."""
    middleware = HardwareMiddleware(bus_id)
    
    if auto_start:
        # Note: In real usage, you'd need to await middleware.start()
        # This is just for convenience in sync contexts
        pass
    
    return middleware

# Export the main classes
__all__ = [
    'HardwareMiddleware',
    'DeviceAbstractionLayer',
    'CommunicationBus',
    'DeviceDiscoveryService',
    'DeviceMetadata',
    'DeviceType',
    'DeviceStatus',
    'Message',
    'MessageType',
    'DiscoveredDevice',
    'DeviceEvent',
    'create_hardware_middleware',
    'create_camera_device',
    'create_audio_device',
    'create_usb_device'
]

# Example usage and testing
if __name__ == "__main__":
    async def demo():
        print("Hardware Middleware - Complete Integration Demo")
        print("=" * 60)
        
        # Create and start middleware
        middleware = HardwareMiddleware("demo_middleware")
        
        # Add device event callbacks
        def on_device_discovered(device_id: str):
            print(f"🔍 Device discovered: {device_id}")
        
        def on_device_connected(device_id: str):
            print(f"🔗 Device connected: {device_id}")
        
        middleware.add_device_callback('device_discovered', on_device_discovered)
        middleware.add_device_callback('device_connected', on_device_connected)
        
        # Start middleware
        await middleware.start()
        print("✅ Hardware Middleware started")
        
        # Wait for device discovery
        print("\n🔍 Discovering devices...")
        await asyncio.sleep(3)
        
        # Show discovered devices
        discovered = await middleware.get_discovered_devices()
        print(f"\n📱 Found {len(discovered)} devices:")
        for device_id, device in discovered.items():
            if device.is_available:
                print(f"   • {device.device_metadata.name} ({device.device_metadata.device_type.value})")
        
        # Show registered devices
        registered = await middleware.get_all_devices()
        print(f"\n📋 Registered {len(registered)} devices:")
        for device_id, metadata in registered.items():
            print(f"   • {metadata.name} ({metadata.device_type.value})")
        
        # Try to use devices if available
        try:
            # Try to capture image if camera available
            cameras = await middleware.get_devices_by_type(DeviceType.CAMERA)
            if cameras:
                print(f"\n📸 Testing camera capture...")
                camera_id = list(cameras.keys())[0]
                await middleware.connect_device(camera_id)
                
                result = await middleware.capture_image(camera_id)
                if result.success:
                    print(f"   ✅ Image captured: {result.data['filename']}")
                else:
                    print(f"   ❌ Capture failed: {result.error_message}")
            
            # Try to record audio if available
            audio_devices = await middleware.get_devices_by_type(DeviceType.AUDIO)
            if audio_devices:
                print(f"\n🎤 Testing audio recording...")
                audio_id = list(audio_devices.keys())[0]
                await middleware.connect_device(audio_id)
                
                result = await middleware.record_audio(audio_id, duration=2)
                if result.success:
                    print(f"   ✅ Audio recorded: {result.data['filename']}")
                else:
                    print(f"   ❌ Recording failed: {result.error_message}")
        
        except Exception as e:
            print(f"   ⚠️  Device operation error: {e}")
        
        # Show system health
        print(f"\n🏥 System Health Check:")
        health = await middleware.get_system_health()
        print(f"   Middleware running: {health['middleware_running']}")
        print(f"   Connected devices: {health['device_abstraction_layer']['connected_devices']}")
        print(f"   Total devices: {health['device_abstraction_layer']['total_devices']}")
        print(f"   Bus messages sent: {health['communication_bus']['messages_sent']}")
        print(f"   Discovery running: {health['device_discovery']['running']}")
        
        # Stop middleware
        await middleware.stop()
        print(f"\n✅ Hardware Middleware Demo completed!")
    
    asyncio.run(demo())
