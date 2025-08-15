# Hardware Middleware System

Complete hardware abstraction and device management system for the microagents conversational AI platform.

## Overview

The Hardware Middleware provides a unified interface for managing hardware devices across platforms, enabling seamless integration of cameras, audio devices, USB peripherals, and more into the AI orchestration system.

## Architecture

### Core Components

#### 1. Device Abstraction Layer (DAL)
- **File**: `device_abstraction_layer.py`
- **Purpose**: Normalizes communication with heterogeneous devices
- **Features**:
  - Unified device interfaces for all hardware types
  - Device-specific adapters (Camera, Audio, USB, etc.)
  - Command/response pattern with async execution
  - Device health monitoring and telemetry
  - Hot-plug detection and management

#### 2. Communication Bus
- **File**: `communication_bus.py`
- **Purpose**: Unified messaging infrastructure for all components
- **Features**:
  - Priority-based message queuing
  - Quality of Service (QoS) levels
  - Topic-based pub/sub messaging
  - Multiple transport support (ZeroMQ, MQTT, gRPC)
  - Message persistence and delivery guarantees

#### 3. Device Discovery Service
- **File**: `device_discovery_service.py`
- **Purpose**: Automatic hardware detection and lifecycle management
- **Features**:
  - Multi-platform device discovery (Linux, macOS, Windows)
  - Hot-plug detection and notifications
  - Device classification and metadata extraction
  - Continuous monitoring and health checks

### Supported Device Types

| Device Type | Connection Types | Capabilities |
|-------------|------------------|--------------|
| **Camera** | USB, Network | Image capture, video recording, streaming |
| **Audio** | USB, Built-in | Recording, playback, monitoring |
| **USB** | USB 2.0/3.0 | Data transfer, status queries |
| **Serial** | RS232, RS485 | Serial communication |
| **Network** | Ethernet, WiFi | Network-attached devices |
| **Storage** | USB, SATA | File operations, backup |
| **Sensor** | I2C, SPI, GPIO | Environmental monitoring |
| **Actuator** | GPIO, PWM | Control and automation |

## Quick Start

### Basic Usage

```python
import asyncio
from hardware_middleware import HardwareMiddleware

async def main():
    # Create and start middleware
    middleware = HardwareMiddleware("my_system")
    await middleware.start()
    
    # Capture an image
    result = await middleware.capture_image(resolution="1920x1080")
    if result.success:
        print(f"Image saved: {result.data['filename']}")
    
    # Record audio
    result = await middleware.record_audio(duration=5, format_type="wav")
    if result.success:
        print(f"Audio saved: {result.data['filename']}")
    
    # Stop middleware
    await middleware.stop()

asyncio.run(main())
```

### Advanced Device Management

```python
from hardware_middleware import (
    HardwareMiddleware, DeviceType, 
    create_camera_device, create_audio_device
)

async def advanced_example():
    middleware = HardwareMiddleware()
    await middleware.start()
    
    # Get devices by type
    cameras = await middleware.get_devices_by_type(DeviceType.CAMERA)
    audio_devices = await middleware.get_devices_by_type(DeviceType.AUDIO)
    
    # Connect to specific device
    if cameras:
        camera_id = list(cameras.keys())[0]
        await middleware.connect_device(camera_id)
        
        # Execute custom command
        result = await middleware.execute_device_command(
            camera_id, 
            "capture_image", 
            {"resolution": "4K", "format": "raw"}
        )
    
    # Subscribe to device events
    def on_device_event(message):
        print(f"Device event: {message.payload}")
    
    await middleware.subscribe_to_events("event/device_*", on_device_event)
    
    await middleware.stop()
```

## Device Discovery

The system automatically discovers devices using platform-specific methods:

### Linux Discovery
- **USB**: `lsusb` command and sysfs
- **Video**: Video4Linux (`/dev/video*`)
- **Audio**: ALSA (`arecord`/`aplay`)
- **Serial**: `/dev/tty*` enumeration

### macOS Discovery
- **USB**: `system_profiler SPUSBDataType`
- **Video**: AVFoundation framework
- **Audio**: Core Audio APIs

### Windows Discovery
- **USB**: WMI queries
- **Video**: DirectShow enumeration
- **Audio**: Windows Audio APIs

## Communication Patterns

### Message Types
- **COMMAND**: Device operation requests
- **RESPONSE**: Command execution results  
- **EVENT**: Device lifecycle and status changes
- **TELEMETRY**: Real-time device metrics
- **HEARTBEAT**: Device health monitoring

### Quality of Service
- **AT_MOST_ONCE**: Fire-and-forget delivery
- **AT_LEAST_ONCE**: Guaranteed delivery
- **EXACTLY_ONCE**: Guaranteed unique delivery

### Topic Patterns
```
command/{device_id}     # Device commands
response/{device_id}    # Command responses
event/{event_type}      # System events
telemetry/{device_id}   # Device metrics
heartbeat/{device_id}   # Health monitoring
```

## Configuration

### Environment Variables
```bash
# Device discovery settings
HARDWARE_DISCOVERY_INTERVAL=10        # Discovery scan interval (seconds)
HARDWARE_DEVICE_TIMEOUT=30           # Device timeout (seconds)
HARDWARE_AUTO_REGISTER=true          # Auto-register discovered devices

# Communication bus settings
HARDWARE_BUS_MAX_QUEUE_SIZE=10000    # Message queue size
HARDWARE_BUS_PORT=5555               # ZeroMQ port
HARDWARE_MQTT_BROKER=localhost       # MQTT broker address
HARDWARE_MQTT_PORT=1883              # MQTT broker port

# Logging settings
HARDWARE_LOG_LEVEL=INFO              # Logging level
```

### Device Configuration
```python
# Custom device registration
from hardware_middleware import create_camera_device

custom_camera = create_camera_device(
    device_id="custom_cam_001",
    name="High-End Camera",
    vendor="CameraCorp",
    model="ProCam 4K",
    max_resolution="3840x2160",
    capabilities=["4k_recording", "raw_capture", "night_vision"]
)

middleware.device_abstraction_layer.register_device(custom_camera)
```

## Integration with Tokenizer System

The hardware middleware integrates seamlessly with the action tokenizer:

```python
from tokenizer.action_tokenizer import ActionTokenizer
from hardware_middleware import HardwareMiddleware

async def integrated_example():
    # Initialize both systems
    middleware = HardwareMiddleware()
    tokenizer = ActionTokenizer()
    
    await middleware.start()
    
    # Tokenize hardware-related intent
    intent = "take a photo with the front camera in high resolution"
    tokens = await tokenizer.tokenize(intent, mode=TokenMode.LLM_GUIDED)
    
    # Extract hardware commands from tokens
    for token in tokens:
        if token.token_type == "hardware_command":
            device_type = token.parameters.get("device_type")
            action = token.parameters.get("action")
            
            if device_type == "camera" and action == "capture":
                # Execute on hardware
                result = await middleware.capture_image(
                    resolution=token.parameters.get("resolution", "1920x1080")
                )
                print(f"Captured: {result.data['filename']}")
```

## Testing

Run the comprehensive test suite:

```bash
python test_hardware_middleware.py
```

This tests:
- ✅ Device Abstraction Layer functionality
- ✅ Communication Bus messaging
- ✅ Device Discovery Service
- ✅ Event system integration
- ✅ Telemetry collection
- ✅ Error handling and recovery

## Platform Support

| Platform | USB | Video | Audio | Serial | Network |
|----------|-----|-------|-------|---------|---------|
| **Linux** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **macOS** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Windows** | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ |

⚠️ = Partial implementation (requires additional libraries)

## Dependencies

### Core Dependencies
```bash
# Already included in Python standard library
asyncio          # Async programming
threading        # Thread management
subprocess       # System command execution
json            # Data serialization
logging         # Logging system
```

### Optional Dependencies
```bash
# For enhanced functionality (install as needed)
pyusb           # USB device access
pyserial        # Serial communication
opencv-python   # Camera operations
pyaudio         # Audio operations
paho-mqtt       # MQTT transport
pyzmq           # ZeroMQ transport
```

## API Reference

### HardwareMiddleware Class

#### Core Methods
- `start()` - Start all middleware components
- `stop()` - Stop all middleware components
- `get_all_devices()` - Get all registered devices
- `get_devices_by_type(device_type)` - Get devices by type
- `connect_device(device_id)` - Connect to specific device
- `disconnect_device(device_id)` - Disconnect from device
- `execute_device_command(device_id, action, parameters)` - Execute device command

#### Convenience Methods
- `capture_image(camera_id, resolution, format_type)` - Capture image
- `record_video(camera_id, duration, resolution, fps)` - Record video
- `record_audio(audio_device_id, duration, sample_rate)` - Record audio
- `play_audio(audio_device_id, filename)` - Play audio

#### Event Management
- `subscribe_to_events(topic_pattern, callback)` - Subscribe to events
- `publish_event(event_type, data)` - Publish event
- `add_device_callback(event_type, callback)` - Add device callback

#### Monitoring
- `get_system_health()` - Get system health status
- `get_telemetry()` - Get device telemetry data
- `scan_for_devices()` - Manual device discovery scan

## Error Handling

The system provides comprehensive error handling:

```python
from hardware_middleware import HardwareMiddleware

async def error_handling_example():
    middleware = HardwareMiddleware()
    
    try:
        await middleware.start()
        
        # This will raise an exception if no cameras are available
        result = await middleware.capture_image()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await middleware.stop()
```

## Performance Considerations

### Memory Usage
- Device adapters use minimal memory footprint
- Message queue has configurable size limits
- Automatic cleanup of expired messages and telemetry

### CPU Usage
- Async/await pattern for non-blocking operations
- Background workers for continuous monitoring
- Efficient event-driven architecture

### Network Usage
- Configurable discovery intervals
- Efficient binary protocols for device communication
- Optional compression for large data transfers

## Security Considerations

### Device Access Control
- Device permissions based on user context
- Secure credential management for network devices
- Audit logging for all device operations

### Communication Security
- TLS encryption for network transports
- Message authentication and integrity
- Secure key exchange for device pairing

## Contributing

When adding new device types or adapters:

1. Extend the `DeviceAdapter` base class
2. Add appropriate `DeviceType` enum value  
3. Implement platform-specific discovery methods
4. Add comprehensive test coverage
5. Update this documentation

## License

This hardware middleware system is part of the microagents conversational AI platform.

---

**Status**: ✅ Complete and tested
**Version**: 1.0.0  
**Last Updated**: December 2024
