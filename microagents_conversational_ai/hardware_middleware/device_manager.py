"""
Device Manager - Central hardware device abstraction layer
Handles USB, serial, cameras, audio, network devices, and more.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Any
from threading import Thread, Lock
import json

# Optional imports with graceful fallback
try:
    import pyudev
    HAS_PYUDEV = True
except ImportError:
    HAS_PYUDEV = False
    
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

logger = logging.getLogger(__name__)

class Device:
    """Represents a hardware device with unified interface."""
    
    def __init__(self, device_id: str, device_type: str, metadata: Dict[str, Any]):
        self.device_id = device_id
        self.device_type = device_type
        self.metadata = metadata
        self.status = "detected"
        self.last_seen = time.time()
        self._connection = None
    
    def connect(self) -> bool:
        """Connect to the device."""
        try:
            if self.device_type == "serial" and HAS_SERIAL:
                self._connection = serial.Serial(
                    port=self.metadata.get('port'),
                    baudrate=self.metadata.get('baudrate', 9600),
                    timeout=1
                )
            elif self.device_type == "camera" and HAS_OPENCV:
                self._connection = cv2.VideoCapture(self.metadata.get('index', 0))
            
            self.status = "connected" if self._connection else "failed"
            return self.status == "connected"
        except Exception as e:
            logger.error(f"Failed to connect to device {self.device_id}: {e}")
            self.status = "failed"
            return False
    
    def disconnect(self):
        """Disconnect from the device."""
        if self._connection:
            if hasattr(self._connection, 'close'):
                self._connection.close()
            elif hasattr(self._connection, 'release'):
                self._connection.release()
            self._connection = None
        self.status = "disconnected"
    
    def send_command(self, command: str) -> Optional[str]:
        """Send command to device and return response."""
        if not self._connection:
            return None
            
        try:
            if self.device_type == "serial" and hasattr(self._connection, 'write'):
                self._connection.write(command.encode())
                response = self._connection.readline().decode().strip()
                return response
        except Exception as e:
            logger.error(f"Command failed on device {self.device_id}: {e}")
        return None
    
    def read_data(self) -> Optional[Any]:
        """Read data from device."""
        if not self._connection:
            return None
            
        try:
            if self.device_type == "camera" and hasattr(self._connection, 'read'):
                ret, frame = self._connection.read()
                return frame if ret else None
            elif self.device_type == "serial" and hasattr(self._connection, 'readline'):
                return self._connection.readline().decode().strip()
        except Exception as e:
            logger.error(f"Read failed on device {self.device_id}: {e}")
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert device to dictionary representation."""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'status': self.status,
            'metadata': self.metadata,
            'last_seen': self.last_seen
        }

class DeviceManager:
    """Central device manager for all hardware abstraction."""
    
    def __init__(self):
        self.devices: Dict[str, Device] = {}
        self._lock = Lock()
        self._monitoring = False
        self._monitor_thread = None
        
        # Initialize device discovery
        self._discover_initial_devices()
        
    def _discover_initial_devices(self):
        """Discover all initially available devices."""
        logger.info("Starting initial device discovery...")
        
        # Discover USB/Serial devices
        if HAS_SERIAL:
            self._discover_serial_devices()
        
        # Discover cameras
        if HAS_OPENCV:
            self._discover_camera_devices()
            
        # Discover audio devices
        if HAS_AUDIO:
            self._discover_audio_devices()
            
        # Discover network interfaces
        self._discover_network_devices()
        
        logger.info(f"Found {len(self.devices)} devices")
    
    def _discover_serial_devices(self):
        """Discover serial/USB devices."""
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                device_id = f"serial_{port.device}"
                metadata = {
                    'port': port.device,
                    'description': port.description,
                    'hwid': port.hwid,
                    'vid': port.vid,
                    'pid': port.pid
                }
                device = Device(device_id, "serial", metadata)
                with self._lock:
                    self.devices[device_id] = device
                logger.info(f"Found serial device: {device_id}")
        except Exception as e:
            logger.error(f"Serial device discovery failed: {e}")
    
    def _discover_camera_devices(self):
        """Discover camera devices."""
        for i in range(5):  # Check first 5 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    device_id = f"camera_{i}"
                    metadata = {
                        'index': i,
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': cap.get(cv2.CAP_PROP_FPS)
                    }
                    device = Device(device_id, "camera", metadata)
                    with self._lock:
                        self.devices[device_id] = device
                    logger.info(f"Found camera device: {device_id}")
                    cap.release()
            except Exception as e:
                logger.debug(f"Camera {i} check failed: {e}")
    
    def _discover_audio_devices(self):
        """Discover audio devices."""
        try:
            devices = sd.query_devices()
            for i, device_info in enumerate(devices):
                device_id = f"audio_{i}"
                metadata = {
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['max_input_channels'],
                    'sample_rate': device_info['default_samplerate']
                }
                device = Device(device_id, "audio", metadata)
                with self._lock:
                    self.devices[device_id] = device
            logger.info(f"Found {len(devices)} audio devices")
        except Exception as e:
            logger.error(f"Audio device discovery failed: {e}")
    
    def _discover_network_devices(self):
        """Discover network interfaces."""
        try:
            import psutil
            interfaces = psutil.net_if_addrs()
            for name, addrs in interfaces.items():
                device_id = f"network_{name}"
                metadata = {
                    'interface': name,
                    'addresses': [addr.address for addr in addrs]
                }
                device = Device(device_id, "network", metadata)
                with self._lock:
                    self.devices[device_id] = device
            logger.info(f"Found {len(interfaces)} network interfaces")
        except ImportError:
            logger.warning("psutil not available, skipping network discovery")
        except Exception as e:
            logger.error(f"Network device discovery failed: {e}")
    
    def start_monitoring(self):
        """Start continuous device monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = Thread(target=self._monitor_devices, daemon=True)
        self._monitor_thread.start()
        logger.info("Started device monitoring")
    
    def stop_monitoring(self):
        """Stop device monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped device monitoring")
    
    def _monitor_devices(self):
        """Monitor for device changes (hotplug/removal)."""
        if not HAS_PYUDEV:
            logger.warning("pyudev not available, device hotplug monitoring disabled")
            return
            
        try:
            context = pyudev.Context()
            monitor = pyudev.Monitor.from_netlink(context)
            monitor.filter_by('usb')
            
            for device in iter(monitor.poll, None):
                if not self._monitoring:
                    break
                    
                if device.action == 'add':
                    self._handle_device_added(device)
                elif device.action == 'remove':
                    self._handle_device_removed(device)
        except Exception as e:
            logger.error(f"Device monitoring failed: {e}")
    
    def _handle_device_added(self, device):
        """Handle device addition event."""
        device_id = f"usb_{device.device_node}" if device.device_node else f"usb_{device.sys_name}"
        metadata = {
            'sys_name': device.sys_name,
            'device_node': device.device_node,
            'subsystem': device.subsystem,
            'device_type': device.device_type
        }
        new_device = Device(device_id, "usb", metadata)
        
        with self._lock:
            self.devices[device_id] = new_device
        
        logger.info(f"Device added: {device_id}")
    
    def _handle_device_removed(self, device):
        """Handle device removal event."""
        device_id = f"usb_{device.device_node}" if device.device_node else f"usb_{device.sys_name}"
        
        with self._lock:
            if device_id in self.devices:
                removed_device = self.devices.pop(device_id)
                removed_device.disconnect()
        
        logger.info(f"Device removed: {device_id}")
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by ID."""
        with self._lock:
            return self.devices.get(device_id)
    
    def get_devices_by_type(self, device_type: str) -> List[Device]:
        """Get all devices of a specific type."""
        with self._lock:
            return [device for device in self.devices.values() 
                   if device.device_type == device_type]
    
    def list_devices(self) -> Dict[str, Device]:
        """Get all devices."""
        with self._lock:
            return self.devices.copy()
    
    def connect_device(self, device_id: str) -> bool:
        """Connect to a specific device."""
        device = self.get_device(device_id)
        if device:
            return device.connect()
        return False
    
    def disconnect_device(self, device_id: str) -> bool:
        """Disconnect from a specific device."""
        device = self.get_device(device_id)
        if device:
            device.disconnect()
            return True
        return False
    
    def send_command(self, device_id: str, command: str) -> Optional[str]:
        """Send command to device."""
        device = self.get_device(device_id)
        if device:
            return device.send_command(command)
        return None
    
    def read_data(self, device_id: str) -> Optional[Any]:
        """Read data from device."""
        device = self.get_device(device_id)
        if device:
            return device.read_data()
        return None
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get status summary of all devices."""
        with self._lock:
            status = {
                'total_devices': len(self.devices),
                'by_type': {},
                'by_status': {},
                'devices': {}
            }
            
            for device in self.devices.values():
                # Count by type
                if device.device_type not in status['by_type']:
                    status['by_type'][device.device_type] = 0
                status['by_type'][device.device_type] += 1
                
                # Count by status
                if device.status not in status['by_status']:
                    status['by_status'][device.status] = 0
                status['by_status'][device.status] += 1
                
                # Add device details
                status['devices'][device.device_id] = device.to_dict()
            
            return status
    
    def export_config(self, filename: str):
        """Export device configuration to file."""
        config = {
            'devices': {dev_id: device.to_dict() 
                       for dev_id, device in self.devices.items()},
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Device configuration exported to {filename}")
    
    def cleanup(self):
        """Cleanup all resources."""
        self.stop_monitoring()
        
        with self._lock:
            for device in self.devices.values():
                device.disconnect()
            self.devices.clear()
        
        logger.info("DeviceManager cleanup completed")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize device manager
    dm = DeviceManager()
    
    # Start monitoring
    dm.start_monitoring()
    
    try:
        # Print device status
        status = dm.get_device_status()
        print("Device Status:")
        print(json.dumps(status, indent=2))
        
        # Test device operations
        devices = dm.list_devices()
        for device_id, device in devices.items():
            print(f"\nTesting device: {device_id}")
            
            # Try to connect
            if dm.connect_device(device_id):
                print(f"  ✓ Connected to {device_id}")
                
                # Try to read data (non-blocking)
                try:
                    data = dm.read_data(device_id)
                    if data:
                        print(f"  ✓ Read data: {str(data)[:100]}...")
                except Exception as e:
                    print(f"  ⚠ Read failed: {e}")
                
                # Disconnect
                dm.disconnect_device(device_id)
                print(f"  ✓ Disconnected from {device_id}")
            else:
                print(f"  ✗ Failed to connect to {device_id}")
        
        # Export configuration
        dm.export_config("device_config.json")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        dm.cleanup()
