"""
Device Discovery Service - Auto-detection and hotplug monitoring
Continuously monitors for new devices and updates the device registry.
"""

import time
import logging
import threading
from typing import Dict, List, Callable, Optional, Set
import json

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

class DeviceEvent:
    """Represents a device discovery event."""
    
    def __init__(self, event_type: str, device_info: Dict, timestamp: float = None):
        self.event_type = event_type  # 'added', 'removed', 'changed'
        self.device_info = device_info
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type,
            'device_info': self.device_info,
            'timestamp': self.timestamp
        }

class DeviceDiscoveryService:
    """Service for discovering and monitoring hardware devices."""
    
    def __init__(self, device_manager=None, message_bus=None):
        self.device_manager = device_manager
        self.message_bus = message_bus
        self.callbacks: List[Callable[[DeviceEvent], None]] = []
        
        # Monitoring state
        self._monitoring = False
        self._monitor_threads: List[threading.Thread] = []
        self._known_devices: Set[str] = set()
        
        # Discovery modules
        self.discovery_modules = {
            'usb': self._discover_usb_devices,
            'network': self._discover_network_devices,
            'storage': self._discover_storage_devices,
            'audio': self._discover_audio_devices,
            'video': self._discover_video_devices
        }
        
        # Monitoring modules  
        self.monitor_modules = {
            'udev': self._monitor_udev,
            'network': self._monitor_network,
            'storage': self._monitor_storage
        }
    
    def start_discovery(self):
        """Start the device discovery service."""
        logger.info("Starting device discovery service")
        
        # Initial discovery sweep
        self._perform_initial_discovery()
        
        # Start monitoring threads
        self._monitoring = True
        for name, monitor_func in self.monitor_modules.items():
            thread = threading.Thread(target=monitor_func, name=f"monitor_{name}", daemon=True)
            thread.start()
            self._monitor_threads.append(thread)
            logger.info(f"Started {name} monitoring thread")
    
    def stop_discovery(self):
        """Stop the device discovery service."""
        logger.info("Stopping device discovery service")
        
        self._monitoring = False
        
        # Wait for monitor threads to finish
        for thread in self._monitor_threads:
            thread.join(timeout=5)
        
        self._monitor_threads.clear()
        logger.info("Device discovery service stopped")
    
    def _perform_initial_discovery(self):
        """Perform initial device discovery sweep."""
        logger.info("Performing initial device discovery sweep")
        
        discovered_count = 0
        for name, discover_func in self.discovery_modules.items():
            try:
                devices = discover_func()
                for device_info in devices:
                    device_id = device_info.get('device_id')
                    if device_id and device_id not in self._known_devices:
                        self._known_devices.add(device_id)
                        event = DeviceEvent('added', device_info)
                        self._notify_event(event)
                        discovered_count += 1
            except Exception as e:
                logger.error(f"Discovery module {name} failed: {e}")
        
        logger.info(f"Initial discovery completed: {discovered_count} devices found")
    
    def _discover_usb_devices(self) -> List[Dict]:
        """Discover USB devices."""
        devices = []
        
        if not HAS_PYUDEV:
            logger.warning("pyudev not available, USB discovery limited")
            return devices
        
        try:
            context = pyudev.Context()
            for device in context.list_devices(subsystem='usb'):
                if device.device_type == 'usb_device':
                    device_info = {
                        'device_id': f"usb_{device.sys_name}",
                        'type': 'usb',
                        'subsystem': device.subsystem,
                        'sys_name': device.sys_name,
                        'device_node': device.device_node,
                        'vendor_id': device.get('ID_VENDOR_ID'),
                        'product_id': device.get('ID_MODEL_ID'),
                        'vendor': device.get('ID_VENDOR'),
                        'model': device.get('ID_MODEL'),
                        'serial': device.get('ID_SERIAL_SHORT'),
                        'driver': device.get('DRIVER'),
                        'properties': dict(device.properties)
                    }
                    devices.append(device_info)
        except Exception as e:
            logger.error(f"USB device discovery failed: {e}")
        
        return devices
    
    def _discover_network_devices(self) -> List[Dict]:
        """Discover network interfaces and devices."""
        devices = []
        
        if not HAS_PSUTIL:
            logger.warning("psutil not available, network discovery limited")
            return devices
        
        try:
            # Network interfaces
            interfaces = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            
            for interface_name, addresses in interfaces.items():
                stat = stats.get(interface_name)
                device_info = {
                    'device_id': f"network_{interface_name}",
                    'type': 'network_interface',
                    'interface_name': interface_name,
                    'addresses': [
                        {
                            'family': addr.family.name,
                            'address': addr.address,
                            'netmask': addr.netmask,
                            'broadcast': addr.broadcast
                        } for addr in addresses
                    ],
                    'is_up': stat.isup if stat else False,
                    'duplex': stat.duplex.name if stat else 'unknown',
                    'speed': stat.speed if stat else 0,
                    'mtu': stat.mtu if stat else 0
                }
                devices.append(device_info)
        except Exception as e:
            logger.error(f"Network device discovery failed: {e}")
        
        return devices
    
    def _discover_storage_devices(self) -> List[Dict]:
        """Discover storage devices."""
        devices = []
        
        if not HAS_PSUTIL:
            return devices
        
        try:
            # Disk partitions
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    device_info = {
                        'device_id': f"storage_{partition.device.replace('/', '_')}",
                        'type': 'storage',
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'opts': partition.opts,
                        'total_bytes': usage.total,
                        'used_bytes': usage.used,
                        'free_bytes': usage.free,
                        'usage_percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                    }
                    devices.append(device_info)
                except (PermissionError, OSError):
                    # Skip inaccessible partitions
                    continue
        except Exception as e:
            logger.error(f"Storage device discovery failed: {e}")
        
        return devices
    
    def _discover_audio_devices(self) -> List[Dict]:
        """Discover audio devices."""
        devices = []
        
        try:
            import sounddevice as sd
            audio_devices = sd.query_devices()
            
            for i, device in enumerate(audio_devices):
                device_info = {
                    'device_id': f"audio_{i}",
                    'type': 'audio',
                    'index': i,
                    'name': device['name'],
                    'hostapi': device['hostapi'],
                    'max_input_channels': device['max_input_channels'],
                    'max_output_channels': device['max_output_channels'],
                    'default_samplerate': device['default_samplerate']
                }
                devices.append(device_info)
        except ImportError:
            logger.warning("sounddevice not available, audio discovery skipped")
        except Exception as e:
            logger.error(f"Audio device discovery failed: {e}")
        
        return devices
    
    def _discover_video_devices(self) -> List[Dict]:
        """Discover video devices (cameras)."""
        devices = []
        
        try:
            import cv2
            
            # Test first 10 video device indices
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    device_info = {
                        'device_id': f"video_{i}",
                        'type': 'video',
                        'index': i,
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'format': int(cap.get(cv2.CAP_PROP_FORMAT)),
                        'backend': cap.getBackendName()
                    }
                    devices.append(device_info)
                    cap.release()
        except ImportError:
            logger.warning("opencv not available, video discovery skipped")
        except Exception as e:
            logger.error(f"Video device discovery failed: {e}")
        
        return devices
    
    def _monitor_udev(self):
        """Monitor udev events for device changes."""
        if not HAS_PYUDEV:
            logger.warning("pyudev not available, udev monitoring disabled")
            return
        
        try:
            context = pyudev.Context()
            monitor = pyudev.Monitor.from_netlink(context)
            monitor.filter_by('usb')
            
            logger.info("Started udev monitoring")
            
            for device in iter(monitor.poll, None):
                if not self._monitoring:
                    break
                
                try:
                    device_id = f"usb_{device.sys_name}"
                    
                    if device.action == 'add':
                        if device_id not in self._known_devices:
                            self._known_devices.add(device_id)
                            device_info = {
                                'device_id': device_id,
                                'type': 'usb',
                                'subsystem': device.subsystem,
                                'sys_name': device.sys_name,
                                'device_node': device.device_node,
                                'vendor_id': device.get('ID_VENDOR_ID'),
                                'product_id': device.get('ID_MODEL_ID'),
                                'action': device.action
                            }
                            event = DeviceEvent('added', device_info)
                            self._notify_event(event)
                    
                    elif device.action == 'remove':
                        if device_id in self._known_devices:
                            self._known_devices.remove(device_id)
                            device_info = {
                                'device_id': device_id,
                                'type': 'usb',
                                'action': device.action
                            }
                            event = DeviceEvent('removed', device_info)
                            self._notify_event(event)
                
                except Exception as e:
                    logger.error(f"Error processing udev event: {e}")
        
        except Exception as e:
            logger.error(f"udev monitoring failed: {e}")
    
    def _monitor_network(self):
        """Monitor network interface changes."""
        if not HAS_PSUTIL:
            return
        
        logger.info("Started network monitoring")
        last_interfaces = set()
        
        while self._monitoring:
            try:
                current_interfaces = set(psutil.net_if_addrs().keys())
                
                # Check for new interfaces
                new_interfaces = current_interfaces - last_interfaces
                for interface in new_interfaces:
                    device_info = {
                        'device_id': f"network_{interface}",
                        'type': 'network_interface',
                        'interface_name': interface,
                        'action': 'added'
                    }
                    event = DeviceEvent('added', device_info)
                    self._notify_event(event)
                
                # Check for removed interfaces
                removed_interfaces = last_interfaces - current_interfaces
                for interface in removed_interfaces:
                    device_info = {
                        'device_id': f"network_{interface}",
                        'type': 'network_interface',
                        'interface_name': interface,
                        'action': 'removed'
                    }
                    event = DeviceEvent('removed', device_info)
                    self._notify_event(event)
                
                last_interfaces = current_interfaces
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Network monitoring error: {e}")
                time.sleep(10)
    
    def _monitor_storage(self):
        """Monitor storage device changes."""
        if not HAS_PSUTIL:
            return
        
        logger.info("Started storage monitoring")
        last_partitions = set()
        
        while self._monitoring:
            try:
                current_partitions = {p.device for p in psutil.disk_partitions()}
                
                # Check for new partitions
                new_partitions = current_partitions - last_partitions
                for partition in new_partitions:
                    device_info = {
                        'device_id': f"storage_{partition.replace('/', '_')}",
                        'type': 'storage',
                        'device': partition,
                        'action': 'added'
                    }
                    event = DeviceEvent('added', device_info)
                    self._notify_event(event)
                
                # Check for removed partitions
                removed_partitions = last_partitions - current_partitions
                for partition in removed_partitions:
                    device_info = {
                        'device_id': f"storage_{partition.replace('/', '_')}",
                        'type': 'storage',
                        'device': partition,
                        'action': 'removed'
                    }
                    event = DeviceEvent('removed', device_info)
                    self._notify_event(event)
                
                last_partitions = current_partitions
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Storage monitoring error: {e}")
                time.sleep(15)
    
    def _notify_event(self, event: DeviceEvent):
        """Notify all callbacks about a device event."""
        logger.info(f"Device event: {event.event_type} - {event.device_info.get('device_id')}")
        
        # Notify registered callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")
        
        # Notify device manager if available
        if self.device_manager:
            try:
                if event.event_type == 'added':
                    # Device manager should handle adding the device
                    pass
                elif event.event_type == 'removed':
                    # Device manager should handle removing the device
                    pass
            except Exception as e:
                logger.error(f"Device manager notification failed: {e}")
        
        # Publish to message bus if available
        if self.message_bus:
            try:
                topic = f"device.{event.event_type}"
                self.message_bus.publish(topic, event.device_info, source="discovery_service")
            except Exception as e:
                logger.error(f"Message bus notification failed: {e}")
    
    def add_callback(self, callback: Callable[[DeviceEvent], None]):
        """Add a callback for device events."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[DeviceEvent], None]):
        """Remove a callback for device events."""
        try:
            self.callbacks.remove(callback)
        except ValueError:
            pass
    
    def force_rediscovery(self):
        """Force a complete rediscovery of all devices."""
        logger.info("Forcing device rediscovery")
        self._known_devices.clear()
        self._perform_initial_discovery()
    
    def get_known_devices(self) -> Set[str]:
        """Get the set of currently known device IDs."""
        return self._known_devices.copy()
    
    def is_monitoring(self) -> bool:
        """Check if the discovery service is currently monitoring."""
        return self._monitoring

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def device_event_handler(event: DeviceEvent):
        print(f"Device Event: {event.event_type}")
        print(f"  Device ID: {event.device_info.get('device_id')}")
        print(f"  Type: {event.device_info.get('type')}")
        print(f"  Timestamp: {event.timestamp}")
        print()
    
    # Create discovery service
    discovery = DeviceDiscoveryService()
    
    # Add event handler
    discovery.add_callback(device_event_handler)
    
    # Start discovery
    discovery.start_discovery()
    
    try:
        print("Device discovery running. Press Ctrl+C to stop.")
        print(f"Known devices: {len(discovery.get_known_devices())}")
        
        # Keep running and show periodic status
        while True:
            time.sleep(30)
            known_devices = discovery.get_known_devices()
            print(f"Status check - Known devices: {len(known_devices)}")
            
    except KeyboardInterrupt:
        print("\nShutting down device discovery...")
    finally:
        discovery.stop_discovery()
