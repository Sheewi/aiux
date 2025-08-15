"""
Telemetry Aggregator - Real-time device telemetry collection and processing
Collects, normalizes, and aggregates telemetry data from all hardware devices.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import statistics
import uuid

# Optional imports with graceful fallback
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

@dataclass
class TelemetryReading:
    """Represents a single telemetry reading."""
    device_id: str
    metric_name: str
    value: Any
    unit: str
    timestamp: float
    quality: float = 1.0  # 0.0 to 1.0, where 1.0 is perfect quality
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TelemetryReading':
        return cls(**data)

@dataclass
class AggregatedMetric:
    """Represents aggregated telemetry data."""
    device_id: str
    metric_name: str
    values: List[float]
    timestamps: List[float]
    unit: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    first_timestamp: float
    last_timestamp: float
    quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class TelemetryBuffer:
    """Thread-safe circular buffer for telemetry readings."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add(self, reading: TelemetryReading):
        """Add a telemetry reading to the buffer."""
        with self._lock:
            self.buffer.append(reading)
    
    def get_recent(self, count: int = None) -> List[TelemetryReading]:
        """Get recent readings from the buffer."""
        with self._lock:
            if count is None:
                return list(self.buffer)
            else:
                return list(self.buffer)[-count:]
    
    def get_since(self, timestamp: float) -> List[TelemetryReading]:
        """Get readings since a specific timestamp."""
        with self._lock:
            return [r for r in self.buffer if r.timestamp >= timestamp]
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self.buffer)

class TelemetryCollector:
    """Base class for device-specific telemetry collectors."""
    
    def __init__(self, device_id: str, collection_interval: float = 1.0):
        self.device_id = device_id
        self.collection_interval = collection_interval
        self._collecting = False
        self._collection_thread = None
        self.callbacks: List[Callable[[TelemetryReading], None]] = []
    
    def start_collection(self):
        """Start telemetry collection."""
        if self._collecting:
            return
        
        self._collecting = True
        self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collection_thread.start()
        logger.info(f"Started telemetry collection for {self.device_id}")
    
    def stop_collection(self):
        """Stop telemetry collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info(f"Stopped telemetry collection for {self.device_id}")
    
    def _collect_loop(self):
        """Main collection loop."""
        while self._collecting:
            try:
                readings = self.collect_metrics()
                for reading in readings:
                    self._notify_callbacks(reading)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Telemetry collection error for {self.device_id}: {e}")
                time.sleep(self.collection_interval * 2)  # Back off on error
    
    def collect_metrics(self) -> List[TelemetryReading]:
        """Collect metrics from the device. Override in subclasses."""
        return []
    
    def add_callback(self, callback: Callable[[TelemetryReading], None]):
        """Add a callback for telemetry readings."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[TelemetryReading], None]):
        """Remove a callback."""
        try:
            self.callbacks.remove(callback)
        except ValueError:
            pass
    
    def _notify_callbacks(self, reading: TelemetryReading):
        """Notify all callbacks of a new reading."""
        for callback in self.callbacks:
            try:
                callback(reading)
            except Exception as e:
                logger.error(f"Telemetry callback error: {e}")

class SystemTelemetryCollector(TelemetryCollector):
    """Collects system-level telemetry using psutil."""
    
    def __init__(self, device_id: str = "system", collection_interval: float = 1.0):
        super().__init__(device_id, collection_interval)
        self._last_network_io = None
        self._last_disk_io = None
    
    def collect_metrics(self) -> List[TelemetryReading]:
        """Collect system metrics."""
        if not HAS_PSUTIL:
            return []
        
        readings = []
        timestamp = time.time()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            readings.append(TelemetryReading(
                self.device_id, "cpu_percent", cpu_percent, "%", timestamp,
                quality=1.0, tags={"type": "system"}
            ))
            
            cpu_count = psutil.cpu_count()
            readings.append(TelemetryReading(
                self.device_id, "cpu_count", cpu_count, "cores", timestamp,
                quality=1.0, tags={"type": "system"}
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            readings.extend([
                TelemetryReading(self.device_id, "memory_total", memory.total, "bytes", timestamp),
                TelemetryReading(self.device_id, "memory_used", memory.used, "bytes", timestamp),
                TelemetryReading(self.device_id, "memory_free", memory.free, "bytes", timestamp),
                TelemetryReading(self.device_id, "memory_percent", memory.percent, "%", timestamp)
            ])
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            readings.extend([
                TelemetryReading(self.device_id, "disk_total", disk_usage.total, "bytes", timestamp),
                TelemetryReading(self.device_id, "disk_used", disk_usage.used, "bytes", timestamp),
                TelemetryReading(self.device_id, "disk_free", disk_usage.free, "bytes", timestamp),
                TelemetryReading(self.device_id, "disk_percent", 
                               (disk_usage.used / disk_usage.total) * 100, "%", timestamp)
            ])
            
            # Network I/O rates
            network_io = psutil.net_io_counters()
            if self._last_network_io:
                time_delta = timestamp - self._last_network_io[1]
                if time_delta > 0:
                    bytes_sent_rate = (network_io.bytes_sent - self._last_network_io[0].bytes_sent) / time_delta
                    bytes_recv_rate = (network_io.bytes_recv - self._last_network_io[0].bytes_recv) / time_delta
                    
                    readings.extend([
                        TelemetryReading(self.device_id, "network_tx_rate", bytes_sent_rate, "bytes/sec", timestamp),
                        TelemetryReading(self.device_id, "network_rx_rate", bytes_recv_rate, "bytes/sec", timestamp)
                    ])
            
            self._last_network_io = (network_io, timestamp)
            
            # Disk I/O rates
            disk_io = psutil.disk_io_counters()
            if disk_io and self._last_disk_io:
                time_delta = timestamp - self._last_disk_io[1]
                if time_delta > 0:
                    read_rate = (disk_io.read_bytes - self._last_disk_io[0].read_bytes) / time_delta
                    write_rate = (disk_io.write_bytes - self._last_disk_io[0].write_bytes) / time_delta
                    
                    readings.extend([
                        TelemetryReading(self.device_id, "disk_read_rate", read_rate, "bytes/sec", timestamp),
                        TelemetryReading(self.device_id, "disk_write_rate", write_rate, "bytes/sec", timestamp)
                    ])
            
            if disk_io:
                self._last_disk_io = (disk_io, timestamp)
            
            # Temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                for sensor_name, sensors in temps.items():
                    for i, sensor in enumerate(sensors):
                        readings.append(TelemetryReading(
                            self.device_id, f"temperature_{sensor_name}_{i}", 
                            sensor.current, "celsius", timestamp,
                            tags={"sensor": sensor_name, "label": sensor.label or f"sensor_{i}"}
                        ))
            except (AttributeError, OSError):
                pass  # Temperature sensors not available
            
        except Exception as e:
            logger.error(f"System telemetry collection error: {e}")
        
        return readings

class NetworkTelemetryCollector(TelemetryCollector):
    """Collects network interface telemetry."""
    
    def __init__(self, interface_name: str, collection_interval: float = 1.0):
        super().__init__(f"network_{interface_name}", collection_interval)
        self.interface_name = interface_name
        self._last_stats = None
    
    def collect_metrics(self) -> List[TelemetryReading]:
        """Collect network interface metrics."""
        if not HAS_PSUTIL:
            return []
        
        readings = []
        timestamp = time.time()
        
        try:
            # Interface statistics
            stats = psutil.net_if_stats()
            if self.interface_name in stats:
                stat = stats[self.interface_name]
                readings.extend([
                    TelemetryReading(self.device_id, "is_up", int(stat.isup), "boolean", timestamp),
                    TelemetryReading(self.device_id, "speed", stat.speed, "mbps", timestamp),
                    TelemetryReading(self.device_id, "mtu", stat.mtu, "bytes", timestamp)
                ])
            
            # I/O counters per interface
            io_counters = psutil.net_io_counters(pernic=True)
            if self.interface_name in io_counters:
                io = io_counters[self.interface_name]
                
                # Calculate rates if we have previous data
                if self._last_stats:
                    time_delta = timestamp - self._last_stats[1]
                    if time_delta > 0:
                        last_io = self._last_stats[0]
                        
                        tx_rate = (io.bytes_sent - last_io.bytes_sent) / time_delta
                        rx_rate = (io.bytes_recv - last_io.bytes_recv) / time_delta
                        tx_packets_rate = (io.packets_sent - last_io.packets_sent) / time_delta
                        rx_packets_rate = (io.packets_recv - last_io.packets_recv) / time_delta
                        
                        readings.extend([
                            TelemetryReading(self.device_id, "tx_rate", tx_rate, "bytes/sec", timestamp),
                            TelemetryReading(self.device_id, "rx_rate", rx_rate, "bytes/sec", timestamp),
                            TelemetryReading(self.device_id, "tx_packets_rate", tx_packets_rate, "packets/sec", timestamp),
                            TelemetryReading(self.device_id, "rx_packets_rate", rx_packets_rate, "packets/sec", timestamp)
                        ])
                
                # Total counters
                readings.extend([
                    TelemetryReading(self.device_id, "tx_bytes", io.bytes_sent, "bytes", timestamp),
                    TelemetryReading(self.device_id, "rx_bytes", io.bytes_recv, "bytes", timestamp),
                    TelemetryReading(self.device_id, "tx_packets", io.packets_sent, "packets", timestamp),
                    TelemetryReading(self.device_id, "rx_packets", io.packets_recv, "packets", timestamp),
                    TelemetryReading(self.device_id, "tx_errors", io.errin, "errors", timestamp),
                    TelemetryReading(self.device_id, "rx_errors", io.errout, "errors", timestamp)
                ])
                
                self._last_stats = (io, timestamp)
        
        except Exception as e:
            logger.error(f"Network telemetry collection error for {self.interface_name}: {e}")
        
        return readings

class TelemetryAggregator:
    """Central telemetry aggregation and processing service."""
    
    def __init__(self, buffer_size: int = 10000, aggregation_window: int = 60):
        self.buffer_size = buffer_size
        self.aggregation_window = aggregation_window  # seconds
        
        # Telemetry storage
        self.buffers: Dict[str, TelemetryBuffer] = defaultdict(
            lambda: TelemetryBuffer(buffer_size)
        )
        
        # Collectors
        self.collectors: Dict[str, TelemetryCollector] = {}
        
        # Aggregation state
        self._aggregating = False
        self._aggregation_thread = None
        
        # Callbacks for aggregated data
        self.aggregation_callbacks: List[Callable[[AggregatedMetric], None]] = []
        
        # Message bus for publishing telemetry
        self.message_bus = None
    
    def set_message_bus(self, message_bus):
        """Set the message bus for publishing telemetry."""
        self.message_bus = message_bus
    
    def add_collector(self, collector: TelemetryCollector):
        """Add a telemetry collector."""
        collector.add_callback(self._on_telemetry_reading)
        self.collectors[collector.device_id] = collector
        logger.info(f"Added telemetry collector for {collector.device_id}")
    
    def remove_collector(self, device_id: str):
        """Remove a telemetry collector."""
        if device_id in self.collectors:
            collector = self.collectors.pop(device_id)
            collector.stop_collection()
            logger.info(f"Removed telemetry collector for {device_id}")
    
    def start_collection(self):
        """Start all telemetry collectors."""
        for collector in self.collectors.values():
            collector.start_collection()
        logger.info("Started all telemetry collectors")
    
    def stop_collection(self):
        """Stop all telemetry collectors."""
        for collector in self.collectors.values():
            collector.stop_collection()
        logger.info("Stopped all telemetry collectors")
    
    def start_aggregation(self):
        """Start telemetry aggregation."""
        if self._aggregating:
            return
        
        self._aggregating = True
        self._aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._aggregation_thread.start()
        logger.info("Started telemetry aggregation")
    
    def stop_aggregation(self):
        """Stop telemetry aggregation."""
        self._aggregating = False
        if self._aggregation_thread:
            self._aggregation_thread.join(timeout=5)
        logger.info("Stopped telemetry aggregation")
    
    def _on_telemetry_reading(self, reading: TelemetryReading):
        """Handle incoming telemetry reading."""
        # Store in buffer
        buffer_key = f"{reading.device_id}:{reading.metric_name}"
        self.buffers[buffer_key].add(reading)
        
        # Publish to message bus if available
        if self.message_bus:
            try:
                topic = f"telemetry.{reading.device_id}.{reading.metric_name}"
                self.message_bus.publish(topic, reading.to_dict(), source="telemetry_aggregator")
            except Exception as e:
                logger.error(f"Failed to publish telemetry: {e}")
    
    def _aggregation_loop(self):
        """Main aggregation loop."""
        while self._aggregating:
            try:
                current_time = time.time()
                window_start = current_time - self.aggregation_window
                
                # Process each buffer
                for buffer_key, buffer in self.buffers.items():
                    device_id, metric_name = buffer_key.split(':', 1)
                    
                    # Get readings in the current window
                    readings = buffer.get_since(window_start)
                    
                    if len(readings) > 1:  # Need at least 2 readings for meaningful aggregation
                        aggregated = self._aggregate_readings(device_id, metric_name, readings)
                        
                        # Notify callbacks
                        for callback in self.aggregation_callbacks:
                            try:
                                callback(aggregated)
                            except Exception as e:
                                logger.error(f"Aggregation callback error: {e}")
                        
                        # Publish aggregated data
                        if self.message_bus:
                            try:
                                topic = f"telemetry.aggregated.{device_id}.{metric_name}"
                                self.message_bus.publish(topic, aggregated.to_dict(), 
                                                       source="telemetry_aggregator")
                            except Exception as e:
                                logger.error(f"Failed to publish aggregated telemetry: {e}")
                
                time.sleep(self.aggregation_window)  # Aggregate once per window
                
            except Exception as e:
                logger.error(f"Aggregation loop error: {e}")
                time.sleep(10)
    
    def _aggregate_readings(self, device_id: str, metric_name: str, 
                          readings: List[TelemetryReading]) -> AggregatedMetric:
        """Aggregate a list of telemetry readings."""
        # Extract numeric values
        values = []
        timestamps = []
        unit = readings[0].unit
        quality_scores = []
        
        for reading in readings:
            try:
                if isinstance(reading.value, (int, float)):
                    values.append(float(reading.value))
                    timestamps.append(reading.timestamp)
                    quality_scores.append(reading.quality)
            except (ValueError, TypeError):
                continue
        
        if not values:
            # Return empty aggregation for non-numeric data
            return AggregatedMetric(
                device_id=device_id,
                metric_name=metric_name,
                values=[],
                timestamps=timestamps,
                unit=unit,
                count=0,
                min_value=0,
                max_value=0,
                mean_value=0,
                median_value=0,
                std_dev=0,
                first_timestamp=timestamps[0] if timestamps else 0,
                last_timestamp=timestamps[-1] if timestamps else 0,
                quality_score=0
            )
        
        # Calculate statistics
        count = len(values)
        min_value = min(values)
        max_value = max(values)
        mean_value = statistics.mean(values)
        median_value = statistics.median(values)
        std_dev = statistics.stdev(values) if count > 1 else 0
        quality_score = statistics.mean(quality_scores) if quality_scores else 1.0
        
        return AggregatedMetric(
            device_id=device_id,
            metric_name=metric_name,
            values=values,
            timestamps=timestamps,
            unit=unit,
            count=count,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            median_value=median_value,
            std_dev=std_dev,
            first_timestamp=timestamps[0],
            last_timestamp=timestamps[-1],
            quality_score=quality_score
        )
    
    def get_recent_readings(self, device_id: str, metric_name: str, 
                          count: int = 100) -> List[TelemetryReading]:
        """Get recent readings for a specific device and metric."""
        buffer_key = f"{device_id}:{metric_name}"
        if buffer_key in self.buffers:
            return self.buffers[buffer_key].get_recent(count)
        return []
    
    def get_readings_since(self, device_id: str, metric_name: str, 
                          timestamp: float) -> List[TelemetryReading]:
        """Get readings since a specific timestamp."""
        buffer_key = f"{device_id}:{metric_name}"
        if buffer_key in self.buffers:
            return self.buffers[buffer_key].get_since(timestamp)
        return []
    
    def get_current_values(self) -> Dict[str, Dict[str, Any]]:
        """Get current values for all metrics."""
        current_values = {}
        
        for buffer_key, buffer in self.buffers.items():
            device_id, metric_name = buffer_key.split(':', 1)
            
            if device_id not in current_values:
                current_values[device_id] = {}
            
            recent_readings = buffer.get_recent(1)
            if recent_readings:
                reading = recent_readings[0]
                current_values[device_id][metric_name] = {
                    'value': reading.value,
                    'unit': reading.unit,
                    'timestamp': reading.timestamp,
                    'quality': reading.quality
                }
        
        return current_values
    
    def add_aggregation_callback(self, callback: Callable[[AggregatedMetric], None]):
        """Add a callback for aggregated metrics."""
        self.aggregation_callbacks.append(callback)
    
    def export_telemetry(self, filename: str, hours: int = 24):
        """Export telemetry data to file."""
        cutoff_time = time.time() - (hours * 3600)
        export_data = {}
        
        for buffer_key, buffer in self.buffers.items():
            device_id, metric_name = buffer_key.split(':', 1)
            
            if device_id not in export_data:
                export_data[device_id] = {}
            
            readings = buffer.get_since(cutoff_time)
            export_data[device_id][metric_name] = [r.to_dict() for r in readings]
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {hours}h of telemetry data to {filename}")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def aggregation_handler(metric: AggregatedMetric):
        print(f"Aggregated: {metric.device_id}.{metric.metric_name}")
        print(f"  Count: {metric.count}, Mean: {metric.mean_value:.2f} {metric.unit}")
        print(f"  Range: {metric.min_value:.2f} - {metric.max_value:.2f}")
        print()
    
    # Create telemetry aggregator
    aggregator = TelemetryAggregator(aggregation_window=10)  # 10 second window for demo
    aggregator.add_aggregation_callback(aggregation_handler)
    
    # Add system telemetry collector
    system_collector = SystemTelemetryCollector(collection_interval=2)
    aggregator.add_collector(system_collector)
    
    # Add network collectors for available interfaces
    if HAS_PSUTIL:
        interfaces = psutil.net_if_addrs()
        for interface in list(interfaces.keys())[:2]:  # First 2 interfaces
            net_collector = NetworkTelemetryCollector(interface, collection_interval=3)
            aggregator.add_collector(net_collector)
    
    # Start everything
    aggregator.start_collection()
    aggregator.start_aggregation()
    
    try:
        print("Telemetry collection running. Press Ctrl+C to stop.")
        
        # Show periodic status
        while True:
            time.sleep(30)
            current_values = aggregator.get_current_values()
            print(f"Current telemetry - {len(current_values)} devices")
            
            for device_id, metrics in current_values.items():
                print(f"  {device_id}: {len(metrics)} metrics")
    
    except KeyboardInterrupt:
        print("\nShutting down telemetry collection...")
    finally:
        aggregator.stop_collection()
        aggregator.stop_aggregation()
        
        # Export data
        aggregator.export_telemetry("telemetry_export.json", hours=1)
