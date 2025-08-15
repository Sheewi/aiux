"""
Message Bus - Central communication hub for hardware middleware
Handles pub/sub messaging between AI orchestrator and hardware devices.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import uuid

# Optional imports with graceful fallback
try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Standard message format for the bus."""
    id: str
    topic: str
    payload: Dict[str, Any]
    timestamp: float
    source: str
    target: Optional[str] = None
    priority: int = 5
    ttl: float = 300.0  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(**data)
    
    def is_expired(self) -> bool:
        return time.time() > (self.timestamp + self.ttl)

class MessageBus:
    """Central message bus for hardware middleware communication."""
    
    def __init__(self, transport="internal", **kwargs):
        self.transport = transport
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue = Queue()
        self._running = False
        self._worker_thread = None
        self._lock = threading.Lock()
        
        # Transport-specific initialization
        if transport == "zmq" and HAS_ZMQ:
            self._init_zmq(**kwargs)
        elif transport == "mqtt" and HAS_MQTT:
            self._init_mqtt(**kwargs)
        else:
            self._init_internal()
    
    def _init_internal(self):
        """Initialize internal (in-memory) transport."""
        logger.info("Initialized internal message bus")
    
    def _init_zmq(self, port=5555, **kwargs):
        """Initialize ZeroMQ transport."""
        try:
            self.zmq_context = zmq.Context()
            
            # Publisher socket
            self.zmq_pub = self.zmq_context.socket(zmq.PUB)
            self.zmq_pub.bind(f"tcp://*:{port}")
            
            # Subscriber socket
            self.zmq_sub = self.zmq_context.socket(zmq.SUB)
            self.zmq_sub.connect(f"tcp://localhost:{port}")
            self.zmq_sub.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all
            
            logger.info(f"Initialized ZeroMQ message bus on port {port}")
        except Exception as e:
            logger.error(f"ZeroMQ initialization failed: {e}")
            self._init_internal()
    
    def _init_mqtt(self, broker="localhost", port=1883, **kwargs):
        """Initialize MQTT transport."""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()
            
            logger.info(f"Initialized MQTT message bus at {broker}:{port}")
        except Exception as e:
            logger.error(f"MQTT initialization failed: {e}")
            self._init_internal()
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            client.subscribe("#")  # Subscribe to all topics
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            data = json.loads(msg.payload.decode())
            message = Message.from_dict(data)
            self._route_message(message)
        except Exception as e:
            logger.error(f"Failed to process MQTT message: {e}")
    
    def start(self):
        """Start the message bus."""
        if self._running:
            return
            
        self._running = True
        self._worker_thread = threading.Thread(target=self._message_worker, daemon=True)
        self._worker_thread.start()
        
        logger.info("Message bus started")
    
    def stop(self):
        """Stop the message bus."""
        self._running = False
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        
        # Cleanup transport-specific resources
        if hasattr(self, 'zmq_context'):
            self.zmq_context.term()
        
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        logger.info("Message bus stopped")
    
    def _message_worker(self):
        """Worker thread for processing messages."""
        while self._running:
            try:
                # Check for ZeroMQ messages
                if hasattr(self, 'zmq_sub'):
                    try:
                        topic, payload = self.zmq_sub.recv_multipart(zmq.NOBLOCK)
                        data = json.loads(payload.decode())
                        message = Message.from_dict(data)
                        self._route_message(message)
                    except zmq.Again:
                        pass
                
                # Process internal queue
                try:
                    message = self.message_queue.get_nowait()
                    self._route_message(message)
                except Empty:
                    pass
                
                time.sleep(0.01)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.error(f"Message worker error: {e}")
    
    def _route_message(self, message: Message):
        """Route message to subscribers."""
        if message.is_expired():
            logger.debug(f"Dropping expired message: {message.id}")
            return
        
        with self._lock:
            # Route to specific topic subscribers
            subscribers = self.subscribers.get(message.topic, [])
            
            # Also route to wildcard subscribers
            wildcard_subscribers = self.subscribers.get("*", [])
            
            all_subscribers = subscribers + wildcard_subscribers
        
        for callback in all_subscribers:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Subscriber callback failed: {e}")
    
    def publish(self, topic: str, payload: Dict[str, Any], 
                source: str = "unknown", target: Optional[str] = None, 
                priority: int = 5, ttl: float = 300.0) -> str:
        """Publish a message to the bus."""
        message = Message(
            id=str(uuid.uuid4()),
            topic=topic,
            payload=payload,
            timestamp=time.time(),
            source=source,
            target=target,
            priority=priority,
            ttl=ttl
        )
        
        # Send via transport
        if self.transport == "zmq" and hasattr(self, 'zmq_pub'):
            try:
                self.zmq_pub.send_multipart([
                    topic.encode(),
                    json.dumps(message.to_dict()).encode()
                ])
            except Exception as e:
                logger.error(f"ZeroMQ publish failed: {e}")
        
        elif self.transport == "mqtt" and hasattr(self, 'mqtt_client'):
            try:
                self.mqtt_client.publish(topic, json.dumps(message.to_dict()))
            except Exception as e:
                logger.error(f"MQTT publish failed: {e}")
        
        else:
            # Internal transport
            self.message_queue.put(message)
        
        logger.debug(f"Published message {message.id} to topic {topic}")
        return message.id
    
    def subscribe(self, topic: str, callback: Callable[[Message], None]):
        """Subscribe to messages on a topic."""
        with self._lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)
        
        logger.debug(f"Added subscriber to topic: {topic}")
    
    def unsubscribe(self, topic: str, callback: Callable[[Message], None]):
        """Unsubscribe from messages on a topic."""
        with self._lock:
            if topic in self.subscribers:
                try:
                    self.subscribers[topic].remove(callback)
                    if not self.subscribers[topic]:
                        del self.subscribers[topic]
                except ValueError:
                    pass
        
        logger.debug(f"Removed subscriber from topic: {topic}")
    
    def request_response(self, topic: str, payload: Dict[str, Any], 
                        timeout: float = 30.0) -> Optional[Message]:
        """Send a request and wait for response."""
        response_topic = f"response_{uuid.uuid4()}"
        response_received = threading.Event()
        response_message = None
        
        def response_handler(message: Message):
            nonlocal response_message
            response_message = message
            response_received.set()
        
        # Subscribe to response topic
        self.subscribe(response_topic, response_handler)
        
        try:
            # Send request with response topic
            request_payload = payload.copy()
            request_payload['_response_topic'] = response_topic
            
            self.publish(topic, request_payload)
            
            # Wait for response
            if response_received.wait(timeout):
                return response_message
            else:
                logger.warning(f"Request timeout on topic {topic}")
                return None
                
        finally:
            self.unsubscribe(response_topic, response_handler)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bus statistics."""
        with self._lock:
            stats = {
                'transport': self.transport,
                'running': self._running,
                'subscriber_count': sum(len(subs) for subs in self.subscribers.values()),
                'topics': list(self.subscribers.keys()),
                'queue_size': self.message_queue.qsize() if hasattr(self.message_queue, 'qsize') else 0
            }
        
        return stats

class MessageRouter:
    """Advanced message routing with patterns and filters."""
    
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.routes: List[Dict[str, Any]] = []
        
    def add_route(self, pattern: str, transform: Callable[[Message], Message] = None,
                  filter_func: Callable[[Message], bool] = None,
                  target_topic: str = None):
        """Add a routing rule."""
        route = {
            'pattern': pattern,
            'transform': transform,
            'filter': filter_func,
            'target_topic': target_topic
        }
        self.routes.append(route)
        
        # Subscribe to the pattern
        self.bus.subscribe(pattern, self._route_handler)
    
    def _route_handler(self, message: Message):
        """Handle routing for a message."""
        for route in self.routes:
            if self._matches_pattern(message.topic, route['pattern']):
                # Apply filter
                if route['filter'] and not route['filter'](message):
                    continue
                
                # Apply transformation
                routed_message = message
                if route['transform']:
                    routed_message = route['transform'](message)
                
                # Route to target topic
                if route['target_topic']:
                    self.bus.publish(
                        route['target_topic'],
                        routed_message.payload,
                        source=f"router:{message.source}",
                        target=routed_message.target,
                        priority=routed_message.priority
                    )
    
    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports wildcards)."""
        if pattern == "*":
            return True
        
        if "*" in pattern:
            import re
            regex_pattern = pattern.replace("*", ".*")
            return bool(re.match(regex_pattern, topic))
        
        return topic == pattern

# Device-specific message handlers
class DeviceMessageHandler:
    """Handles device-specific message patterns."""
    
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup standard device message handlers."""
        # Device commands
        self.bus.subscribe("device.command", self._handle_device_command)
        
        # Device telemetry
        self.bus.subscribe("device.telemetry", self._handle_device_telemetry)
        
        # Device status updates
        self.bus.subscribe("device.status", self._handle_device_status)
    
    def _handle_device_command(self, message: Message):
        """Handle device command messages."""
        logger.info(f"Device command: {message.payload}")
        
        # Extract command details
        device_id = message.payload.get('device_id')
        command = message.payload.get('command')
        args = message.payload.get('args', {})
        
        # Process command (placeholder - integrate with DeviceManager)
        result = {
            'command_id': message.id,
            'device_id': device_id,
            'status': 'executed',
            'result': f"Executed {command} with args {args}"
        }
        
        # Send response if requested
        response_topic = message.payload.get('_response_topic')
        if response_topic:
            self.bus.publish(response_topic, result, source="device_handler")
    
    def _handle_device_telemetry(self, message: Message):
        """Handle device telemetry messages."""
        logger.debug(f"Device telemetry: {message.payload.get('device_id')}")
        # Process telemetry (placeholder)
    
    def _handle_device_status(self, message: Message):
        """Handle device status messages."""
        logger.info(f"Device status: {message.payload}")
        # Process status update (placeholder)

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test internal transport
    print("Testing internal transport...")
    bus = MessageBus(transport="internal")
    bus.start()
    
    # Add a subscriber
    def test_subscriber(message: Message):
        print(f"Received: {message.topic} - {message.payload}")
    
    bus.subscribe("test.topic", test_subscriber)
    
    # Publish some messages
    bus.publish("test.topic", {"data": "Hello World"}, source="test")
    bus.publish("test.topic", {"data": "Second message"}, source="test")
    
    time.sleep(1)
    
    # Test request-response
    def echo_handler(message: Message):
        response_topic = message.payload.get('_response_topic')
        if response_topic:
            response = {"echo": message.payload.get('data', 'no data')}
            bus.publish(response_topic, response, source="echo_service")
    
    bus.subscribe("echo", echo_handler)
    
    response = bus.request_response("echo", {"data": "ping"}, timeout=5)
    if response:
        print(f"Got response: {response.payload}")
    else:
        print("No response received")
    
    # Test message router
    router = MessageRouter(bus)
    router.add_route(
        "device.*",
        target_topic="processed.device",
        filter_func=lambda msg: msg.payload.get('priority', 0) > 5
    )
    
    # Publish device messages
    bus.publish("device.sensor", {"priority": 10, "value": 42}, source="sensor")
    bus.publish("device.actuator", {"priority": 3, "action": "move"}, source="actuator")
    
    time.sleep(1)
    
    # Print stats
    stats = bus.get_stats()
    print(f"Bus stats: {json.dumps(stats, indent=2)}")
    
    # Test ZeroMQ if available
    if HAS_ZMQ:
        print("\nTesting ZeroMQ transport...")
        zmq_bus = MessageBus(transport="zmq", port=5556)
        zmq_bus.start()
        
        zmq_bus.subscribe("zmq.test", test_subscriber)
        zmq_bus.publish("zmq.test", {"data": "ZeroMQ message"}, source="zmq_test")
        
        time.sleep(1)
        zmq_bus.stop()
    
    bus.stop()
    print("Tests completed")
