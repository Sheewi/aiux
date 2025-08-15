"""
Communication Bus - Universal message passing system with AI integration
Provides high-performance inter-device and AI orchestrator communication.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class MessageType(Enum):
    """Message type classification"""
    COMMAND = "command"
    RESPONSE = "response"
    EVENT = "event"
    DATA = "data"
    STATUS = "status"
    AI_REQUEST = "ai_request"
    AI_RESPONSE = "ai_response"

@dataclass
class BusMessage:
    """Universal message structure for communication bus"""
    message_id: str
    topic: str
    message_type: MessageType
    sender: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return self.expires_at is not None and time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'message_id': self.message_id,
            'topic': self.topic,
            'message_type': self.message_type.value,
            'sender': self.sender,
            'data': self.data,
            'timestamp': self.timestamp,
            'priority': self.priority.value,
            'correlation_id': self.correlation_id,
            'expires_at': self.expires_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data['message_id'],
            topic=data['topic'],
            message_type=MessageType(data['message_type']),
            sender=data['sender'],
            data=data['data'],
            timestamp=data.get('timestamp', time.time()),
            priority=MessagePriority(data.get('priority', MessagePriority.NORMAL.value)),
            correlation_id=data.get('correlation_id'),
            expires_at=data.get('expires_at'),
            metadata=data.get('metadata', {})
        )

class MessageFilter:
    """Message filtering for subscriptions"""
    
    def __init__(self, topic_pattern: str = "*", message_type: Optional[MessageType] = None, 
                 sender_pattern: str = "*", min_priority: MessagePriority = MessagePriority.LOW):
        self.topic_pattern = topic_pattern
        self.message_type = message_type
        self.sender_pattern = sender_pattern
        self.min_priority = min_priority
    
    def matches(self, message: BusMessage) -> bool:
        """Check if message matches filter criteria"""
        # Topic pattern matching (simple wildcard support)
        if self.topic_pattern != "*" and not self._pattern_match(message.topic, self.topic_pattern):
            return False
        
        # Message type filtering
        if self.message_type and message.message_type != self.message_type:
            return False
        
        # Sender pattern matching
        if self.sender_pattern != "*" and not self._pattern_match(message.sender, self.sender_pattern):
            return False
        
        # Priority filtering
        if message.priority.value < self.min_priority.value:
            return False
        
        return True
    
    def _pattern_match(self, text: str, pattern: str) -> bool:
        """Simple wildcard pattern matching"""
        if pattern == "*":
            return True
        if "*" not in pattern:
            return text == pattern
        
        # Simple wildcard matching (supports * at beginning/end)
        if pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in text
        elif pattern.startswith("*"):
            return text.endswith(pattern[1:])
        elif pattern.endswith("*"):
            return text.startswith(pattern[:-1])
        else:
            return text == pattern

class CommunicationBus:
    """High-performance communication bus with AI integration"""
    
    def __init__(self, ai_orchestrator=None, max_queue_size: int = 10000):
        self.ai_orchestrator = ai_orchestrator
        self.max_queue_size = max_queue_size
        
        # Message routing
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_filters: Dict[str, MessageFilter] = {}
        
        # Message queues (priority-based)
        self.message_queues: Dict[MessagePriority, deque] = {
            priority: deque(maxlen=max_queue_size) for priority in MessagePriority
        }
        
        # Message processing
        self._running = False
        self._processor_task = None
        self._ai_processor_task = None
        
        # Statistics and monitoring
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_dropped': 0,
            'ai_messages_processed': 0,
            'subscribers_count': 0,
            'average_latency': 0.0
        }
        
        # AI message queue
        self.ai_message_queue = asyncio.Queue(maxsize=1000) if ai_orchestrator else None
        
        # Message history for debugging
        self.message_history: deque = deque(maxlen=1000)
        
        logger.info(f"Communication Bus initialized (AI integration: {ai_orchestrator is not None})")
    
    async def start(self):
        """Start the communication bus"""
        if self._running:
            logger.warning("Communication bus already running")
            return
        
        self._running = True
        logger.info("Starting communication bus...")
        
        # Start message processor
        self._processor_task = asyncio.create_task(self._process_messages())
        
        # Start AI processor if available
        if self.ai_orchestrator:
            self._ai_processor_task = asyncio.create_task(self._process_ai_messages())
    
    async def stop(self):
        """Stop the communication bus"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping communication bus...")
        
        # Cancel processor tasks
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        if self._ai_processor_task:
            self._ai_processor_task.cancel()
            try:
                await self._ai_processor_task
            except asyncio.CancelledError:
                pass
    
    def subscribe(self, topic: str, handler: Callable[[BusMessage], None], 
                 message_filter: Optional[MessageFilter] = None) -> str:
        """Subscribe to messages on a topic"""
        subscriber_id = str(uuid.uuid4())
        
        # Store handler
        self.subscribers[topic].append((subscriber_id, handler))
        
        # Store filter if provided
        if message_filter:
            self.message_filters[subscriber_id] = message_filter
        
        self.stats['subscribers_count'] = sum(len(handlers) for handlers in self.subscribers.values())
        
        logger.info(f"Subscribed to topic '{topic}' with ID {subscriber_id}")
        return subscriber_id
    
    def unsubscribe(self, topic: str, subscriber_id: str):
        """Unsubscribe from a topic"""
        if topic in self.subscribers:
            self.subscribers[topic] = [
                (sid, handler) for sid, handler in self.subscribers[topic] 
                if sid != subscriber_id
            ]
            
            # Remove filter
            if subscriber_id in self.message_filters:
                del self.message_filters[subscriber_id]
            
            self.stats['subscribers_count'] = sum(len(handlers) for handlers in self.subscribers.values())
            logger.info(f"Unsubscribed from topic '{topic}' with ID {subscriber_id}")
    
    async def publish(self, topic: str, data: Any, sender: str = "unknown", 
                     message_type: MessageType = MessageType.DATA,
                     priority: MessagePriority = MessagePriority.NORMAL,
                     correlation_id: Optional[str] = None,
                     ttl_seconds: Optional[float] = None) -> str:
        """Publish a message to a topic"""
        try:
            message_id = str(uuid.uuid4())
            expires_at = time.time() + ttl_seconds if ttl_seconds else None
            
            message = BusMessage(
                message_id=message_id,
                topic=topic,
                message_type=message_type,
                sender=sender,
                data=data,
                priority=priority,
                correlation_id=correlation_id,
                expires_at=expires_at
            )
            
            # Add to appropriate priority queue
            self.message_queues[priority].append(message)
            
            # Track statistics
            self.stats['messages_sent'] += 1
            
            # Add to history
            self.message_history.append(message)
            
            # Check if this should be processed by AI
            if self.ai_orchestrator and self._should_process_with_ai(message):
                await self.ai_message_queue.put(message)
            
            logger.debug(f"Published message {message_id} to topic '{topic}'")
            return message_id
            
        except Exception as e:
            logger.error(f"Error publishing message to topic '{topic}': {e}")
            raise
    
    async def publish_command(self, target: str, command: str, parameters: Dict[str, Any], 
                            sender: str, priority: MessagePriority = MessagePriority.HIGH) -> str:
        """Publish a command message"""
        command_data = {
            'command': command,
            'parameters': parameters,
            'target': target
        }
        
        return await self.publish(
            topic=f"command.{target}",
            data=command_data,
            sender=sender,
            message_type=MessageType.COMMAND,
            priority=priority
        )
    
    async def publish_response(self, original_message: BusMessage, response_data: Any, 
                             sender: str, success: bool = True) -> str:
        """Publish a response to a message"""
        response = {
            'success': success,
            'data': response_data,
            'original_message_id': original_message.message_id
        }
        
        return await self.publish(
            topic=f"response.{original_message.sender}",
            data=response,
            sender=sender,
            message_type=MessageType.RESPONSE,
            correlation_id=original_message.message_id
        )
    
    async def publish_event(self, event_type: str, event_data: Any, sender: str) -> str:
        """Publish an event message"""
        return await self.publish(
            topic=f"event.{event_type}",
            data=event_data,
            sender=sender,
            message_type=MessageType.EVENT
        )
    
    async def request_ai_analysis(self, data: Any, analysis_type: str, sender: str) -> str:
        """Request AI analysis of data"""
        if not self.ai_orchestrator:
            raise ValueError("AI orchestrator not available")
        
        ai_request = {
            'analysis_type': analysis_type,
            'data': data,
            'requested_by': sender
        }
        
        return await self.publish(
            topic="ai.analysis_request",
            data=ai_request,
            sender=sender,
            message_type=MessageType.AI_REQUEST,
            priority=MessagePriority.HIGH
        )
    
    async def _process_messages(self):
        """Process messages from queues"""
        logger.info("Message processor started")
        
        while self._running:
            try:
                # Process messages by priority (highest first)
                message_processed = False
                
                for priority in sorted(MessagePriority, key=lambda p: p.value, reverse=True):
                    queue = self.message_queues[priority]
                    
                    if queue:
                        message = queue.popleft()
                        
                        # Check if message has expired
                        if message.is_expired():
                            self.stats['messages_dropped'] += 1
                            continue
                        
                        # Deliver message to subscribers
                        await self._deliver_message(message)
                        message_processed = True
                        break
                
                # If no messages processed, sleep briefly
                if not message_processed:
                    await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("Message processor stopped")
    
    async def _deliver_message(self, message: BusMessage):
        """Deliver message to subscribers"""
        delivery_start = time.time()
        delivered_count = 0
        
        try:
            # Find subscribers for this topic
            subscribers = self.subscribers.get(message.topic, [])
            
            # Also check wildcard subscribers
            wildcard_subscribers = self.subscribers.get("*", [])
            all_subscribers = subscribers + wildcard_subscribers
            
            for subscriber_id, handler in all_subscribers:
                try:
                    # Apply message filter if exists
                    if subscriber_id in self.message_filters:
                        message_filter = self.message_filters[subscriber_id]
                        if not message_filter.matches(message):
                            continue
                    
                    # Deliver message
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                    
                    delivered_count += 1
                    
                except Exception as e:
                    logger.error(f"Error delivering message to subscriber {subscriber_id}: {e}")
            
            # Update statistics
            self.stats['messages_delivered'] += delivered_count
            
            # Update average latency
            delivery_time = time.time() - delivery_start
            self.stats['average_latency'] = (
                (self.stats['average_latency'] * (self.stats['messages_delivered'] - delivered_count)) + 
                delivery_time
            ) / self.stats['messages_delivered'] if self.stats['messages_delivered'] > 0 else delivery_time
            
            logger.debug(f"Delivered message {message.message_id} to {delivered_count} subscribers")
            
        except Exception as e:
            logger.error(f"Error in message delivery: {e}")
    
    async def _process_ai_messages(self):
        """Process messages that require AI analysis"""
        if not self.ai_orchestrator:
            return
        
        logger.info("AI message processor started")
        
        while self._running:
            try:
                # Wait for AI messages
                message = await asyncio.wait_for(self.ai_message_queue.get(), timeout=1.0)
                
                # Process with AI orchestrator
                ai_result = await self._analyze_message_with_ai(message)
                
                if ai_result:
                    # Publish AI response
                    await self.publish(
                        topic=f"ai.analysis_response.{message.sender}",
                        data=ai_result,
                        sender="ai_orchestrator",
                        message_type=MessageType.AI_RESPONSE,
                        correlation_id=message.message_id
                    )
                    
                    self.stats['ai_messages_processed'] += 1
                
            except asyncio.TimeoutError:
                continue  # No messages to process
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in AI message processor: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("AI message processor stopped")
    
    def _should_process_with_ai(self, message: BusMessage) -> bool:
        """Determine if message should be processed by AI"""
        if not self.ai_orchestrator:
            return False
        
        # Process AI request messages
        if message.message_type == MessageType.AI_REQUEST:
            return True
        
        # Process high priority commands
        if message.message_type == MessageType.COMMAND and message.priority == MessagePriority.CRITICAL:
            return True
        
        # Process error events
        if message.message_type == MessageType.EVENT and "error" in message.topic.lower():
            return True
        
        # Process device status changes
        if message.topic.startswith("device.") and message.message_type == MessageType.STATUS:
            return True
        
        return False
    
    async def _analyze_message_with_ai(self, message: BusMessage) -> Optional[Dict[str, Any]]:
        """Analyze message with AI orchestrator"""
        try:
            analysis_prompt = [
                f"Analyze this communication bus message:",
                f"Topic: {message.topic}",
                f"Type: {message.message_type.value}",
                f"Sender: {message.sender}",
                f"Priority: {message.priority.value}",
                f"Data: {json.dumps(message.data, indent=2) if isinstance(message.data, dict) else str(message.data)}",
                "Provide insights, recommendations, or actions based on this message."
            ]
            
            ai_result = await self.ai_orchestrator.process_conversation(
                analysis_prompt,
                session_id=f"bus_message_{message.message_id}"
            )
            
            if ai_result.get('status') == 'success':
                return {
                    'message_id': message.message_id,
                    'analysis_timestamp': time.time(),
                    'ai_insights': ai_result.get('response', {}).get('insights', []),
                    'recommended_actions': ai_result.get('response', {}).get('recommendations', []),
                    'priority_assessment': ai_result.get('response', {}).get('summary', ''),
                    'orchestration_suggestions': ai_result.get('orchestration_plan', {})
                }
        
        except Exception as e:
            logger.error(f"Error in AI message analysis: {e}")
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication bus statistics"""
        queue_sizes = {
            priority.name: len(queue) for priority, queue in self.message_queues.items()
        }
        
        return {
            **self.stats,
            'queue_sizes': queue_sizes,
            'total_queue_size': sum(queue_sizes.values()),
            'ai_queue_size': self.ai_message_queue.qsize() if self.ai_message_queue else 0,
            'running': self._running,
            'message_history_size': len(self.message_history)
        }
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from history"""
        recent = list(self.message_history)[-count:]
        return [msg.to_dict() for msg in recent]
    
    def clear_statistics(self):
        """Clear statistics counters"""
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_dropped': 0,
            'ai_messages_processed': 0,
            'subscribers_count': len(self.subscribers),
            'average_latency': 0.0
        }
    
    def get_topic_list(self) -> List[str]:
        """Get list of topics with active subscribers"""
        return list(self.subscribers.keys())
    
    def get_subscriber_count(self, topic: str) -> int:
        """Get number of subscribers for a topic"""
        return len(self.subscribers.get(topic, []))

# Compatibility alias for existing code
MessageBus = CommunicationBus
Message = BusMessage