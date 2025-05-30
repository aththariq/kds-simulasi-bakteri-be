"""
Real-time Updates Service for WebSocket Communication
Task 8.3: Real-time Updates Implementation

This module implements the event-driven architecture for handling real-time data updates
with message queuing, delivery confirmation, out-of-order message handling, and performance optimization.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import heapq
import logging
from concurrent.futures import ThreadPoolExecutor

from schemas.websocket_protocol import (
    WebSocketProtocolMessage, MessageType, MessageFactory, 
    Priority, MessageCategory
)

logger = logging.getLogger(__name__)


class UpdateType(Enum):
    """Types of real-time updates."""
    SIMULATION_GENERATION = "simulation_generation"
    SIMULATION_STATUS = "simulation_status"
    PERFORMANCE_METRICS = "performance_metrics"
    POPULATION_CHANGE = "population_change"
    MUTATION_EVENT = "mutation_event"
    FITNESS_UPDATE = "fitness_update"
    ENVIRONMENT_CHANGE = "environment_change"
    SYSTEM_STATUS = "system_status"
    ERROR_NOTIFICATION = "error_notification"
    BATCH_COMPLETION = "batch_completion"


class DeliveryStatus(Enum):
    """Message delivery status tracking."""
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


class BroadcastScope(Enum):
    """Scope for broadcasting messages."""
    GLOBAL = "global"           # All connected clients
    SIMULATION = "simulation"   # All clients subscribed to a simulation
    AUTHENTICATED = "authenticated"  # All authenticated clients
    USER_GROUP = "user_group"   # Specific user group
    SINGLE_CLIENT = "single_client"  # Single target client


@dataclass
class UpdateMessage:
    """Container for real-time update messages with delivery tracking."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    update_type: UpdateType = UpdateType.SIMULATION_GENERATION
    timestamp: datetime = field(default_factory=datetime.now)
    simulation_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    
    # Delivery tracking
    target_clients: Set[str] = field(default_factory=set)
    delivery_status: Dict[str, DeliveryStatus] = field(default_factory=dict)
    attempts: Dict[str, int] = field(default_factory=dict)
    max_attempts: int = 3
    expires_at: Optional[datetime] = None
    
    # Message ordering
    sequence_number: Optional[int] = None
    parent_message_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize delivery tracking for target clients."""
        if self.expires_at is None:
            self.expires_at = self.timestamp + timedelta(minutes=5)
        
        for client_id in self.target_clients:
            self.delivery_status[client_id] = DeliveryStatus.PENDING
            self.attempts[client_id] = 0
    
    def __lt__(self, other):
        """Enable comparison for heap queue operations."""
        if not isinstance(other, UpdateMessage):
            return NotImplemented
        return self.timestamp < other.timestamp
    
    def __eq__(self, other):
        """Enable equality comparison."""
        if not isinstance(other, UpdateMessage):
            return NotImplemented
        return self.id == other.id
    
    def __hash__(self):
        """Enable hashing for use in sets."""
        return hash(self.id)
    
    def add_target_client(self, client_id: str):
        """Add a target client for delivery."""
        self.target_clients.add(client_id)
        self.delivery_status[client_id] = DeliveryStatus.PENDING
        self.attempts[client_id] = 0
    
    def mark_delivered(self, client_id: str):
        """Mark message as delivered to a client."""
        if client_id in self.delivery_status:
            self.delivery_status[client_id] = DeliveryStatus.DELIVERED
    
    def mark_acknowledged(self, client_id: str):
        """Mark message as acknowledged by a client."""
        if client_id in self.delivery_status:
            self.delivery_status[client_id] = DeliveryStatus.ACKNOWLEDGED
    
    def mark_failed(self, client_id: str):
        """Mark delivery as failed for a client."""
        if client_id in self.delivery_status:
            self.attempts[client_id] += 1
            if self.attempts[client_id] >= self.max_attempts:
                self.delivery_status[client_id] = DeliveryStatus.FAILED
            else:
                self.delivery_status[client_id] = DeliveryStatus.PENDING
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return datetime.now() > self.expires_at
    
    def get_pending_clients(self) -> Set[str]:
        """Get clients with pending delivery."""
        return {
            client_id for client_id, status in self.delivery_status.items()
            if status == DeliveryStatus.PENDING
        }
    
    def is_fully_delivered(self) -> bool:
        """Check if message is delivered to all targets."""
        return all(
            status in [DeliveryStatus.DELIVERED, DeliveryStatus.ACKNOWLEDGED]
            for status in self.delivery_status.values()
        )
    
    def to_websocket_message(self, client_id: str) -> WebSocketProtocolMessage:
        """Convert to WebSocket protocol message."""
        return WebSocketProtocolMessage(
            type=MessageType.SIMULATION_UPDATE,
            client_id=client_id,
            simulation_id=self.simulation_id,
            priority=self.priority,
            data={
                "update_id": self.id,
                "update_type": self.update_type.value,
                "sequence_number": self.sequence_number,
                "parent_message_id": self.parent_message_id,
                **self.data
            },
            expires_at=self.expires_at.isoformat() if self.expires_at else None
        )


@dataclass
class ClientSubscription:
    """Client subscription information for real-time updates."""
    client_id: str
    simulation_id: str
    update_types: Set[UpdateType] = field(default_factory=set)
    last_sequence_number: int = 0
    subscription_time: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    # Delivery preferences
    batch_size: int = 1
    delivery_interval: float = 0.1  # seconds
    priority_filter: Set[Priority] = field(default_factory=lambda: {Priority.HIGH, Priority.NORMAL})
    
    def should_receive_update(self, update: UpdateMessage) -> bool:
        """Check if client should receive this update."""
        if not self.is_active:
            return False
        
        if update.simulation_id and update.simulation_id != self.simulation_id:
            return False
        
        if self.update_types and update.update_type not in self.update_types:
            return False
        
        if update.priority not in self.priority_filter:
            return False
        
        return True


class MessageQueue:
    """Priority queue for managing real-time update messages."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue: List[tuple] = []  # (priority_value, timestamp, message)
        self._sequence_counter = 0
        self._lock = asyncio.Lock()
    
    async def enqueue(self, message: UpdateMessage) -> bool:
        """Add message to queue with priority ordering."""
        async with self._lock:
            if len(self._queue) >= self.max_size:
                # Remove lowest priority expired messages
                await self._cleanup_expired()
                
                if len(self._queue) >= self.max_size:
                    logger.warning(f"Message queue full, dropping message {message.id}")
                    return False
            
            # Assign sequence number
            self._sequence_counter += 1
            message.sequence_number = self._sequence_counter
            
            # Calculate priority value (lower is higher priority)
            priority_value = self._get_priority_value(message.priority)
            timestamp = time.time()
            
            heapq.heappush(self._queue, (priority_value, timestamp, message))
            return True
    
    async def dequeue(self) -> Optional[UpdateMessage]:
        """Get highest priority message from queue."""
        async with self._lock:
            while self._queue:
                priority_value, timestamp, message = heapq.heappop(self._queue)
                
                if message.is_expired():
                    logger.debug(f"Skipping expired message {message.id}")
                    continue
                
                return message
            
            return None
    
    async def peek(self) -> Optional[UpdateMessage]:
        """Look at next message without removing it."""
        async with self._lock:
            while self._queue:
                priority_value, timestamp, message = self._queue[0]
                
                if message.is_expired():
                    heapq.heappop(self._queue)
                    continue
                
                return message
            
            return None
    
    async def get_messages_for_client(self, client_id: str, limit: int = 10) -> List[UpdateMessage]:
        """Get pending messages for a specific client."""
        async with self._lock:
            messages = []
            temp_queue = []
            
            while self._queue and len(messages) < limit:
                priority_value, timestamp, message = heapq.heappop(self._queue)
                
                if message.is_expired():
                    continue
                
                if client_id in message.get_pending_clients():
                    messages.append(message)
                
                temp_queue.append((priority_value, timestamp, message))
            
            # Restore queue
            for item in temp_queue:
                heapq.heappush(self._queue, item)
            
            return messages
    
    def _get_priority_value(self, priority: Priority) -> int:
        """Convert priority enum to numeric value for sorting."""
        priority_map = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.NORMAL: 2,
            Priority.LOW: 3
        }
        return priority_map.get(priority, 2)
    
    async def _cleanup_expired(self):
        """Remove expired messages from queue."""
        cleaned_queue = []
        
        while self._queue:
            priority_value, timestamp, message = heapq.heappop(self._queue)
            if not message.is_expired():
                cleaned_queue.append((priority_value, timestamp, message))
        
        self._queue = cleaned_queue
        heapq.heapify(self._queue)
    
    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return len(self._queue)
    
    async def clear(self):
        """Clear all messages from queue."""
        async with self._lock:
            self._queue.clear()
            self._sequence_counter = 0


class OutOfOrderHandler:
    """Handles out-of-order message delivery and reordering."""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.client_buffers: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.expected_sequences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = asyncio.Lock()
    
    async def process_message(self, client_id: str, simulation_id: str, 
                            message: UpdateMessage) -> List[UpdateMessage]:
        """Process message and return ordered messages ready for delivery."""
        async with self._lock:
            buffer_key = f"{client_id}:{simulation_id}"
            buffer = self.client_buffers[client_id][simulation_id]
            expected_seq = self.expected_sequences[client_id][simulation_id]
            
            if message.sequence_number is None:
                # No sequence number, deliver immediately
                return [message]
            
            if message.sequence_number == expected_seq + 1:
                # Expected message, check if buffered messages can be delivered
                ready_messages = [message]
                self.expected_sequences[client_id][simulation_id] = message.sequence_number
                
                # Check buffer for consecutive messages
                while buffer and buffer[0].sequence_number == expected_seq + 1:
                    ready_messages.append(buffer.popleft())
                    expected_seq += 1
                    self.expected_sequences[client_id][simulation_id] = expected_seq
                
                return ready_messages
            
            elif message.sequence_number > expected_seq + 1:
                # Future message, buffer it
                self._insert_ordered(buffer, message)
                
                # Limit buffer size
                while len(buffer) > self.buffer_size:
                    buffer.popleft()
                
                return []
            
            else:
                # Old message, ignore (already processed or expired)
                logger.debug(f"Ignoring old message {message.id} for client {client_id}")
                return []
    
    def _insert_ordered(self, buffer: deque, message: UpdateMessage):
        """Insert message in order by sequence number."""
        # Simple insertion for now, could be optimized with binary search
        inserted = False
        for i, buffered_msg in enumerate(buffer):
            if message.sequence_number < buffered_msg.sequence_number:
                buffer.insert(i, message)
                inserted = True
                break
        
        if not inserted:
            buffer.append(message)
    
    async def reset_client_sequence(self, client_id: str, simulation_id: str):
        """Reset sequence tracking for a client-simulation pair."""
        async with self._lock:
            if client_id in self.client_buffers:
                self.client_buffers[client_id][simulation_id].clear()
            self.expected_sequences[client_id][simulation_id] = 0


class RealTimeUpdateService:
    """Main service for handling real-time updates with event-driven architecture."""
    
    def __init__(self, connection_manager, max_queue_size: int = 10000):
        self.connection_manager = connection_manager
        self.message_queue = MessageQueue(max_queue_size)
        self.out_of_order_handler = OutOfOrderHandler()
        
        # Subscriptions and routing
        self.subscriptions: Dict[str, ClientSubscription] = {}
        self.simulation_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance monitoring
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "delivery_confirmations": 0,
            "average_delivery_time": 0.0,
            "queue_size": 0
        }
        
        # Background tasks
        self._delivery_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def start(self):
        """Start the real-time update service."""
        if self._running:
            return
        
        self._running = True
        self._delivery_task = asyncio.create_task(self._delivery_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Real-time update service started")
    
    async def stop(self):
        """Stop the real-time update service."""
        if not self._running:
            return
        
        self._running = False
        
        if self._delivery_task:
            self._delivery_task.cancel()
            try:
                await self._delivery_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        logger.info("Real-time update service stopped")
    
    async def subscribe_client(self, client_id: str, simulation_id: str, 
                             update_types: Optional[Set[UpdateType]] = None,
                             delivery_preferences: Optional[Dict[str, Any]] = None) -> bool:
        """Subscribe a client to real-time updates."""
        try:
            subscription = ClientSubscription(
                client_id=client_id,
                simulation_id=simulation_id,
                update_types=update_types or set(UpdateType),
            )
            
            # Apply delivery preferences
            if delivery_preferences:
                if "batch_size" in delivery_preferences:
                    subscription.batch_size = delivery_preferences["batch_size"]
                if "delivery_interval" in delivery_preferences:
                    subscription.delivery_interval = delivery_preferences["delivery_interval"]
                if "priority_filter" in delivery_preferences:
                    subscription.priority_filter = set(delivery_preferences["priority_filter"])
            
            self.subscriptions[client_id] = subscription
            self.simulation_subscribers[simulation_id].add(client_id)
            
            logger.info(f"Client {client_id} subscribed to simulation {simulation_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to subscribe client {client_id}: {e}")
            return False
    
    async def unsubscribe_client(self, client_id: str) -> bool:
        """Unsubscribe a client from real-time updates."""
        try:
            if client_id in self.subscriptions:
                subscription = self.subscriptions[client_id]
                self.simulation_subscribers[subscription.simulation_id].discard(client_id)
                del self.subscriptions[client_id]
                
                # Reset out-of-order handling
                await self.out_of_order_handler.reset_client_sequence(
                    client_id, subscription.simulation_id
                )
                
                logger.info(f"Client {client_id} unsubscribed")
                return True
        
        except Exception as e:
            logger.error(f"Failed to unsubscribe client {client_id}: {e}")
        
        return False
    
    async def send_update(self, update: UpdateMessage, 
                         scope: BroadcastScope = BroadcastScope.SIMULATION) -> bool:
        """Send real-time update based on scope."""
        try:
            # Determine target clients based on scope
            if scope == BroadcastScope.GLOBAL:
                target_clients = set(self.subscriptions.keys())
            elif scope == BroadcastScope.SIMULATION:
                if not update.simulation_id:
                    logger.error("Simulation ID required for SIMULATION scope")
                    return False
                target_clients = self.simulation_subscribers.get(update.simulation_id, set())
            elif scope == BroadcastScope.AUTHENTICATED:
                target_clients = {
                    client_id for client_id, sub in self.subscriptions.items()
                    if self.connection_manager.active_connections.get(client_id, {}).get('authenticated', False)
                }
            elif scope == BroadcastScope.SINGLE_CLIENT:
                if not update.target_clients:
                    logger.error("Target clients required for SINGLE_CLIENT scope")
                    return False
                target_clients = update.target_clients
            else:
                logger.error(f"Unsupported broadcast scope: {scope}")
                return False
            
            # Filter clients based on subscriptions
            filtered_clients = set()
            for client_id in target_clients:
                if client_id in self.subscriptions:
                    subscription = self.subscriptions[client_id]
                    if subscription.should_receive_update(update):
                        filtered_clients.add(client_id)
            
            if not filtered_clients:
                logger.debug(f"No clients to receive update {update.id}")
                return True
            
            # Set target clients
            update.target_clients = filtered_clients
            update.__post_init__()  # Initialize delivery tracking
            
            # Enqueue message
            success = await self.message_queue.enqueue(update)
            if success:
                self.stats["messages_sent"] += 1
                logger.debug(f"Update {update.id} enqueued for {len(filtered_clients)} clients")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to send update {update.id}: {e}")
            return False
    
    async def confirm_delivery(self, client_id: str, message_id: str) -> bool:
        """Confirm message delivery from client."""
        try:
            # This would typically update delivery status in the message queue
            # For now, just update stats
            self.stats["delivery_confirmations"] += 1
            logger.debug(f"Delivery confirmed: client {client_id}, message {message_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to confirm delivery: {e}")
            return False
    
    async def _delivery_loop(self):
        """Background task for delivering messages to clients."""
        while self._running:
            try:
                await asyncio.sleep(0.01)  # 10ms delivery interval
                
                # Process messages from queue
                message = await self.message_queue.dequeue()
                if not message:
                    continue
                
                # Deliver to each target client
                for client_id in message.get_pending_clients():
                    try:
                        # Handle out-of-order messages
                        ordered_messages = await self.out_of_order_handler.process_message(
                            client_id, message.simulation_id or "global", message
                        )
                        
                        # Deliver ordered messages
                        for msg in ordered_messages:
                            await self._deliver_to_client(client_id, msg)
                    
                    except Exception as e:
                        logger.error(f"Failed to deliver message to {client_id}: {e}")
                        message.mark_failed(client_id)
                
                # Update stats
                self.stats["queue_size"] = await self.message_queue.size()
            
            except Exception as e:
                logger.error(f"Error in delivery loop: {e}")
                await asyncio.sleep(1)
    
    async def _deliver_to_client(self, client_id: str, message: UpdateMessage):
        """Deliver a message to a specific client."""
        try:
            start_time = time.time()
            
            # Convert to WebSocket message
            ws_message = message.to_websocket_message(client_id)
            
            # Send via connection manager
            success = await self.connection_manager.send_to_client(client_id, ws_message)
            
            if success:
                message.mark_delivered(client_id)
                self.stats["messages_delivered"] += 1
                
                # Update average delivery time
                delivery_time = time.time() - start_time
                self.stats["average_delivery_time"] = (
                    (self.stats["average_delivery_time"] * 0.9) + (delivery_time * 0.1)
                )
            else:
                message.mark_failed(client_id)
                self.stats["messages_failed"] += 1
        
        except Exception as e:
            logger.error(f"Failed to deliver message {message.id} to client {client_id}: {e}")
            message.mark_failed(client_id)
            self.stats["messages_failed"] += 1
    
    async def _cleanup_loop(self):
        """Background task for cleaning up expired messages and subscriptions."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                
                # Cleanup would go here - removing expired messages, 
                # inactive subscriptions, etc.
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "active_subscriptions": len(self.subscriptions),
            "simulation_count": len(self.simulation_subscribers),
            "service_uptime": "N/A"  # Would track actual uptime
        }
    
    async def get_client_subscription(self, client_id: str) -> Optional[ClientSubscription]:
        """Get subscription information for a client."""
        return self.subscriptions.get(client_id) 