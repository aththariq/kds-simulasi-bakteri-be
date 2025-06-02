"""
Intelligent WebSocket Reconnection Service

This module provides comprehensive reconnection capabilities with:
- Exponential backoff with jitter
- Session state persistence
- Message queuing during disconnection
- Network condition adaptation
- Integration with existing error handling
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime, timedelta
import uuid

from .websocket_error_handler import ErrorHandler, ErrorCode, ErrorSeverity, ErrorCategory, ErrorContext


class ReconnectionState(Enum):
    """States for reconnection tracking."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    DISABLED = "disabled"


class NetworkCondition(Enum):
    """Network condition detection."""
    EXCELLENT = "excellent"  # < 50ms latency
    GOOD = "good"           # 50-200ms latency
    POOR = "poor"           # 200-1000ms latency
    UNSTABLE = "unstable"   # > 1000ms or frequent disconnects


@dataclass
class ReconnectionConfig:
    """Configuration for reconnection behavior."""
    # Backoff parameters
    initial_delay: float = 1.0  # seconds
    max_delay: float = 300.0    # 5 minutes max
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.3  # ±30% randomization
    
    # Retry limits
    max_retries: int = 10
    reset_after_success: int = 30  # seconds
    
    # Network adaptation
    enable_network_detection: bool = True
    poor_network_multiplier: float = 2.0
    unstable_network_multiplier: float = 3.0
    
    # Session persistence
    enable_session_persistence: bool = True
    message_queue_limit: int = 1000
    session_timeout: timedelta = timedelta(hours=1)


@dataclass
class SessionState:
    """Persistent session state for recovery."""
    client_id: str
    auth_token: Optional[str] = None
    subscriptions: Set[str] = field(default_factory=set)
    last_message_id: Optional[str] = None
    queued_messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_expired(self, timeout: timedelta) -> bool:
        """Check if session has expired."""
        return datetime.now() - self.created_at > timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session state."""
        return {
            'client_id': self.client_id,
            'auth_token': self.auth_token,
            'subscriptions': list(self.subscriptions),
            'last_message_id': self.last_message_id,
            'queued_messages': self.queued_messages,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Deserialize session state."""
        return cls(
            client_id=data['client_id'],
            auth_token=data.get('auth_token'),
            subscriptions=set(data.get('subscriptions', [])),
            last_message_id=data.get('last_message_id'),
            queued_messages=data.get('queued_messages', []),
            created_at=datetime.fromisoformat(data['created_at'])
        )


class ExponentialBackoffStrategy:
    """Implements exponential backoff with jitter and network adaptation."""
    
    def __init__(self, config: ReconnectionConfig):
        self.config = config
        self.retry_count = 0
        self.last_success_time = time.time()
        
    def get_delay(self, network_condition: NetworkCondition = NetworkCondition.GOOD) -> float:
        """Calculate next reconnection delay."""
        if self.retry_count == 0:
            return 0  # Immediate first retry
        
        # Base exponential backoff
        base_delay = min(
            self.config.initial_delay * (self.config.backoff_multiplier ** (self.retry_count - 1)),
            self.config.max_delay
        )
        
        # Apply network condition multiplier
        if network_condition == NetworkCondition.POOR:
            base_delay *= self.config.poor_network_multiplier
        elif network_condition == NetworkCondition.UNSTABLE:
            base_delay *= self.config.unstable_network_multiplier
        
        # Add jitter to prevent thundering herd
        jitter = base_delay * self.config.jitter_factor * (random.random() * 2 - 1)
        final_delay = max(0, base_delay + jitter)
        
        return final_delay
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
    
    def reset_on_success(self) -> None:
        """Reset backoff after successful connection."""
        self.retry_count = 0
        self.last_success_time = time.time()
    
    def should_retry(self) -> bool:
        """Check if should attempt another retry."""
        return self.retry_count < self.config.max_retries
    
    def should_reset(self) -> bool:
        """Check if backoff should reset due to time since last success."""
        return (time.time() - self.last_success_time) > self.config.reset_after_success


class NetworkConditionDetector:
    """Detects network conditions for adaptive reconnection."""
    
    def __init__(self):
        self.latency_samples: List[float] = []
        self.disconnect_count = 0
        self.last_disconnect_time = 0
        
    def record_latency(self, latency_ms: float) -> None:
        """Record network latency sample."""
        self.latency_samples.append(latency_ms)
        # Keep only recent samples
        if len(self.latency_samples) > 10:
            self.latency_samples.pop(0)
    
    def record_disconnect(self) -> None:
        """Record network disconnection."""
        current_time = time.time()
        if current_time - self.last_disconnect_time < 300:  # 5 minutes
            self.disconnect_count += 1
        else:
            self.disconnect_count = 1
        self.last_disconnect_time = current_time
    
    def get_condition(self) -> NetworkCondition:
        """Determine current network condition."""
        # Check for unstable network (frequent disconnects) first
        if self.disconnect_count >= 3:
            return NetworkCondition.UNSTABLE
        
        if not self.latency_samples:
            return NetworkCondition.GOOD
        
        avg_latency = sum(self.latency_samples) / len(self.latency_samples)
        
        # Classify by latency
        if avg_latency < 50:
            return NetworkCondition.EXCELLENT
        elif avg_latency < 200:
            return NetworkCondition.GOOD
        else:
            return NetworkCondition.POOR


class ReconnectionManager:
    """Central manager for WebSocket reconnection logic."""
    
    def __init__(
        self,
        config: Optional[ReconnectionConfig] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        self.config = config or ReconnectionConfig()
        self.error_handler = error_handler or ErrorHandler()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.state = ReconnectionState.DISCONNECTED
        self.backoff_strategy = ExponentialBackoffStrategy(self.config)
        self.network_detector = NetworkConditionDetector() if self.config.enable_network_detection else None
        
        # Session management
        self.sessions: Dict[str, SessionState] = {}
        
        # Event handlers
        self.reconnection_callbacks: List[Callable[[ReconnectionState], None]] = []
        self.message_replay_callback: Optional[Callable[[str, List[Dict[str, Any]]], None]] = None
        
        # Background tasks
        self._reconnection_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'total_reconnections': 0,
            'successful_reconnections': 0,
            'failed_reconnections': 0,
            'messages_queued': 0,
            'sessions_recovered': 0
        }
    
    async def start(self) -> None:
        """Start the reconnection manager."""
        self.logger.info("Starting reconnection manager")
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
    
    async def stop(self) -> None:
        """Stop the reconnection manager."""
        self.logger.info("Stopping reconnection manager")
        
        # Cancel background tasks
        if self._reconnection_task:
            self._reconnection_task.cancel()
            try:
                await self._reconnection_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    def register_reconnection_callback(self, callback: Callable[[ReconnectionState], None]) -> None:
        """Register callback for state changes."""
        self.reconnection_callbacks.append(callback)
    
    def register_message_replay_callback(self, callback: Callable[[str, List[Dict[str, Any]]], None]) -> None:
        """Register callback for message replay."""
        self.message_replay_callback = callback
    
    def record_latency(self, latency_ms: float) -> None:
        """Record network latency for condition detection."""
        if self.network_detector:
            self.network_detector.record_latency(latency_ms)
    
    def save_session(self, client_id: str, auth_token: Optional[str] = None, 
                    subscriptions: Optional[Set[str]] = None) -> None:
        """Save session state for recovery."""
        if not self.config.enable_session_persistence:
            return
        
        if client_id not in self.sessions:
            self.sessions[client_id] = SessionState(client_id=client_id)
        
        session = self.sessions[client_id]
        if auth_token:
            session.auth_token = auth_token
        if subscriptions:
            session.subscriptions = subscriptions
        
        self.logger.debug(f"Saved session for client {client_id}")
    
    def queue_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Queue message for replay after reconnection."""
        if not self.config.enable_session_persistence:
            return False
        
        if client_id not in self.sessions:
            self.sessions[client_id] = SessionState(client_id=client_id)
        
        session = self.sessions[client_id]
        if len(session.queued_messages) >= self.config.message_queue_limit:
            # Remove oldest message
            session.queued_messages.pop(0)
        
        session.queued_messages.append(message)
        self.stats['messages_queued'] += 1
        
        self.logger.debug(f"Queued message for client {client_id} (queue size: {len(session.queued_messages)})")
        return True
    
    async def handle_disconnect(self, client_id: str, reason: str = "Unknown") -> None:
        """Handle client disconnection and initiate reconnection."""
        self.logger.info(f"Handling disconnect for client {client_id}: {reason}")
        
        # Record disconnect for network condition detection
        if self.network_detector:
            self.network_detector.record_disconnect()
        
        # Update state
        old_state = self.state
        self.state = ReconnectionState.DISCONNECTED
        
        # Notify callbacks
        await self._notify_state_change(old_state)
        
        # Log error
        await self.error_handler.handle_error(
            ErrorCode.CONNECTION_LOST,
            f"Client {client_id} disconnected: {reason}",
            ErrorSeverity.MEDIUM,
            category=ErrorCategory.CONNECTION,
            context=ErrorContext(client_id=client_id, metadata={"reason": reason})
        )
        
        # Potentially notify error handler about permanent failure
        if self.error_handler:
            await self.error_handler.handle_error(
                error_code=ErrorCode.CONNECTION_LOST,
                message=f"Client {client_id} permanently disconnected after max retries. Reason: {reason}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.CONNECTION,
                context=ErrorContext(client_id=client_id, metadata={"reason": reason, "retries": self.backoff_strategy.retry_count})
            )
        
        # Start reconnection process
        if self._reconnection_task is None or self._reconnection_task.done():
            self._reconnection_task = asyncio.create_task(self._reconnection_loop(client_id))
    
    async def handle_successful_connection(self, client_id: str) -> Optional[SessionState]:
        """Handle successful reconnection and session recovery."""
        self.logger.info(f"Successful connection for client {client_id}")
        
        # Update state
        old_state = self.state
        self.state = ReconnectionState.CONNECTED
        
        # Reset backoff strategy
        self.backoff_strategy.reset_on_success()
        
        # Update statistics
        self.stats['successful_reconnections'] += 1
        
        # Notify callbacks
        await self._notify_state_change(old_state)
        
        # Recover session if available
        session = None
        if self.config.enable_session_persistence and client_id in self.sessions:
            session = self.sessions[client_id]
            if not session.is_expired(self.config.session_timeout):
                self.stats['sessions_recovered'] += 1
                
                # Replay queued messages
                if session.queued_messages and self.message_replay_callback:
                    self.message_replay_callback(client_id, session.queued_messages)
                    session.queued_messages.clear()
                
                self.logger.info(f"Recovered session for client {client_id}")
            else:
                # Remove expired session
                del self.sessions[client_id]
                session = None
                self.logger.info(f"Session expired for client {client_id}")
        
        return session
    
    async def _reconnection_loop(self, client_id: str) -> None:
        """Main reconnection loop with exponential backoff."""
        self.logger.info(f"Starting reconnection loop for client {client_id}")
        
        while self.state == ReconnectionState.DISCONNECTED and self.backoff_strategy.should_retry():
            try:
                # Update state to reconnecting
                old_state = self.state
                self.state = ReconnectionState.RECONNECTING
                await self._notify_state_change(old_state)
                
                # Calculate delay based on network conditions
                network_condition = NetworkCondition.GOOD
                if self.network_detector:
                    network_condition = self.network_detector.get_condition()
                
                delay = self.backoff_strategy.get_delay(network_condition)
                
                if delay > 0:
                    self.logger.info(f"Waiting {delay:.2f}s before reconnection attempt {self.backoff_strategy.retry_count + 1}")
                    await asyncio.sleep(delay)
                
                # Increment retry count
                self.backoff_strategy.increment_retry()
                self.stats['total_reconnections'] += 1
                
                # Attempt reconnection (this would be handled by the WebSocket connection logic)
                self.logger.info(f"Attempting reconnection {self.backoff_strategy.retry_count} for client {client_id}")
                
                # If we reach here and state is still RECONNECTING, the attempt failed
                if self.state == ReconnectionState.RECONNECTING:
                    self.state = ReconnectionState.DISCONNECTED
                    self.stats['failed_reconnections'] += 1
                    
                    await self.error_handler.handle_error(
                        ErrorCode.RECONNECTION_FAILED,
                        f"Reconnection attempt {self.backoff_strategy.retry_count} failed for client {client_id}",
                        ErrorSeverity.MEDIUM,
                        category=ErrorCategory.CONNECTION,
                        context=ErrorContext(client_id=client_id, metadata={'attempt': self.backoff_strategy.retry_count})
                    )
            
            except Exception as e:
                self.logger.error(f"Error in reconnection loop: {e}")
                self.state = ReconnectionState.DISCONNECTED
                self.stats['failed_reconnections'] += 1
                
                await self.error_handler.handle_error(
                    ErrorCode.RECONNECTION_ERROR,
                    f"Reconnection error for client {client_id}: {str(e)}",
                    ErrorSeverity.HIGH,
                    category=ErrorCategory.CONNECTION,
                    context=ErrorContext(client_id=client_id, metadata={'error': str(e)})
                )
        
        # If we've exhausted retries, mark as failed
        if not self.backoff_strategy.should_retry() and self.state != ReconnectionState.CONNECTED:
            old_state = self.state
            self.state = ReconnectionState.FAILED
            await self._notify_state_change(old_state)
            
            await self.error_handler.handle_error(
                ErrorCode.RECONNECTION_MAX_RETRIES,
                f"Reconnection failed permanently for client {client_id} after {self.backoff_strategy.retry_count} attempts",
                ErrorSeverity.HIGH,
                category=ErrorCategory.CONNECTION,
                context=ErrorContext(client_id=client_id, metadata={'attempts': self.backoff_strategy.retry_count})
            )
            
            self.logger.error(f"Reconnection failed permanently for client {client_id}")
    
    async def _notify_state_change(self, old_state: ReconnectionState) -> None:
        """Notify all callbacks of state change."""
        if old_state != self.state:
            for callback in self.reconnection_callbacks:
                try:
                    callback(self.state)
                except Exception as e:
                    self.logger.error(f"Error in reconnection callback: {e}")
    
    async def _cleanup_expired_sessions(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                expired_sessions = []
                for client_id, session in self.sessions.items():
                    if session.is_expired(self.config.session_timeout):
                        expired_sessions.append(client_id)
                
                for client_id in expired_sessions:
                    del self.sessions[client_id]
                    self.logger.info(f"Cleaned up expired session for client {client_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reconnection statistics."""
        return {
            **self.stats,
            'current_state': self.state.value,
            'active_sessions': len(self.sessions),
            'current_retry_count': self.backoff_strategy.retry_count,
            'network_condition': self.network_detector.get_condition().value if self.network_detector else 'unknown'
        }
    
    def reset_reconnection(self) -> None:
        """Manually reset reconnection state (for testing or manual intervention)."""
        self.backoff_strategy.reset_on_success()
        self.state = ReconnectionState.DISCONNECTED
        if self._reconnection_task:
            self._reconnection_task.cancel()
        self.logger.info("Reconnection state manually reset")
    
    def generate_reconnection_id(self, client_id: str) -> str:
        """Generate a unique reconnection ID for a client."""
        timestamp = int(time.time() * 1000)  # milliseconds
        return f"reconnect_{client_id}_{timestamp}_{uuid.uuid4().hex[:8]}"


# Global instance for easy access
_global_reconnection_manager: Optional[ReconnectionManager] = None


def get_reconnection_manager() -> ReconnectionManager:
    """Get the global reconnection manager instance."""
    global _global_reconnection_manager
    if _global_reconnection_manager is None:
        _global_reconnection_manager = ReconnectionManager()
    return _global_reconnection_manager


async def initialize_reconnection_manager(config: Optional[ReconnectionConfig] = None) -> ReconnectionManager:
    """Initialize and start the global reconnection manager."""
    global _global_reconnection_manager
    _global_reconnection_manager = ReconnectionManager(config)
    await _global_reconnection_manager.start()
    return _global_reconnection_manager


async def shutdown_reconnection_manager() -> None:
    """Shutdown the global reconnection manager."""
    global _global_reconnection_manager
    if _global_reconnection_manager:
        await _global_reconnection_manager.stop()
        _global_reconnection_manager = None