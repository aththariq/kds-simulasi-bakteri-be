"""
Tests for WebSocket Reconnection Service.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from services.reconnection_service import (
    ReconnectionState, NetworkCondition, ReconnectionConfig, SessionState,
    ExponentialBackoffStrategy, NetworkConditionDetector, ReconnectionManager,
    get_reconnection_manager, initialize_reconnection_manager, shutdown_reconnection_manager
)
from services.websocket_error_handler import WebSocketErrorHandler


class TestReconnectionConfig:
    """Test reconnection configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ReconnectionConfig()
        
        assert config.initial_delay == 1.0
        assert config.max_delay == 300.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter_factor == 0.3
        assert config.max_retries == 10
        assert config.reset_after_success == 30
        assert config.enable_network_detection is True
        assert config.poor_network_multiplier == 2.0
        assert config.unstable_network_multiplier == 3.0
        assert config.enable_session_persistence is True
        assert config.message_queue_limit == 1000
        assert config.session_timeout == timedelta(hours=1)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReconnectionConfig(
            initial_delay=0.5,
            max_delay=60.0,
            max_retries=5,
            enable_session_persistence=False
        )
        
        assert config.initial_delay == 0.5
        assert config.max_delay == 60.0
        assert config.max_retries == 5
        assert config.enable_session_persistence is False


class TestSessionState:
    """Test session state management."""
    
    def test_session_state_creation(self):
        """Test session state creation."""
        session = SessionState(client_id="test_client")
        
        assert session.client_id == "test_client"
        assert session.auth_token is None
        assert len(session.subscriptions) == 0
        assert session.last_message_id is None
        assert len(session.queued_messages) == 0
        assert isinstance(session.created_at, datetime)
    
    def test_session_state_with_data(self):
        """Test session state with data."""
        subscriptions = {"sim1", "sim2"}
        messages = [{"type": "test", "data": "value"}]
        
        session = SessionState(
            client_id="test_client",
            auth_token="token123",
            subscriptions=subscriptions,
            last_message_id="msg_456",
            queued_messages=messages
        )
        
        assert session.client_id == "test_client"
        assert session.auth_token == "token123"
        assert session.subscriptions == subscriptions
        assert session.last_message_id == "msg_456"
        assert session.queued_messages == messages
    
    def test_session_expiration(self):
        """Test session expiration logic."""
        # Recent session should not be expired
        session = SessionState(client_id="test_client")
        assert not session.is_expired(timedelta(hours=1))
        
        # Old session should be expired
        old_session = SessionState(client_id="test_client")
        old_session.created_at = datetime.now() - timedelta(hours=2)
        assert old_session.is_expired(timedelta(hours=1))
    
    def test_session_serialization(self):
        """Test session serialization and deserialization."""
        original_session = SessionState(
            client_id="test_client",
            auth_token="token123",
            subscriptions={"sim1", "sim2"},
            last_message_id="msg_456",
            queued_messages=[{"type": "test"}]
        )
        
        # Serialize to dict
        session_dict = original_session.to_dict()
        assert session_dict['client_id'] == "test_client"
        assert session_dict['auth_token'] == "token123"
        assert set(session_dict['subscriptions']) == {"sim1", "sim2"}
        assert session_dict['last_message_id'] == "msg_456"
        assert session_dict['queued_messages'] == [{"type": "test"}]
        assert 'created_at' in session_dict
        
        # Deserialize from dict
        restored_session = SessionState.from_dict(session_dict)
        assert restored_session.client_id == original_session.client_id
        assert restored_session.auth_token == original_session.auth_token
        assert restored_session.subscriptions == original_session.subscriptions
        assert restored_session.last_message_id == original_session.last_message_id
        assert restored_session.queued_messages == original_session.queued_messages


class TestExponentialBackoffStrategy:
    """Test exponential backoff strategy."""
    
    def test_strategy_creation(self):
        """Test backoff strategy creation."""
        config = ReconnectionConfig()
        strategy = ExponentialBackoffStrategy(config)
        
        assert strategy.config == config
        assert strategy.retry_count == 0
        assert strategy.last_success_time > 0
    
    def test_first_retry_immediate(self):
        """Test first retry is immediate."""
        config = ReconnectionConfig()
        strategy = ExponentialBackoffStrategy(config)
        
        delay = strategy.get_delay()
        assert delay == 0  # First retry should be immediate
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = ReconnectionConfig(initial_delay=1.0, backoff_multiplier=2.0)
        strategy = ExponentialBackoffStrategy(config)
        
        # Simulate retries
        strategy.increment_retry()  # Retry 1
        delay1 = strategy.get_delay(NetworkCondition.GOOD)
        
        strategy.increment_retry()  # Retry 2
        delay2 = strategy.get_delay(NetworkCondition.GOOD)
        
        strategy.increment_retry()  # Retry 3
        delay3 = strategy.get_delay(NetworkCondition.GOOD)
        
        # Should follow exponential pattern (with jitter tolerance)
        assert 0.7 <= delay1 <= 1.3  # ~1.0 with jitter
        assert 1.4 <= delay2 <= 2.6  # ~2.0 with jitter
        assert 2.8 <= delay3 <= 5.2  # ~4.0 with jitter
    
    def test_max_delay_limit(self):
        """Test maximum delay limit."""
        config = ReconnectionConfig(initial_delay=1.0, max_delay=10.0, backoff_multiplier=2.0)
        strategy = ExponentialBackoffStrategy(config)
        
        # Simulate many retries to reach max delay
        for _ in range(10):
            strategy.increment_retry()
        
        delay = strategy.get_delay(NetworkCondition.GOOD)
        assert delay <= config.max_delay * 1.3  # Account for jitter
    
    def test_network_condition_multipliers(self):
        """Test network condition affects delay."""
        config = ReconnectionConfig(
            initial_delay=2.0,
            poor_network_multiplier=2.0,
            unstable_network_multiplier=3.0
        )
        strategy = ExponentialBackoffStrategy(config)
        strategy.increment_retry()
        
        good_delay = strategy.get_delay(NetworkCondition.GOOD)
        poor_delay = strategy.get_delay(NetworkCondition.POOR)
        unstable_delay = strategy.get_delay(NetworkCondition.UNSTABLE)
        
        # Poor should be roughly 2x good (within jitter tolerance)
        assert poor_delay > good_delay * 1.5
        assert poor_delay < good_delay * 2.5
        
        # Unstable should be roughly 3x good (within jitter tolerance)
        assert unstable_delay > good_delay * 2.5
        assert unstable_delay < good_delay * 3.5
    
    def test_retry_limiting(self):
        """Test retry count limiting."""
        config = ReconnectionConfig(max_retries=3)
        strategy = ExponentialBackoffStrategy(config)
        
        # Should allow retries up to max
        assert strategy.should_retry()
        
        strategy.increment_retry()
        assert strategy.should_retry()
        
        strategy.increment_retry()
        assert strategy.should_retry()
        
        strategy.increment_retry()
        assert not strategy.should_retry()  # Exceeded max retries
    
    def test_reset_on_success(self):
        """Test reset after successful connection."""
        config = ReconnectionConfig()
        strategy = ExponentialBackoffStrategy(config)
        
        # Simulate failed attempts
        strategy.increment_retry()
        strategy.increment_retry()
        assert strategy.retry_count == 2
        
        # Reset on success
        strategy.reset_on_success()
        assert strategy.retry_count == 0
        assert strategy.should_retry()


class TestNetworkConditionDetector:
    """Test network condition detection."""
    
    def test_detector_creation(self):
        """Test detector creation."""
        detector = NetworkConditionDetector()
        
        assert len(detector.latency_samples) == 0
        assert detector.disconnect_count == 0
        assert detector.last_disconnect_time == 0
    
    def test_latency_recording(self):
        """Test latency recording and condition detection."""
        detector = NetworkConditionDetector()
        
        # Record excellent latency
        for _ in range(5):
            detector.record_latency(30.0)  # 30ms
        
        assert detector.get_condition() == NetworkCondition.EXCELLENT
        
        # Record good latency
        detector.latency_samples.clear()
        for _ in range(5):
            detector.record_latency(100.0)  # 100ms
        
        assert detector.get_condition() == NetworkCondition.GOOD
        
        # Record poor latency
        detector.latency_samples.clear()
        for _ in range(5):
            detector.record_latency(500.0)  # 500ms
        
        assert detector.get_condition() == NetworkCondition.POOR
    
    def test_sample_limit(self):
        """Test latency sample limit."""
        detector = NetworkConditionDetector()
        
        # Record more than 10 samples
        for i in range(15):
            detector.record_latency(50.0)
        
        # Should keep only 10 samples
        assert len(detector.latency_samples) == 10
    
    def test_disconnect_tracking(self):
        """Test disconnect tracking for unstable network detection."""
        detector = NetworkConditionDetector()
        
        # Record multiple disconnects within 5 minutes
        current_time = time.time()
        with patch('time.time', return_value=current_time):
            detector.record_disconnect()
            detector.record_disconnect()
            detector.record_disconnect()
        
        # Should detect unstable network
        assert detector.get_condition() == NetworkCondition.UNSTABLE
    
    def test_disconnect_reset_after_time(self):
        """Test disconnect count reset after time."""
        detector = NetworkConditionDetector()
        
        # Record disconnect
        current_time = time.time()
        with patch('time.time', return_value=current_time):
            detector.record_disconnect()
            assert detector.disconnect_count == 1
        
        # Record another disconnect after 6 minutes
        with patch('time.time', return_value=current_time + 360):
            detector.record_disconnect()
            assert detector.disconnect_count == 1  # Should reset


class TestReconnectionManager:
    """Test reconnection manager."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return ReconnectionConfig(
            initial_delay=0.1,
            max_delay=1.0,
            max_retries=3,
            reset_after_success=5
        )
    
    @pytest.fixture
    def error_handler(self):
        """Mock error handler."""
        return AsyncMock(spec=WebSocketErrorHandler)
    
    @pytest.fixture
    def manager(self, config, error_handler):
        """Test reconnection manager."""
        return ReconnectionManager(config, error_handler)
    
    def test_manager_creation(self, manager):
        """Test manager creation."""
        assert manager.state == ReconnectionState.DISCONNECTED
        assert isinstance(manager.backoff_strategy, ExponentialBackoffStrategy)
        assert isinstance(manager.network_detector, NetworkConditionDetector)
        assert len(manager.sessions) == 0
        assert len(manager.reconnection_callbacks) == 0
        assert manager.message_replay_callback is None
    
    @pytest.mark.asyncio
    async def test_manager_lifecycle(self, manager):
        """Test manager start/stop lifecycle."""
        # Start manager
        await manager.start()
        assert manager._cleanup_task is not None
        assert not manager._cleanup_task.done()
        
        # Stop manager
        await manager.stop()
        assert manager._cleanup_task.done()
    
    def test_callback_registration(self, manager):
        """Test callback registration."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        replay_callback = MagicMock()
        
        manager.register_reconnection_callback(callback1)
        manager.register_reconnection_callback(callback2)
        manager.register_message_replay_callback(replay_callback)
        
        assert callback1 in manager.reconnection_callbacks
        assert callback2 in manager.reconnection_callbacks
        assert manager.message_replay_callback == replay_callback
    
    def test_latency_recording(self, manager):
        """Test latency recording."""
        manager.record_latency(50.0)
        manager.record_latency(75.0)
        
        # Should be recorded in network detector
        assert len(manager.network_detector.latency_samples) == 2
        assert 50.0 in manager.network_detector.latency_samples
        assert 75.0 in manager.network_detector.latency_samples
    
    def test_session_management(self, manager):
        """Test session save and retrieval."""
        client_id = "test_client"
        auth_token = "token123"
        subscriptions = {"sim1", "sim2"}
        
        # Save session
        manager.save_session(client_id, auth_token, subscriptions)
        
        assert client_id in manager.sessions
        session = manager.sessions[client_id]
        assert session.auth_token == auth_token
        assert session.subscriptions == subscriptions
    
    def test_message_queuing(self, manager):
        """Test message queuing for replay."""
        client_id = "test_client"
        message1 = {"type": "test1", "data": "value1"}
        message2 = {"type": "test2", "data": "value2"}
        
        # Queue messages
        result1 = manager.queue_message(client_id, message1)
        result2 = manager.queue_message(client_id, message2)
        
        assert result1 is True
        assert result2 is True
        
        # Check queued messages
        assert client_id in manager.sessions
        session = manager.sessions[client_id]
        assert len(session.queued_messages) == 2
        assert message1 in session.queued_messages
        assert message2 in session.queued_messages
        
        # Check statistics
        assert manager.stats['messages_queued'] == 2
    
    def test_message_queue_limit(self, manager):
        """Test message queue size limit."""
        client_id = "test_client"
        manager.config.message_queue_limit = 3
        
        # Queue more messages than limit
        for i in range(5):
            message = {"type": f"test{i}", "data": f"value{i}"}
            manager.queue_message(client_id, message)
        
        # Should keep only the last 3 messages
        session = manager.sessions[client_id]
        assert len(session.queued_messages) == 3
        assert session.queued_messages[0]["type"] == "test2"  # Oldest kept
        assert session.queued_messages[-1]["type"] == "test4"  # Newest
    
    @pytest.mark.asyncio
    async def test_disconnect_handling(self, manager):
        """Test disconnect handling."""
        client_id = "test_client"
        reason = "Connection timeout"
        
        # Mock state change notification
        callback = MagicMock()
        manager.register_reconnection_callback(callback)
        
        # Handle disconnect
        await manager.handle_disconnect(client_id, reason)
        
        # Should update state
        assert manager.state == ReconnectionState.DISCONNECTED
        
        # Should record disconnect
        assert manager.network_detector.disconnect_count > 0
        
        # Should call error handler
        manager.error_handler.handle_error.assert_called_once()
        
        # Should notify callbacks
        callback.assert_called_once_with(ReconnectionState.DISCONNECTED)
        
        # Should start reconnection task
        assert manager._reconnection_task is not None
    
    @pytest.mark.asyncio
    async def test_successful_connection_handling(self, manager):
        """Test successful connection handling."""
        client_id = "test_client"
        
        # Setup session with queued messages
        message = {"type": "test", "data": "value"}
        manager.queue_message(client_id, message)
        
        # Setup message replay callback
        replay_callback = MagicMock()
        manager.register_message_replay_callback(replay_callback)
        
        # Setup state change callback
        state_callback = MagicMock()
        manager.register_reconnection_callback(state_callback)
        
        # Handle successful connection
        session = await manager.handle_successful_connection(client_id)
        
        # Should update state
        assert manager.state == ReconnectionState.CONNECTED
        
        # Should reset backoff strategy
        assert manager.backoff_strategy.retry_count == 0
        
        # Should update statistics
        assert manager.stats['successful_reconnections'] == 1
        assert manager.stats['sessions_recovered'] == 1
        
        # Should notify callbacks
        state_callback.assert_called_once_with(ReconnectionState.CONNECTED)
        
        # Should replay messages
        replay_callback.assert_called_once_with(client_id, [message])
        
        # Should return session
        assert session is not None
        assert session.client_id == client_id
        
        # Queued messages should be cleared
        assert len(session.queued_messages) == 0
    
    @pytest.mark.asyncio
    async def test_expired_session_handling(self, manager):
        """Test handling of expired sessions."""
        client_id = "test_client"
        
        # Create session and make it expired
        manager.save_session(client_id)
        session = manager.sessions[client_id]
        session.created_at = datetime.now() - timedelta(hours=2)
        
        # Handle successful connection
        result_session = await manager.handle_successful_connection(client_id)
        
        # Expired session should be removed
        assert client_id not in manager.sessions
        assert result_session is None
    
    def test_statistics(self, manager):
        """Test statistics collection."""
        # Initial stats
        stats = manager.get_stats()
        assert stats['total_reconnections'] == 0
        assert stats['successful_reconnections'] == 0
        assert stats['failed_reconnections'] == 0
        assert stats['messages_queued'] == 0
        assert stats['sessions_recovered'] == 0
        assert stats['current_state'] == ReconnectionState.DISCONNECTED.value
        assert stats['active_sessions'] == 0
        assert stats['current_retry_count'] == 0
        
        # Add some activity
        manager.stats['total_reconnections'] = 5
        manager.stats['successful_reconnections'] = 3
        manager.stats['failed_reconnections'] = 2
        manager.queue_message("client1", {"test": "message"})
        manager.backoff_strategy.increment_retry()
        
        # Updated stats
        stats = manager.get_stats()
        assert stats['total_reconnections'] == 5
        assert stats['successful_reconnections'] == 3
        assert stats['failed_reconnections'] == 2
        assert stats['messages_queued'] == 1
        assert stats['active_sessions'] == 1
        assert stats['current_retry_count'] == 1
    
    def test_manual_reset(self, manager):
        """Test manual reconnection reset."""
        # Simulate some retry attempts
        manager.backoff_strategy.increment_retry()
        manager.backoff_strategy.increment_retry()
        manager.state = ReconnectionState.FAILED
        
        # Reset
        manager.reset_reconnection()
        
        # Should reset state
        assert manager.state == ReconnectionState.DISCONNECTED
        assert manager.backoff_strategy.retry_count == 0
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, manager):
        """Test session cleanup of expired sessions."""
        # Create sessions, some expired
        manager.save_session("client1")  # Fresh
        manager.save_session("client2")  # Will be expired
        manager.save_session("client3")  # Will be expired
        
        # Make some sessions expired
        manager.sessions["client2"].created_at = datetime.now() - timedelta(hours=2)
        manager.sessions["client3"].created_at = datetime.now() - timedelta(hours=3)
        
        # Manually trigger cleanup
        expired_sessions = []
        for client_id, session in manager.sessions.items():
            if session.is_expired(manager.config.session_timeout):
                expired_sessions.append(client_id)
        
        for client_id in expired_sessions:
            del manager.sessions[client_id]
        
        # Should keep only fresh session
        assert len(manager.sessions) == 1
        assert "client1" in manager.sessions
        assert "client2" not in manager.sessions
        assert "client3" not in manager.sessions


class TestGlobalReconnectionManager:
    """Test global reconnection manager functions."""
    
    @pytest.mark.asyncio
    async def test_global_manager_lifecycle(self):
        """Test global manager initialization and shutdown."""
        # Ensure clean state
        await shutdown_reconnection_manager()
        
        # Initialize global manager
        config = ReconnectionConfig(max_retries=5)
        manager = await initialize_reconnection_manager(config)
        
        assert manager is not None
        assert manager.config.max_retries == 5
        assert manager._cleanup_task is not None
        
        # Get same instance
        same_manager = get_reconnection_manager()
        assert same_manager is manager
        
        # Shutdown
        await shutdown_reconnection_manager()
        
        # Should create new instance after shutdown
        new_manager = get_reconnection_manager()
        assert new_manager is not manager
    
    def test_get_manager_lazy_initialization(self):
        """Test lazy initialization of global manager."""
        # Clear global state
        import services.reconnection_service
        services.reconnection_service._global_reconnection_manager = None
        
        # Should create new instance
        manager1 = get_reconnection_manager()
        assert manager1 is not None
        
        # Should return same instance
        manager2 = get_reconnection_manager()
        assert manager2 is manager1 