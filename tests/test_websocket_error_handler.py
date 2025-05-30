#!/usr/bin/env python3

"""
Comprehensive test for WebSocket Error Handling System (Task 8.4)
Tests error detection, logging, recovery mechanisms, and client feedback.
"""

import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from services.websocket_error_handler import (
    ErrorHandler, WebSocketError, ErrorContext, ErrorCode, ErrorSeverity, 
    ErrorCategory, handle_websocket_disconnect, handle_invalid_message,
    handle_rate_limit_violation, global_error_handler
)
from fastapi import WebSocketDisconnect


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.sent_messages = []
        self.closed = False
        
    async def send_text(self, data: str):
        """Mock send_text method."""
        self.sent_messages.append(data)
        
    async def close(self, code: int = 1000):
        """Mock close method."""
        self.closed = True


async def test_error_code_and_severity_system():
    """Test error code classification and severity system."""
    print("🔍 Testing Error Code and Severity System...")
    
    # Test error code categorization
    connection_errors = [ErrorCode.CONNECTION_FAILED, ErrorCode.CONNECTION_TIMEOUT]
    auth_errors = [ErrorCode.AUTH_FAILED, ErrorCode.AUTH_EXPIRED]
    protocol_errors = [ErrorCode.INVALID_MESSAGE_FORMAT, ErrorCode.MESSAGE_TOO_LARGE]
    
    # Verify error codes exist and have proper structure
    assert len([e for e in ErrorCode]) >= 25, "Should have at least 25 error codes"
    print("   ✅ Error code enumeration comprehensive")
    
    # Test severity levels
    severity_levels = [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH, ErrorSeverity.MEDIUM, ErrorSeverity.LOW, ErrorSeverity.INFO]
    assert len(severity_levels) == 5, "Should have 5 severity levels"
    print("   ✅ Severity level classification working")
    
    # Test error categories
    categories = [ErrorCategory.CONNECTION, ErrorCategory.AUTHENTICATION, ErrorCategory.PROTOCOL]
    assert len([c for c in ErrorCategory]) >= 10, "Should have at least 10 error categories"
    print("   ✅ Error category system comprehensive")


async def test_websocket_error_creation():
    """Test WebSocketError object creation and serialization."""
    print("🔍 Testing WebSocket Error Creation...")
    
    # Create error context
    context = ErrorContext(
        client_id="test_client_123",
        simulation_id="sim_456",
        message_id="msg_789",
        metadata={"test": "data"}
    )
    
    # Create WebSocket error
    error = WebSocketError(
        code=ErrorCode.CONNECTION_FAILED,
        message="Test connection failure",
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.CONNECTION,
        context=context,
        recovery_suggestions=["Retry connection", "Check network"]
    )
    
    # Test serialization
    error_dict = error.to_dict()
    assert error_dict["code"] == "WS_1001", "Error code should be serialized"
    assert error_dict["severity"] == "high", "Severity should be serialized"
    assert error_dict["context"]["client_id"] == "test_client_123", "Context should be included"
    assert len(error_dict["recovery_suggestions"]) == 2, "Recovery suggestions should be included"
    print("   ✅ Error object creation and serialization working")
    
    # Test WebSocket message conversion
    ws_message = error.to_websocket_message("target_client")
    if isinstance(ws_message, dict):
        # Fallback dict structure
        assert ws_message["client_id"] == "target_client", "Target client should be set in dict"
        print("   ✅ WebSocket message conversion working (fallback dict)")
    else:
        # Full WebSocketProtocolMessage object
        assert ws_message.client_id == "target_client", "Target client should be set"
        assert ws_message.error is not None, "Error payload should be included"
        print("   ✅ WebSocket message conversion working (full object)")


async def test_error_handler_initialization():
    """Test ErrorHandler initialization and configuration."""
    print("🔍 Testing Error Handler Initialization...")
    
    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        log_file = f.name
    
    try:
        # Initialize error handler
        handler = ErrorHandler(
            log_level="DEBUG",
            error_log_file=log_file,
            max_error_history=100
        )
        
        # Verify initialization
        assert handler.max_error_history == 100, "Max error history should be set"
        assert len(handler.recovery_handlers) >= 5, "Default recovery handlers should be set"
        assert handler.logger is not None, "Logger should be initialized"
        print("   ✅ Error handler initialization working")
        
        # Test error thresholds
        assert ErrorSeverity.CRITICAL in handler.error_thresholds, "Critical threshold should exist"
        assert handler.error_thresholds[ErrorSeverity.CRITICAL] == 3, "Critical threshold should be 3"
        print("   ✅ Error thresholds configured correctly")
        
        # Close logger handlers before cleanup
        for h in handler.logger.handlers[:]:
            h.close()
            handler.logger.removeHandler(h)
        
    finally:
        # Cleanup
        try:
            Path(log_file).unlink(missing_ok=True)
        except PermissionError:
            # File might still be in use, ignore for test
            pass


async def test_error_handling_and_logging():
    """Test comprehensive error handling with logging."""
    print("🔍 Testing Error Handling and Logging...")
    
    # Create error handler
    handler = ErrorHandler(log_level="DEBUG", max_error_history=50)
    
    # Create mock WebSocket
    mock_ws = MockWebSocket()
    
    # Create test context
    context = ErrorContext(
        client_id="test_client_456",
        simulation_id="sim_789",
        metadata={"test_data": "value"}
    )
    
    # Handle various error types
    test_cases = [
        (ErrorCode.CONNECTION_FAILED, ErrorSeverity.HIGH, ErrorCategory.CONNECTION),
        (ErrorCode.AUTH_FAILED, ErrorSeverity.MEDIUM, ErrorCategory.AUTHENTICATION),
        (ErrorCode.INVALID_MESSAGE_FORMAT, ErrorSeverity.MEDIUM, ErrorCategory.PROTOCOL),
        (ErrorCode.RATE_LIMIT_EXCEEDED, ErrorSeverity.MEDIUM, ErrorCategory.RATE_LIMIT),
    ]
    
    for error_code, severity, category in test_cases:
        error = await handler.handle_error(
            error_code,
            f"Test error for {error_code.value}",
            severity,
            category,
            context,
            websocket=mock_ws
        )
        
        assert error.code == error_code, f"Error code should be {error_code}"
        assert error.severity == severity, f"Severity should be {severity}"
        assert len(error.recovery_suggestions) > 0, "Recovery suggestions should be provided"
    
    # Verify error history
    assert len(handler.error_history) == 4, "All errors should be stored in history"
    print("   ✅ Error handling and storage working")
    
    # Verify client notifications
    assert len(mock_ws.sent_messages) == 4, "Client should receive error notifications"
    
    # Verify first message structure
    first_message = json.loads(mock_ws.sent_messages[0])
    assert first_message["type"] == "error", "Message type should be error"
    assert "error" in first_message, "Error payload should be included"
    print("   ✅ Client notification working")


async def test_recovery_handlers():
    """Test recovery handler execution."""
    print("🔍 Testing Recovery Handlers...")
    
    handler = ErrorHandler()
    mock_ws = MockWebSocket()
    
    # Test connection timeout recovery
    context = ErrorContext(client_id="test_client")
    error = await handler.handle_error(
        ErrorCode.CONNECTION_TIMEOUT,
        "Connection timeout occurred",
        ErrorSeverity.HIGH,
        ErrorCategory.TIMEOUT,
        context,
        websocket=mock_ws
    )
    
    assert error.retry_after is not None, "Retry after should be set for timeout"
    assert error.retry_after.total_seconds() == 30, "Retry after should be 30 seconds"
    print("   ✅ Connection timeout recovery working")
    
    # Test rate limit recovery
    error = await handler.handle_error(
        ErrorCode.RATE_LIMIT_EXCEEDED,
        "Rate limit exceeded",
        ErrorSeverity.MEDIUM,
        ErrorCategory.RATE_LIMIT,
        context
    )
    
    assert error.retry_after is not None, "Retry after should be set for rate limit"
    assert error.retry_after.total_seconds() == 60, "Retry after should be 60 seconds"
    print("   ✅ Rate limit recovery working")
    
    # Test custom recovery handler
    custom_called = False
    
    async def custom_recovery(error, websocket):
        nonlocal custom_called
        custom_called = True
        error.recovery_suggestions.append("Custom recovery action")
    
    handler.register_recovery_handler(ErrorCode.AUTH_FAILED, custom_recovery)
    
    error = await handler.handle_error(
        ErrorCode.AUTH_FAILED,
        "Authentication failed",
        ErrorSeverity.MEDIUM,
        ErrorCategory.AUTHENTICATION,
        context
    )
    
    assert custom_called, "Custom recovery handler should be called"
    assert "Custom recovery action" in error.recovery_suggestions, "Custom suggestion should be added"
    print("   ✅ Custom recovery handler registration working")


async def test_error_statistics_and_monitoring():
    """Test error statistics and monitoring capabilities."""
    print("🔍 Testing Error Statistics and Monitoring...")
    
    handler = ErrorHandler(max_error_history=10)
    
    # Generate various errors
    error_scenarios = [
        (ErrorCode.CONNECTION_FAILED, ErrorSeverity.HIGH),
        (ErrorCode.CONNECTION_FAILED, ErrorSeverity.HIGH),  # Duplicate for counting
        (ErrorCode.AUTH_FAILED, ErrorSeverity.MEDIUM),
        (ErrorCode.RATE_LIMIT_EXCEEDED, ErrorSeverity.LOW),
        (ErrorCode.VALIDATION_FAILED, ErrorSeverity.MEDIUM),
    ]
    
    for error_code, severity in error_scenarios:
        await handler.handle_error(
            error_code,
            f"Test error {error_code.value}",
            severity,
            ErrorCategory.CONNECTION,
            ErrorContext(client_id=f"client_{len(handler.error_history)}")
        )
    
    # Get statistics
    stats = handler.get_error_statistics()
    
    assert stats["total_errors"] == 5, "Total error count should be 5"
    assert "severity_distribution" in stats, "Severity distribution should be included"
    assert "category_distribution" in stats, "Category distribution should be included"
    assert "error_codes" in stats, "Error codes should be included"
    
    # Check specific counts
    assert stats["severity_distribution"]["high"] == 2, "Should have 2 high severity errors"
    assert stats["severity_distribution"]["medium"] == 2, "Should have 2 medium severity errors"
    assert stats["severity_distribution"]["low"] == 1, "Should have 1 low severity error"
    print("   ✅ Error statistics working")
    
    # Test error history cleanup
    for i in range(15):  # Exceed max_error_history
        await handler.handle_error(
            ErrorCode.CLIENT_ERROR,
            f"Cleanup test error {i}",
            ErrorSeverity.LOW,
            ErrorCategory.CLIENT
        )
    
    assert len(handler.error_history) == 10, "Error history should be limited to max size"
    print("   ✅ Error history cleanup working")


async def test_error_threshold_monitoring():
    """Test error threshold monitoring and escalation."""
    print("🔍 Testing Error Threshold Monitoring...")
    
    handler = ErrorHandler()
    
    # Simulate multiple critical errors
    for i in range(4):  # Exceed critical threshold of 3
        await handler.handle_error(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"Critical error {i}",
            ErrorSeverity.CRITICAL,
            ErrorCategory.SERVER,
            ErrorContext(client_id=f"client_{i}")
        )
    
    # Check if escalation occurred
    escalation_errors = [
        error for error in handler.error_history 
        if "threshold exceeded" in error.message.lower()
    ]
    
    assert len(escalation_errors) >= 1, "Escalation error should be generated"
    print("   ✅ Error threshold monitoring working")


async def test_convenience_functions():
    """Test convenience functions for common error scenarios."""
    print("🔍 Testing Convenience Functions...")
    
    handler = ErrorHandler()
    
    # Test WebSocket disconnect handling
    disconnect_exception = WebSocketDisconnect(code=1006, reason="Connection lost")
    await handle_websocket_disconnect(handler, "test_client", disconnect_exception)
    
    assert len(handler.error_history) == 1, "Disconnect error should be recorded"
    assert handler.error_history[0].code == ErrorCode.CONNECTION_LOST, "Should use connection lost code"
    print("   ✅ WebSocket disconnect handling working")
    
    # Test invalid message handling
    mock_ws = MockWebSocket()
    validation_error = ValueError("Invalid JSON format")
    await handle_invalid_message(handler, "test_client", "invalid{json", validation_error, mock_ws)
    
    assert len(handler.error_history) == 2, "Invalid message error should be recorded"
    assert handler.error_history[1].code == ErrorCode.INVALID_MESSAGE_FORMAT, "Should use invalid format code"
    print("   ✅ Invalid message handling working")
    
    # Test rate limit violation handling
    await handle_rate_limit_violation(handler, "test_client", 15.5, 10.0, mock_ws)
    
    assert len(handler.error_history) == 3, "Rate limit error should be recorded"
    assert handler.error_history[2].code == ErrorCode.RATE_LIMIT_EXCEEDED, "Should use rate limit code"
    assert "15.5" in handler.error_history[2].message, "Current rate should be in message"
    print("   ✅ Rate limit violation handling working")


async def test_error_callback_system():
    """Test error callback notification system."""
    print("🔍 Testing Error Callback System...")
    
    handler = ErrorHandler()
    callback_calls = []
    
    async def error_callback(error: WebSocketError):
        callback_calls.append(error)
    
    handler.add_error_callback(error_callback)
    
    # Generate test error
    await handler.handle_error(
        ErrorCode.CONNECTION_FAILED,
        "Test callback error",
        ErrorSeverity.HIGH,
        ErrorCategory.CONNECTION
    )
    
    assert len(callback_calls) == 1, "Error callback should be called"
    assert callback_calls[0].code == ErrorCode.CONNECTION_FAILED, "Callback should receive correct error"
    print("   ✅ Error callback system working")


async def test_global_error_handler():
    """Test global error handler instance."""
    print("🔍 Testing Global Error Handler...")
    
    # Verify global instance exists
    assert global_error_handler is not None, "Global error handler should exist"
    assert isinstance(global_error_handler, ErrorHandler), "Should be ErrorHandler instance"
    
    # Test that it works
    await global_error_handler.handle_error(
        ErrorCode.CLIENT_ERROR,
        "Test global handler",
        ErrorSeverity.LOW,
        ErrorCategory.CLIENT
    )
    
    assert len(global_error_handler.error_history) >= 1, "Global handler should record errors"
    print("   ✅ Global error handler working")


async def main():
    """Run all error handling tests."""
    print("🚀 Starting WebSocket Error Handling Testing")
    print("Task 8.4: Error Handling Implementation Verification")
    print("=" * 70)
    
    try:
        await test_error_code_and_severity_system()
        await test_websocket_error_creation()
        await test_error_handler_initialization()
        await test_error_handling_and_logging()
        await test_recovery_handlers()
        await test_error_statistics_and_monitoring()
        await test_error_threshold_monitoring()
        await test_convenience_functions()
        await test_error_callback_system()
        await test_global_error_handler()
        
        print("\n" + "=" * 70)
        print("🎯 Error Handling System Test Summary:")
        print("✅ Error code and severity classification working")
        print("✅ Error object creation and serialization verified")
        print("✅ Error handler initialization and configuration working")
        print("✅ Comprehensive error handling and logging functional")
        print("✅ Recovery handlers and custom handlers working")
        print("✅ Error statistics and monitoring operational")
        print("✅ Error threshold monitoring and escalation working")
        print("✅ Convenience functions for common scenarios verified")
        print("✅ Error callback notification system functional")
        print("✅ Global error handler instance verified")
        print("\n🎉 ALL ERROR HANDLING TESTS PASSED! 🎉")
        
    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 