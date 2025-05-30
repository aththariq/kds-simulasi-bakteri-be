#!/usr/bin/env python3

"""
Simplified test for WebSocket Error Handling System (Task 8.4)
Core functionality test without FastAPI dependencies.
"""

import asyncio
import tempfile
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

# Import core error handling without fastapi dependencies
from services.websocket_error_handler import (
    ErrorCode, ErrorSeverity, ErrorCategory, ErrorContext,
    WebSocketError, ErrorHandler
)


async def test_error_code_system():
    """Test error code and severity system."""
    print("🔍 Testing Error Code System...", flush=True)
    
    # Test error code enumeration
    assert len([e for e in ErrorCode]) >= 25, "Should have at least 25 error codes"
    assert ErrorCode.CONNECTION_FAILED.value == "WS_1001", "Connection failed code should be WS_1001"
    print("   ✅ Error code enumeration working", flush=True)
    
    # Test severity levels
    assert len([s for s in ErrorSeverity]) == 5, "Should have 5 severity levels"
    print("   ✅ Severity levels working", flush=True)
    
    # Test categories
    assert len([c for c in ErrorCategory]) >= 10, "Should have at least 10 categories"
    print("   ✅ Error categories working", flush=True)


async def test_error_object_creation():
    """Test WebSocketError object creation."""
    print("🔍 Testing Error Object Creation...")
    
    # Create error context
    context = ErrorContext(
        client_id="test_client_123",
        simulation_id="sim_456",
        metadata={"test": "data"}
    )
    
    # Create error object
    error = WebSocketError(
        code=ErrorCode.CONNECTION_FAILED,
        message="Test connection failure",
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.CONNECTION,
        context=context,
        recovery_suggestions=["Retry", "Check network"]
    )
    
    # Test properties
    assert error.code == ErrorCode.CONNECTION_FAILED, "Error code should be set"
    assert error.severity == ErrorSeverity.HIGH, "Severity should be set"
    assert error.context.client_id == "test_client_123", "Context should be preserved"
    assert len(error.recovery_suggestions) == 2, "Recovery suggestions should be set"
    print("   ✅ Error object creation working")
    
    # Test serialization
    error_dict = error.to_dict()
    assert error_dict["code"] == "WS_1001", "Code should be serialized"
    assert error_dict["severity"] == "high", "Severity should be serialized"
    assert error_dict["context"]["client_id"] == "test_client_123", "Context should be serialized"
    print("   ✅ Error serialization working")


async def test_error_handler_basic():
    """Test basic error handler functionality."""
    print("🔍 Testing Error Handler Basic Functionality...")
    
    # Create handler
    handler = ErrorHandler(max_error_history=10)
    
    # Test initialization
    assert handler.max_error_history == 10, "Max history should be set"
    assert len(handler.error_history) == 0, "Error history should start empty"
    assert len(handler.recovery_handlers) >= 5, "Default recovery handlers should exist"
    print("   ✅ Error handler initialization working")
    
    # Create test context
    context = ErrorContext(client_id="test_client")
    
    # Handle an error
    error = await handler.handle_error(
        ErrorCode.CONNECTION_FAILED,
        "Test connection error",
        ErrorSeverity.HIGH,
        ErrorCategory.CONNECTION,
        context
    )
    
    # Verify error handling
    assert error.code == ErrorCode.CONNECTION_FAILED, "Error code should match"
    assert len(handler.error_history) == 1, "Error should be stored in history"
    assert handler.error_history[0] == error, "Stored error should match"
    print("   ✅ Basic error handling working")


async def test_error_statistics():
    """Test error statistics functionality."""
    print("🔍 Testing Error Statistics...")
    
    handler = ErrorHandler()
    
    # Generate test errors
    test_scenarios = [
        (ErrorCode.CONNECTION_FAILED, ErrorSeverity.HIGH),
        (ErrorCode.AUTH_FAILED, ErrorSeverity.MEDIUM),
        (ErrorCode.RATE_LIMIT_EXCEEDED, ErrorSeverity.LOW),
        (ErrorCode.CONNECTION_FAILED, ErrorSeverity.HIGH),  # Duplicate for counting
    ]
    
    for i, (code, severity) in enumerate(test_scenarios):
        await handler.handle_error(
            code,
            f"Test error {i}",
            severity,
            ErrorCategory.CONNECTION,
            ErrorContext(client_id=f"client_{i}")
        )
    
    # Get statistics
    stats = handler.get_error_statistics()
    
    assert stats["total_errors"] == 4, "Should have 4 total errors"
    assert "severity_distribution" in stats, "Should have severity distribution"
    assert "category_distribution" in stats, "Should have category distribution"
    
    # Check severity counts
    assert stats["severity_distribution"]["high"] == 2, "Should have 2 high severity"
    assert stats["severity_distribution"]["medium"] == 1, "Should have 1 medium severity"
    assert stats["severity_distribution"]["low"] == 1, "Should have 1 low severity"
    print("   ✅ Error statistics working")


async def test_recovery_suggestions():
    """Test recovery suggestion system."""
    print("🔍 Testing Recovery Suggestions...")
    
    handler = ErrorHandler()
    
    # Test different error types and their suggestions
    test_cases = [
        (ErrorCode.CONNECTION_FAILED, ["Check network connectivity", "Verify server is running"]),
        (ErrorCode.AUTH_FAILED, ["Verify authentication credentials", "Check if token is valid"]),
        (ErrorCode.RATE_LIMIT_EXCEEDED, ["Reduce message sending frequency", "Implement client-side rate limiting"]),
    ]
    
    for error_code, expected_suggestions in test_cases:
        error = await handler.handle_error(
            error_code,
            f"Test error {error_code.value}",
            ErrorSeverity.MEDIUM,
            ErrorCategory.CONNECTION,
            ErrorContext(client_id="test_client")
        )
        
        # Check that suggestions are provided
        assert len(error.recovery_suggestions) > 0, f"Should have suggestions for {error_code}"
        
        # Check that expected suggestions are included (partial match)
        suggestion_text = " ".join(error.recovery_suggestions).lower()
        for expected in expected_suggestions:
            assert any(word in suggestion_text for word in expected.lower().split()[:2]), \
                f"Should contain suggestion related to: {expected}"
    
    print("   ✅ Recovery suggestions working")


async def test_error_thresholds():
    """Test error threshold monitoring."""
    print("🔍 Testing Error Thresholds...")
    
    handler = ErrorHandler()
    
    # Generate multiple critical errors to trigger threshold
    for i in range(4):  # Exceed threshold of 3
        await handler.handle_error(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"Critical error {i}",
            ErrorSeverity.CRITICAL,
            ErrorCategory.SERVER,
            ErrorContext(client_id=f"client_{i}")
        )
    
    # Check for threshold escalation
    escalation_errors = [
        error for error in handler.error_history
        if "threshold exceeded" in error.message.lower()
    ]
    
    assert len(escalation_errors) >= 1, "Should generate escalation error when threshold exceeded"
    print("   ✅ Error threshold monitoring working")


async def test_custom_recovery_handler():
    """Test custom recovery handler registration."""
    print("🔍 Testing Custom Recovery Handler...")
    
    handler = ErrorHandler()
    custom_called = False
    
    async def custom_recovery(error, websocket):
        nonlocal custom_called
        custom_called = True
        error.recovery_suggestions.append("Custom recovery performed")
    
    # Register custom handler
    handler.register_recovery_handler(ErrorCode.AUTH_FAILED, custom_recovery)
    
    # Trigger error that should call custom handler
    error = await handler.handle_error(
        ErrorCode.AUTH_FAILED,
        "Authentication test error",
        ErrorSeverity.MEDIUM,
        ErrorCategory.AUTHENTICATION,
        ErrorContext(client_id="test_client")
    )
    
    assert custom_called, "Custom recovery handler should be called"
    assert "Custom recovery performed" in error.recovery_suggestions, "Custom suggestion should be added"
    print("   ✅ Custom recovery handler working")


async def test_error_history_cleanup():
    """Test error history size limit and cleanup."""
    print("🔍 Testing Error History Cleanup...")
    
    handler = ErrorHandler(max_error_history=5)
    
    # Generate more errors than the limit
    for i in range(10):
        await handler.handle_error(
            ErrorCode.CLIENT_ERROR,
            f"Test error {i}",
            ErrorSeverity.LOW,
            ErrorCategory.CLIENT,
            ErrorContext(client_id=f"client_{i}")
        )
    
    # Verify history size is limited
    assert len(handler.error_history) == 5, "Error history should be limited to max size"
    
    # Verify newest errors are kept
    assert handler.error_history[-1].message == "Test error 9", "Newest error should be kept"
    print("   ✅ Error history cleanup working")


async def main():
    """Run all simplified error handling tests."""
    print("🚀 Starting Simplified Error Handling Testing")
    print("Task 8.4: Error Handling Core Functionality Verification")
    print("=" * 70)
    
    try:
        await test_error_code_system()
        await test_error_object_creation()
        await test_error_handler_basic()
        await test_error_statistics()
        await test_recovery_suggestions()
        await test_error_thresholds()
        await test_custom_recovery_handler()
        await test_error_history_cleanup()
        
        print("\n" + "=" * 70)
        print("🎯 Error Handling Core Functionality Test Summary:")
        print("✅ Error code and severity system working")
        print("✅ Error object creation and serialization verified")
        print("✅ Error handler basic functionality operational")
        print("✅ Error statistics and monitoring working")
        print("✅ Recovery suggestion system functional")
        print("✅ Error threshold monitoring working")
        print("✅ Custom recovery handler registration working")
        print("✅ Error history cleanup and size limits working")
        print("\n🎉 ALL CORE ERROR HANDLING TESTS PASSED! 🎉")
        print("\n📝 Note: WebSocket integration tests require FastAPI environment")
        print("🔧 Core error handling system is production-ready")
        
    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 