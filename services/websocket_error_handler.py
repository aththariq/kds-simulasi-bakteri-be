"""
WebSocket Error Handling System for Task 8.4
Comprehensive error detection, recovery, and client feedback mechanisms for WebSocket communication.
"""

import asyncio
import json
import logging
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

# Conditional imports for WebSocket functionality
try:
    from fastapi import WebSocket, WebSocketDisconnect
    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback for testing without FastAPI
    WebSocket = None
    WebSocketDisconnect = Exception
    FASTAPI_AVAILABLE = False

try:
    from schemas.websocket_protocol import WebSocketProtocolMessage, MessageFactory, MessageType
    PROTOCOL_AVAILABLE = True
except ImportError:
    PROTOCOL_AVAILABLE = False


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    CRITICAL = "critical"      # Service-breaking errors requiring immediate attention
    HIGH = "high"             # Major functionality errors affecting user experience
    MEDIUM = "medium"         # Moderate errors with workarounds available
    LOW = "low"               # Minor errors that don't affect core functionality
    INFO = "info"             # Informational messages, not actual errors


class ErrorCategory(Enum):
    """Error categories for systematic classification."""
    CONNECTION = "connection"          # Connection establishment/maintenance issues
    AUTHENTICATION = "authentication" # Authentication and authorization errors
    PROTOCOL = "protocol"             # Message protocol violations
    VALIDATION = "validation"         # Data validation failures
    RESOURCE = "resource"             # Resource allocation/limit errors
    TIMEOUT = "timeout"               # Timeout-related errors
    NETWORK = "network"               # Network connectivity issues
    SERVER = "server"                 # Internal server errors
    CLIENT = "client"                 # Client-side error conditions
    RATE_LIMIT = "rate_limit"         # Rate limiting violations
    HEARTBEAT = "heartbeat"           # Heartbeat and keep-alive related issues


class ErrorCode(Enum):
    """Standardized error codes for WebSocket communication."""
    # Connection Errors (1000-1099)
    CONNECTION_FAILED = "WS_1001"
    CONNECTION_TIMEOUT = "WS_1002"
    CONNECTION_REJECTED = "WS_1003"
    CONNECTION_LOST = "WS_1004"
    HANDSHAKE_FAILED = "WS_1005"
    RECONNECTION_FAILED = "WS_1006"
    RECONNECTION_TIMEOUT = "WS_1007"
    RECONNECTION_MAX_RETRIES = "WS_1008"
    
    # Authentication Errors (1100-1199)
    AUTH_REQUIRED = "WS_1101"
    AUTH_FAILED = "WS_1102"
    AUTH_EXPIRED = "WS_1103"
    AUTH_INVALID_TOKEN = "WS_1104"
    AUTH_INSUFFICIENT_PERMISSIONS = "WS_1105"
    
    # Protocol Errors (1200-1299)
    INVALID_MESSAGE_FORMAT = "WS_1201"
    UNSUPPORTED_MESSAGE_TYPE = "WS_1202"
    PROTOCOL_VERSION_MISMATCH = "WS_1203"
    MESSAGE_TOO_LARGE = "WS_1204"
    INVALID_PAYLOAD = "WS_1205"
    
    # Validation Errors (1300-1399)
    VALIDATION_FAILED = "WS_1301"
    MISSING_REQUIRED_FIELD = "WS_1302"
    INVALID_FIELD_VALUE = "WS_1303"
    SIMULATION_NOT_FOUND = "WS_1304"
    CLIENT_ID_INVALID = "WS_1305"
    
    # Resource Errors (1400-1499)
    RESOURCE_EXHAUSTED = "WS_1401"
    QUEUE_FULL = "WS_1402"
    MEMORY_LIMIT_EXCEEDED = "WS_1403"
    TOO_MANY_CONNECTIONS = "WS_1404"
    SUBSCRIPTION_LIMIT_REACHED = "WS_1405"
    
    # Timeout Errors (1500-1599)
    REQUEST_TIMEOUT = "WS_1501"
    HEARTBEAT_TIMEOUT = "WS_1502"
    RESPONSE_TIMEOUT = "WS_1503"
    OPERATION_TIMEOUT = "WS_1504"
    
    # Network Errors (1600-1699)
    NETWORK_UNREACHABLE = "WS_1601"
    CONNECTION_RESET = "WS_1602"
    PACKET_LOSS = "WS_1603"
    BANDWIDTH_EXCEEDED = "WS_1604"
    
    # Server Errors (1700-1799)
    INTERNAL_SERVER_ERROR = "WS_1701"
    SERVICE_UNAVAILABLE = "WS_1702"
    DATABASE_ERROR = "WS_1703"
    EXTERNAL_SERVICE_ERROR = "WS_1704"
    
    # Rate Limiting Errors (1800-1899)
    RATE_LIMIT_EXCEEDED = "WS_1801"
    TOO_MANY_REQUESTS = "WS_1802"
    BURST_LIMIT_EXCEEDED = "WS_1803"
    
    # Client Errors (1900-1999)
    CLIENT_ERROR = "WS_1901"
    INVALID_STATE = "WS_1902"
    UNSUPPORTED_OPERATION = "WS_1903"


@dataclass
class ErrorContext:
    """Context information for error tracking."""
    client_id: Optional[str] = None
    simulation_id: Optional[str] = None
    message_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "client_id": self.client_id,
            "simulation_id": self.simulation_id,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "metadata": self.metadata
        }


@dataclass
class WebSocketError:
    """Comprehensive error information container."""
    code: ErrorCode
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Optional[ErrorContext] = None
    original_exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    retry_after: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for serialization."""
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "context": {
                "client_id": self.context.client_id if self.context else None,
                "simulation_id": self.context.simulation_id if self.context else None,
                "message_id": self.context.message_id if self.context else None,
                "metadata": self.context.metadata if self.context else {}
            },
            "recovery_suggestions": self.recovery_suggestions,
            "retry_after": self.retry_after.total_seconds() if self.retry_after else None
        }
    
    def to_websocket_message(self, client_id: str) -> Optional[WebSocketProtocolMessage]:
        """Convert error to WebSocket protocol message."""
        if not PROTOCOL_AVAILABLE:
            return None
        
        try:
            error_payload = MessageFactory.create_error(
                client_id=client_id,
                error_code=self.code.value,
                error_message=self.message,
                severity=self.severity.value,
                details={
                    "category": self.category.value,
                    "timestamp": self.timestamp.isoformat(),
                    "correlation_id": self.correlation_id,
                    "recovery_suggestions": self.recovery_suggestions,
                    "context": self.context.to_dict() if self.context else None
                }
            )
            return error_payload
            
        except Exception as e:
            # Fallback to basic dict structure if MessageFactory fails
            return {
                "type": "error",
                "client_id": client_id,
                "error_code": self.code.value,
                "message": self.message,
                "severity": self.severity.value,
                "timestamp": self.timestamp.isoformat()
            }


class ErrorHandler:
    """Centralized error handling system for WebSocket communication."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 error_log_file: Optional[str] = None,
                 max_error_history: int = 1000):
        """
        Initialize error handler.
        
        Args:
            log_level: Logging level for error messages
            error_log_file: Optional file path for error logging
            max_error_history: Maximum number of errors to keep in memory
        """
        self.logger = self._setup_logger(log_level, error_log_file)
        self.error_history: List[WebSocketError] = []
        self.max_error_history = max_error_history
        self.error_counts: Dict[str, int] = {}
        self.recovery_handlers: Dict[ErrorCode, Callable] = {}
        self.error_callbacks: List[Callable[[WebSocketError], None]] = []
        
        # Error thresholds for automatic actions
        self.error_thresholds = {
            ErrorSeverity.CRITICAL: 3,   # Max 3 critical errors before escalation
            ErrorSeverity.HIGH: 10,      # Max 10 high severity errors
            ErrorSeverity.MEDIUM: 50,    # Max 50 medium severity errors
        }
        
        # Setup default recovery handlers
        self._setup_default_recovery_handlers()
    
    def _setup_logger(self, log_level: str, error_log_file: Optional[str]) -> logging.Logger:
        """Setup structured logging for error handling."""
        logger = logging.getLogger("websocket_error_handler")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if error_log_file:
            file_handler = logging.FileHandler(error_log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_default_recovery_handlers(self):
        """Setup default recovery handlers for common error scenarios."""
        self.recovery_handlers = {
            ErrorCode.CONNECTION_TIMEOUT: self._handle_connection_timeout,
            ErrorCode.AUTH_EXPIRED: self._handle_auth_expired,
            ErrorCode.RATE_LIMIT_EXCEEDED: self._handle_rate_limit,
            ErrorCode.QUEUE_FULL: self._handle_queue_full,
            ErrorCode.MEMORY_LIMIT_EXCEEDED: self._handle_memory_limit,
        }
    
    async def handle_error(self, 
                          error_code: ErrorCode,
                          message: str,
                          severity: ErrorSeverity,
                          category: ErrorCategory,
                          context: Optional[ErrorContext] = None,
                          original_exception: Optional[Exception] = None,
                          websocket: Optional[WebSocket] = None) -> WebSocketError:
        """
        Handle a WebSocket error with comprehensive logging and recovery.
        
        Args:
            error_code: Standardized error code
            message: Human-readable error message
            severity: Error severity level
            category: Error category for classification
            context: Additional context information
            original_exception: Original exception that caused the error
            websocket: WebSocket connection for client notification
            
        Returns:
            WebSocketError object containing all error information
        """
        # Create error object
        ws_error = WebSocketError(
            code=error_code,
            message=message,
            severity=severity,
            category=category,
            context=context,
            original_exception=original_exception,
            stack_trace=traceback.format_exc() if original_exception else None
        )
        
        # Add recovery suggestions based on error type
        ws_error.recovery_suggestions = self._get_recovery_suggestions(error_code)
        
        # Log the error
        await self._log_error(ws_error)
        
        # Store in error history
        self._store_error(ws_error)
        
        # Notify client if WebSocket is available
        if websocket and context and context.client_id:
            await self._notify_client(websocket, ws_error, context.client_id)
        
        # Execute recovery handler if available
        if error_code in self.recovery_handlers:
            try:
                await self.recovery_handlers[error_code](ws_error, websocket)
            except Exception as e:
                self.logger.error(f"Recovery handler failed for {error_code}: {e}")
        
        # Trigger error callbacks
        for callback in self.error_callbacks:
            try:
                await callback(ws_error)
            except Exception as e:
                self.logger.error(f"Error callback failed: {e}")
        
        # Check for error threshold violations
        await self._check_error_thresholds(ws_error)
        
        return ws_error
    
    def _get_recovery_suggestions(self, error_code: ErrorCode) -> List[str]:
        """Get recovery suggestions based on error code."""
        suggestions_map = {
            ErrorCode.CONNECTION_FAILED: [
                "Check network connectivity",
                "Verify server is running",
                "Try reconnecting with exponential backoff"
            ],
            ErrorCode.AUTH_FAILED: [
                "Verify authentication credentials",
                "Check if token is valid and not expired",
                "Re-authenticate with valid credentials"
            ],
            ErrorCode.INVALID_MESSAGE_FORMAT: [
                "Verify message follows protocol specification",
                "Check JSON formatting and required fields",
                "Validate message against schema"
            ],
            ErrorCode.RATE_LIMIT_EXCEEDED: [
                "Reduce message sending frequency",
                "Implement client-side rate limiting",
                "Wait for rate limit reset period"
            ],
            ErrorCode.SIMULATION_NOT_FOUND: [
                "Verify simulation ID is correct",
                "Check if simulation exists and is active",
                "Create simulation before subscribing"
            ],
            ErrorCode.MEMORY_LIMIT_EXCEEDED: [
                "Clear unnecessary subscriptions",
                "Reduce message queue size",
                "Contact administrator if persistent"
            ]
        }
        
        return suggestions_map.get(error_code, ["Contact support if issue persists"])
    
    async def _log_error(self, error: WebSocketError):
        """Log error with appropriate level and structured information."""
        log_data = {
            "correlation_id": error.correlation_id,
            "error_code": error.code.value,
            "severity": error.severity.value,
            "category": error.category.value,
            "client_id": error.context.client_id if error.context else None,
            "simulation_id": error.context.simulation_id if error.context else None
        }
        
        # Use appropriate log level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error.message, extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(error.message, extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error.message, extra=log_data)
        else:
            self.logger.info(error.message, extra=log_data)
        
        # Log stack trace for exceptions
        if error.original_exception and error.stack_trace:
            self.logger.debug(f"Stack trace for {error.correlation_id}: {error.stack_trace}")
    
    def _store_error(self, error: WebSocketError):
        """Store error in history with automatic cleanup."""
        self.error_history.append(error)
        
        # Maintain error count statistics
        error_key = f"{error.code.value}_{error.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Cleanup old errors if limit exceeded
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    async def _notify_client(self, websocket: WebSocket, error: WebSocketError, client_id: str):
        """Send error notification to the client."""
        if websocket and FASTAPI_AVAILABLE:
            try:
                error_message_obj = error.to_websocket_message(client_id)
                
                if error_message_obj:
                    if isinstance(error_message_obj, dict): # Fallback dict from to_websocket_message
                        await websocket.send_json(error_message_obj)
                    elif hasattr(error_message_obj, 'to_json'): # WebSocketProtocolMessage
                        await websocket.send_text(error_message_obj.to_json())
                    else: # Should not happen if to_websocket_message is typed correctly
                        error_type_value = "ERROR"
                        try:
                            from schemas.websocket_protocol import MessageType as WSMessageType
                            error_type_value = WSMessageType.ERROR.value
                        except Exception:
                            logger.error("Could not import MessageType for generic error, using literal 'ERROR'")

                        error_data = {
                            "type": error_type_value,
                            "client_id": client_id,
                            "id": str(uuid.uuid4()),
                            "timestamp": datetime.now().isoformat(),
                            "protocol_version": "1.0", # Use literal
                            "priority": "high", 
                            "error": {
                                "error_code": error.code.value if error.code else "UNKNOWN_ERROR",
                                "error_message": error.message or "An unknown error occurred.",
                                "severity": error.severity.value if error.severity else "medium",
                                "details": {
                                    "category": error.category.value if error.category else "UNKNOWN",
                                    "correlation_id": error.correlation_id,
                                    "timestamp": error.timestamp.isoformat()
                                }
                            }
                        }
                        await websocket.send_json(error_data)
                        logger.warning(f"Sent generic JSON error to {client_id} due to unexpected error message object type: {type(error_message_obj)}")
                else: 
                    logger.warning(f"Protocol unavailable or MessageFactory failed, sending basic JSON error to {client_id} for code {error.code.value if error.code else 'UNKNOWN'}")
                    error_type_value = "ERROR"
                    try:
                        from schemas.websocket_protocol import MessageType as WSMessageType
                        error_type_value = WSMessageType.ERROR.value
                    except Exception: 
                        logger.error("Could not import MessageType for basic error, using literal 'ERROR'")

                    fallback_error = {
                        "type": error_type_value, 
                        "client_id": client_id,
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat(),
                        "protocol_version": "1.0", # Use literal
                        "priority": "high",
                        "error": {
                            "error_code": error.code.value if error.code else "UNKNOWN_ERROR_CODE",
                            "error_message": error.message or "Fallback error message",
                            "severity": error.severity.value if error.severity else "high",
                            "category": error.category.value if error.category else "SERVER",
                            "correlation_id": error.correlation_id,
                            "timestamp": error.timestamp.isoformat() 
                        }
                    }
                    await websocket.send_json(fallback_error)

            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected before error could be sent.")
            except Exception as e:
                logger.error(f"Failed to send error to client {client_id}: {e}", exc_info=True)
    
    async def _check_error_thresholds(self, error: WebSocketError):
        """Check if error thresholds are exceeded and take action."""
        current_count = sum(
            count for key, count in self.error_counts.items()
            if key.endswith(f"_{error.severity.value}")
        )
        
        threshold = self.error_thresholds.get(error.severity)
        if threshold and current_count >= threshold:
            escalation_error = WebSocketError(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"Error threshold exceeded for {error.severity.value} errors: {current_count}/{threshold}",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SERVER,
                context=error.context
            )
            
            # Log and store the escalation error
            await self._log_error(escalation_error)
            self._store_error(escalation_error)
            
            # Execute error callbacks for escalation error
            for callback in self.error_callbacks:
                try:
                    await callback(escalation_error)
                except Exception as e:
                    self.logger.error(f"Error callback failed for escalation: {e}")
            
            # Could trigger additional escalation actions here
    
    # Recovery handler implementations
    async def _handle_connection_timeout(self, error: WebSocketError, websocket: Optional[WebSocket]):
        """Handle connection timeout with retry logic."""
        if websocket and error.context:
            error.retry_after = timedelta(seconds=30)
            self.logger.info(f"Connection timeout for {error.context.client_id}, suggesting retry in 30s")
    
    async def _handle_auth_expired(self, error: WebSocketError, websocket: Optional[WebSocket]):
        """Handle expired authentication."""
        if error.context:
            self.logger.info(f"Authentication expired for {error.context.client_id}, requiring re-auth")
    
    async def _handle_rate_limit(self, error: WebSocketError, websocket: Optional[WebSocket]):
        """Handle rate limit violations."""
        error.retry_after = timedelta(minutes=1)
        self.logger.info(f"Rate limit exceeded, suggesting retry in 1 minute")
    
    async def _handle_queue_full(self, error: WebSocketError, websocket: Optional[WebSocket]):
        """Handle queue full scenarios."""
        self.logger.warning("Message queue full, considering queue cleanup")
    
    async def _handle_memory_limit(self, error: WebSocketError, websocket: Optional[WebSocket]):
        """Handle memory limit exceeded."""
        self.logger.critical("Memory limit exceeded, may need to reduce connections")
    
    # Public interface methods
    def register_recovery_handler(self, error_code: ErrorCode, handler: Callable):
        """Register custom recovery handler for specific error code."""
        self.recovery_handlers[error_code] = handler
    
    def add_error_callback(self, callback: Callable[[WebSocketError], None]):
        """Add callback to be notified of all errors."""
        self.error_callbacks.append(callback)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and metrics."""
        total_errors = len(self.error_history)
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        severity_counts = {}
        category_counts = {}
        
        for error in self.error_history:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
        
        return {
            "total_errors": total_errors,
            "recent_errors_1h": len(recent_errors),
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "error_codes": dict(self.error_counts),
            "last_error": self.error_history[-1].to_dict() if self.error_history else None
        }
    
    def clear_error_history(self):
        """Clear error history and reset counters."""
        self.error_history.clear()
        self.error_counts.clear()
    
    async def handle_websocket_error(self, client_id: str, exception: Exception) -> WebSocketError:
        """Handle WebSocket-specific errors with appropriate categorization."""
        # Determine error code and category based on exception type
        if isinstance(exception, WebSocketDisconnect) if FASTAPI_AVAILABLE else False:
            error_code = ErrorCode.CONNECTION_LOST
            category = ErrorCategory.CONNECTION
            severity = ErrorSeverity.MEDIUM
        elif "timeout" in str(exception).lower():
            error_code = ErrorCode.CONNECTION_TIMEOUT
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.HIGH
        elif "auth" in str(exception).lower():
            error_code = ErrorCode.AUTH_FAILED
            category = ErrorCategory.AUTHENTICATION
            severity = ErrorSeverity.MEDIUM
        elif "validation" in str(exception).lower() or "invalid" in str(exception).lower():
            error_code = ErrorCode.VALIDATION_FAILED
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.MEDIUM
        else:
            error_code = ErrorCode.INTERNAL_SERVER_ERROR
            category = ErrorCategory.SERVER
            severity = ErrorSeverity.HIGH
        
        context = ErrorContext(
            client_id=client_id,
            metadata={"exception_type": type(exception).__name__}
        )
        
        return await self.handle_error(
            error_code=error_code,
            message=f"WebSocket error for client {client_id}: {str(exception)}",
            severity=severity,
            category=category,
            context=context,
            original_exception=exception
        )