"""
WebSocket Message Protocol Specification v1.0
Task 8.2: Message Protocol Design

This module defines the comprehensive WebSocket message protocol for the 
bacterial simulation system, including message types, payload formats, 
validation rules, and versioning strategy.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import uuid
from pydantic import BaseModel, Field, validator, ValidationError


# Protocol Version and Constants
PROTOCOL_VERSION = "1.0"
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
MAX_PAYLOAD_SIZE = 512 * 1024   # 512KB


class MessageCategory(Enum):
    """High-level message categories for protocol organization."""
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    SUBSCRIPTION = "subscription"
    SIMULATION = "simulation"
    DATA = "data"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SYSTEM = "system"


class MessageType(Enum):
    """Comprehensive WebSocket message types organized by category."""
    
    # Connection Lifecycle (CONNECTION category)
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_TERMINATED = "connection_terminated"
    CONNECTION_INFO = "connection_info"
    CONNECTION_STATS = "connection_stats"
    
    # Authentication (AUTHENTICATION category)
    AUTH_REQUEST = "auth_request"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"
    AUTH_REFRESH = "auth_refresh"
    AUTH_LOGOUT = "auth_logout"
    
    # Subscription Management (SUBSCRIPTION category)
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    UNSUBSCRIPTION_CONFIRMED = "unsubscription_confirmed"
    SUBSCRIPTION_LIST = "subscription_list"
    SUBSCRIPTION_ERROR = "subscription_error"
    
    # Simulation Control (SIMULATION category)
    SIMULATION_START = "simulation_start"
    SIMULATION_STOP = "simulation_stop"
    SIMULATION_PAUSE = "simulation_pause"
    SIMULATION_RESUME = "simulation_resume"
    SIMULATION_RESET = "simulation_reset"
    SIMULATION_CONFIG = "simulation_config"
    SIMULATION_STATUS = "simulation_status"
    
    # Data Updates (DATA category)
    SIMULATION_UPDATE = "simulation_update"
    PERFORMANCE_UPDATE = "performance_update"
    STATUS_UPDATE = "status_update"
    BATCH_UPDATE = "batch_update"
    SNAPSHOT_UPDATE = "snapshot_update"
    METRICS_UPDATE = "metrics_update"
    
    # Error Handling (ERROR category)
    ERROR = "error"
    WARNING = "warning"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    
    # Heartbeat (HEARTBEAT category)
    PING = "ping"
    PONG = "pong"
    
    # System Management (SYSTEM category)
    SYSTEM_STATUS = "system_status"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_MAINTENANCE = "system_maintenance"
    
    @classmethod
    def get_category(cls, message_type: 'MessageType') -> MessageCategory:
        """Get the category for a given message type."""
        category_mapping = {
            cls.CONNECTION_ESTABLISHED: MessageCategory.CONNECTION,
            cls.CONNECTION_TERMINATED: MessageCategory.CONNECTION,
            cls.CONNECTION_INFO: MessageCategory.CONNECTION,
            cls.CONNECTION_STATS: MessageCategory.CONNECTION,
            
            cls.AUTH_REQUEST: MessageCategory.AUTHENTICATION,
            cls.AUTH_SUCCESS: MessageCategory.AUTHENTICATION,
            cls.AUTH_FAILED: MessageCategory.AUTHENTICATION,
            cls.AUTH_REFRESH: MessageCategory.AUTHENTICATION,
            cls.AUTH_LOGOUT: MessageCategory.AUTHENTICATION,
            
            cls.SUBSCRIBE: MessageCategory.SUBSCRIPTION,
            cls.UNSUBSCRIBE: MessageCategory.SUBSCRIPTION,
            cls.SUBSCRIPTION_CONFIRMED: MessageCategory.SUBSCRIPTION,
            cls.UNSUBSCRIPTION_CONFIRMED: MessageCategory.SUBSCRIPTION,
            cls.SUBSCRIPTION_LIST: MessageCategory.SUBSCRIPTION,
            cls.SUBSCRIPTION_ERROR: MessageCategory.SUBSCRIPTION,
            
            cls.SIMULATION_START: MessageCategory.SIMULATION,
            cls.SIMULATION_STOP: MessageCategory.SIMULATION,
            cls.SIMULATION_PAUSE: MessageCategory.SIMULATION,
            cls.SIMULATION_RESUME: MessageCategory.SIMULATION,
            cls.SIMULATION_RESET: MessageCategory.SIMULATION,
            cls.SIMULATION_CONFIG: MessageCategory.SIMULATION,
            cls.SIMULATION_STATUS: MessageCategory.SIMULATION,
            
            cls.SIMULATION_UPDATE: MessageCategory.DATA,
            cls.PERFORMANCE_UPDATE: MessageCategory.DATA,
            cls.STATUS_UPDATE: MessageCategory.DATA,
            cls.BATCH_UPDATE: MessageCategory.DATA,
            cls.SNAPSHOT_UPDATE: MessageCategory.DATA,
            cls.METRICS_UPDATE: MessageCategory.DATA,
            
            cls.ERROR: MessageCategory.ERROR,
            cls.WARNING: MessageCategory.ERROR,
            cls.VALIDATION_ERROR: MessageCategory.ERROR,
            cls.RATE_LIMIT_ERROR: MessageCategory.ERROR,
            
            cls.PING: MessageCategory.HEARTBEAT,
            cls.PONG: MessageCategory.HEARTBEAT,
            
            cls.SYSTEM_STATUS: MessageCategory.SYSTEM,
            cls.SYSTEM_SHUTDOWN: MessageCategory.SYSTEM,
            cls.SYSTEM_MAINTENANCE: MessageCategory.SYSTEM,
        }
        return category_mapping.get(message_type, MessageCategory.SYSTEM)


class Priority(Enum):
    """Message priority levels for processing order."""
    CRITICAL = "critical"  # System errors, shutdowns
    HIGH = "high"         # Authentication, connection events
    NORMAL = "normal"     # Regular data updates
    LOW = "low"          # Heartbeats, status updates


class CompressionType(Enum):
    """Supported compression methods for large payloads."""
    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"


# Pydantic Models for Payload Validation

class AuthPayload(BaseModel):
    """Authentication request payload."""
    api_key: str = Field(..., min_length=10, max_length=128)
    client_info: Optional[Dict[str, Any]] = None
    refresh_token: Optional[str] = None


class SubscriptionPayload(BaseModel):
    """Subscription management payload."""
    simulation_id: str = Field(..., min_length=10, max_length=100)
    subscription_type: Literal["simulation", "performance", "all"] = "simulation"
    filters: Optional[Dict[str, Any]] = None


class SimulationControlPayload(BaseModel):
    """Simulation control payload."""
    simulation_id: str = Field(..., min_length=10, max_length=100)
    parameters: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class SimulationDataPayload(BaseModel):
    """Simulation data update payload."""
    simulation_id: str
    generation: int = Field(..., ge=0)
    timestamp: str
    population_data: Dict[str, Any]
    fitness_data: Dict[str, Any]
    mutation_data: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class PerformanceDataPayload(BaseModel):
    """Performance metrics payload."""
    timestamp: str
    cpu_usage: float = Field(..., ge=0, le=100)
    memory_usage: float = Field(..., ge=0)
    network_latency: Optional[float] = None
    active_connections: int = Field(..., ge=0)
    message_rate: Optional[float] = None


class ErrorPayload(BaseModel):
    """Error information payload."""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"


class BatchUpdatePayload(BaseModel):
    """Batch update payload for multiple data points."""
    updates: List[Dict[str, Any]] = Field(..., max_items=100)
    batch_id: str
    total_batches: int = Field(..., ge=1)
    batch_index: int = Field(..., ge=0)


# Core Protocol Message Structure

class WebSocketProtocolMessage(BaseModel):
    """
    Core WebSocket message structure following protocol v1.0 specification.
    
    This is the main message format for all WebSocket communication in the
    bacterial simulation system.
    """
    
    # Message Identification
    type: MessageType = Field(..., description="Message type identifier")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message ID")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="ISO 8601 timestamp")
    
    # Protocol Information
    protocol_version: str = Field(default=PROTOCOL_VERSION, description="Protocol version")
    priority: Priority = Field(default=Priority.NORMAL, description="Message priority")
    
    # Connection Context
    client_id: str = Field(..., description="Client connection identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    
    # Message Content
    simulation_id: Optional[str] = Field(None, description="Target simulation ID if applicable")
    data: Optional[Dict[str, Any]] = Field(None, description="Message payload data")
    
    # Error Information
    error: Optional[ErrorPayload] = Field(None, description="Error details if applicable")
    
    # Protocol Features
    compression: CompressionType = Field(default=CompressionType.NONE, description="Payload compression")
    encrypted: bool = Field(default=False, description="Whether payload is encrypted")
    
    # Message Metadata
    correlation_id: Optional[str] = Field(None, description="ID for request-response correlation")
    reply_to: Optional[str] = Field(None, description="Message ID this is replying to")
    expires_at: Optional[str] = Field(None, description="Message expiration timestamp")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format. Use ISO 8601.')
    
    @validator('client_id')
    def validate_client_id(cls, v):
        """Validate client ID format."""
        if not v or len(v) < 8 or len(v) > 128:
            raise ValueError('Client ID must be between 8 and 128 characters')
        return v
    
    @validator('simulation_id')
    def validate_simulation_id(cls, v):
        """Validate simulation ID format."""
        if v and (len(v) < 10 or len(v) > 100):
            raise ValueError('Simulation ID must be between 10 and 100 characters')
        return v
    
    def get_category(self) -> MessageCategory:
        """Get the message category."""
        return MessageType.get_category(self.type)
    
    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return self.json(exclude_none=True, by_alias=True)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketProtocolMessage':
        """Deserialize message from JSON string."""
        return cls.parse_raw(json_str)
    
    def validate_payload(self) -> bool:
        """Validate message payload based on message type."""
        if not self.data:
            return True  # No payload to validate
        
        try:
            if self.type in [MessageType.AUTH_REQUEST]:
                AuthPayload(**self.data)
            elif self.type in [MessageType.SUBSCRIBE, MessageType.UNSUBSCRIBE]:
                SubscriptionPayload(**self.data)
            elif self.type in [MessageType.SIMULATION_START, MessageType.SIMULATION_STOP, 
                             MessageType.SIMULATION_PAUSE, MessageType.SIMULATION_RESUME]:
                SimulationControlPayload(**self.data)
            elif self.type == MessageType.SIMULATION_UPDATE:
                SimulationDataPayload(**self.data)
            elif self.type == MessageType.PERFORMANCE_UPDATE:
                PerformanceDataPayload(**self.data)
            elif self.type == MessageType.BATCH_UPDATE:
                BatchUpdatePayload(**self.data)
            
            return True
        except ValidationError:
            return False
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if not self.expires_at:
            return False
        
        try:
            expiry = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
            return datetime.now(expiry.tzinfo) > expiry
        except ValueError:
            return False


# Protocol Versioning and Compatibility

class ProtocolVersionInfo(BaseModel):
    """Protocol version information."""
    version: str = PROTOCOL_VERSION
    supported_versions: List[str] = ["1.0"]
    deprecated_versions: List[str] = []
    min_client_version: str = "1.0"
    features: List[str] = [
        "authentication",
        "subscription_management", 
        "real_time_updates",
        "heartbeat",
        "error_handling",
        "compression",
        "batch_updates",
        "message_expiration",
        "correlation_ids"
    ]
    message_types: List[str] = [msg_type.value for msg_type in MessageType]


# Message Factory and Utilities

class MessageFactory:
    """Factory class for creating standardized WebSocket messages."""
    
    @staticmethod
    def create_connection_established(client_id: str, server_info: Dict[str, Any]) -> WebSocketProtocolMessage:
        """Create connection established message."""
        return WebSocketProtocolMessage(
            type=MessageType.CONNECTION_ESTABLISHED,
            client_id=client_id,
            priority=Priority.HIGH,
            data={
                "server_info": server_info,
                "protocol_version": PROTOCOL_VERSION,
                "capabilities": ProtocolVersionInfo().features
            }
        )
    
    @staticmethod
    def create_auth_request(client_id: str, api_key: str, client_info: Optional[Dict] = None) -> WebSocketProtocolMessage:
        """Create authentication request message."""
        return WebSocketProtocolMessage(
            type=MessageType.AUTH_REQUEST,
            client_id=client_id,
            priority=Priority.HIGH,
            data={
                "api_key": api_key,
                "client_info": client_info or {}
            }
        )
    
    @staticmethod
    def create_subscription(client_id: str, simulation_id: str, action: Literal["subscribe", "unsubscribe"]) -> WebSocketProtocolMessage:
        """Create subscription management message."""
        message_type = MessageType.SUBSCRIBE if action == "subscribe" else MessageType.UNSUBSCRIBE
        return WebSocketProtocolMessage(
            type=message_type,
            client_id=client_id,
            simulation_id=simulation_id,
            data={
                "simulation_id": simulation_id,
                "subscription_type": "simulation"
            }
        )
    
    @staticmethod
    def create_simulation_update(client_id: str, simulation_id: str, generation_data: Dict[str, Any]) -> WebSocketProtocolMessage:
        """Create simulation data update message."""
        return WebSocketProtocolMessage(
            type=MessageType.SIMULATION_UPDATE,
            client_id=client_id,
            simulation_id=simulation_id,
            data=generation_data
        )
    
    @staticmethod
    def create_error(client_id: str, error_code: str, error_message: str, 
                    severity: str = "medium", details: Optional[Dict] = None) -> WebSocketProtocolMessage:
        """Create error message."""
        return WebSocketProtocolMessage(
            type=MessageType.ERROR,
            client_id=client_id,
            priority=Priority.HIGH if severity in ["high", "critical"] else Priority.NORMAL,
            error=ErrorPayload(
                error_code=error_code,
                error_message=error_message,
                severity=severity,
                details=details or {},
                timestamp=datetime.now().isoformat()
            )
        )
    
    @staticmethod
    def create_heartbeat(client_id: str, heartbeat_type: Literal["ping", "pong"]) -> WebSocketProtocolMessage:
        """Create heartbeat message."""
        message_type = MessageType.PING if heartbeat_type == "ping" else MessageType.PONG
        return WebSocketProtocolMessage(
            type=message_type,
            client_id=client_id,
            priority=Priority.LOW
        )


# Protocol Validation and Utilities

class ProtocolValidator:
    """Validator for WebSocket protocol messages."""
    
    @staticmethod
    def validate_message_size(message: str) -> bool:
        """Validate message size limits."""
        return len(message.encode('utf-8')) <= MAX_MESSAGE_SIZE
    
    @staticmethod
    def validate_payload_size(payload: Dict[str, Any]) -> bool:
        """Validate payload size limits."""
        payload_size = len(json.dumps(payload).encode('utf-8'))
        return payload_size <= MAX_PAYLOAD_SIZE
    
    @staticmethod
    def validate_message_format(message_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate message format and return validation result."""
        try:
            WebSocketProtocolMessage(**message_data)
            return True, None
        except ValidationError as e:
            return False, str(e)
    
    @staticmethod
    def sanitize_client_id(client_id: str) -> str:
        """Sanitize client ID for security."""
        # Remove any potentially dangerous characters
        sanitized = ''.join(c for c in client_id if c.isalnum() or c in '-_')
        return sanitized[:128]  # Limit length


# Documentation and Schema Export

def get_protocol_documentation() -> Dict[str, Any]:
    """Get comprehensive protocol documentation."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "description": "WebSocket Message Protocol for Bacterial Simulation System",
        "message_categories": {cat.value: cat.name for cat in MessageCategory},
        "message_types": {msg.value: {
            "name": msg.name,
            "category": MessageType.get_category(msg).value,
            "description": f"Message type for {msg.value.replace('_', ' ')}"
        } for msg in MessageType},
        "priority_levels": {p.value: p.name for p in Priority},
        "compression_types": {c.value: c.name for c in CompressionType},
        "message_structure": WebSocketProtocolMessage.schema(),
        "payload_schemas": {
            "auth": AuthPayload.schema(),
            "subscription": SubscriptionPayload.schema(),
            "simulation_control": SimulationControlPayload.schema(),
            "simulation_data": SimulationDataPayload.schema(),
            "performance_data": PerformanceDataPayload.schema(),
            "error": ErrorPayload.schema(),
            "batch_update": BatchUpdatePayload.schema()
        },
        "limits": {
            "max_message_size": MAX_MESSAGE_SIZE,
            "max_payload_size": MAX_PAYLOAD_SIZE
        },
        "version_info": ProtocolVersionInfo().dict()
    }


def export_protocol_schema() -> str:
    """Export protocol schema as JSON string."""
    return json.dumps(get_protocol_documentation(), indent=2) 