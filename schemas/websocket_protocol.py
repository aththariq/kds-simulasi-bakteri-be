"""
WebSocket Message Protocol Specification v1.0
Task 8.2: Message Protocol Design

This module defines the comprehensive WebSocket message protocol for the 
bacterial simulation system, including message types, payload formats, 
validation rules, and versioning strategy.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from dataclasses import dataclass, field
from enum import Enum, auto
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


class MessageType(str, Enum):
    """Comprehensive WebSocket message types organized by category."""
    
    # Client to Server
    START_SIMULATION = "START_SIMULATION"
    PAUSE_SIMULATION = "PAUSE_SIMULATION"
    RESUME_SIMULATION = "RESUME_SIMULATION"
    STOP_SIMULATION = "STOP_SIMULATION"
    GET_STATUS = "GET_STATUS"
    UPDATE_PARAMETERS = "UPDATE_PARAMETERS"
    CLIENT_READY = "CLIENT_READY"  # New: Client signals readiness
    REQUEST_RECONNECT = "REQUEST_RECONNECT" # Client requests reconnection
    
    # Authentication
    AUTH_REQUEST = "AUTH_REQUEST"
    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILED = "AUTH_FAILED"
    
    # Subscription management
    SUBSCRIBE = "SUBSCRIBE"
    UNSUBSCRIBE = "UNSUBSCRIBE"
    SUBSCRIPTION_CONFIRMED = "SUBSCRIPTION_CONFIRMED"
    UNSUBSCRIPTION_CONFIRMED = "UNSUBSCRIPTION_CONFIRMED"

    # Server to Client
    SIMULATION_STARTED = "SIMULATION_STARTED"
    SIMULATION_PAUSED = "SIMULATION_PAUSED"
    SIMULATION_RESUMED = "SIMULATION_RESUMED"
    SIMULATION_STOPPED = "SIMULATION_STOPPED"
    SIMULATION_COMPLETED = "SIMULATION_COMPLETED"
    STATUS_UPDATE = "STATUS_UPDATE"
    DATA_UPDATE = "DATA_UPDATE"
    ERROR = "ERROR"
    WARNING = "WARNING"
    PONG = "PONG"  # Response to PING
    RECONNECT_TOKEN = "RECONNECT_TOKEN" # Server provides a token for reconnection
    RECONNECTION_SUCCESSFUL = "RECONNECTION_SUCCESSFUL"
    RECONNECTION_FAILED = "RECONNECTION_FAILED"
    SYSTEM_STATUS = "SYSTEM_STATUS" # Added based on error
    CONNECTION_ESTABLISHED = "CONNECTION_ESTABLISHED" # Added based on error
    CONNECTION_TERMINATED = "CONNECTION_TERMINATED"

    # Bidirectional
    PING = "PING"
    HEARTBEAT = "HEARTBEAT"  # Used for keep-alive
    DEBUG = "DEBUG" # For debugging purposes
    CONTROL_MESSAGE = "CONTROL_MESSAGE" # Generic control messages
    INFO = "INFO" # General information messages

    @classmethod
    def get_category(cls, message_type: 'MessageType') -> MessageCategory:
        """Get the category for a given message type."""
        # Simplified mapping based on current MessageType values
        if message_type in [cls.START_SIMULATION, cls.PAUSE_SIMULATION, cls.RESUME_SIMULATION, cls.STOP_SIMULATION, cls.UPDATE_PARAMETERS]:
            return MessageCategory.SIMULATION
        elif message_type in [cls.SIMULATION_STARTED, cls.SIMULATION_PAUSED, cls.SIMULATION_RESUMED, cls.SIMULATION_STOPPED, cls.SIMULATION_COMPLETED, cls.DATA_UPDATE]:
            return MessageCategory.DATA
        elif message_type == cls.GET_STATUS or message_type == cls.STATUS_UPDATE or message_type == cls.SYSTEM_STATUS:
            return MessageCategory.SYSTEM
        elif message_type == cls.ERROR:
            return MessageCategory.ERROR
        elif message_type in [cls.PING, cls.PONG, cls.HEARTBEAT]:
            return MessageCategory.HEARTBEAT
        elif message_type in [cls.CLIENT_READY, cls.REQUEST_RECONNECT, cls.RECONNECT_TOKEN, cls.RECONNECTION_SUCCESSFUL, cls.RECONNECTION_FAILED, cls.CONNECTION_ESTABLISHED]:
            return MessageCategory.CONNECTION
        # Default or more granular categories can be added here
        return MessageCategory.SYSTEM # Default category


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