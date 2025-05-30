# WebSocket Message Protocol v1.0 Specification

## Overview

This document defines the WebSocket message protocol for the Bacterial Simulation System. The protocol provides standardized communication between clients and the server for real-time simulation data streaming, control commands, and system management.

## Protocol Information

- **Version**: 1.0
- **Transport**: WebSocket (RFC 6455)
- **Serialization**: JSON
- **Character Encoding**: UTF-8
- **Maximum Message Size**: 1 MB (1,048,576 bytes)
- **Maximum Payload Size**: 512 KB (524,288 bytes)

## Connection Endpoints

### Simulation WebSocket

- **URL**: `/ws/simulation`
- **Purpose**: Real-time simulation data streaming and control
- **Features**: Authentication, subscription management, simulation control

### Global WebSocket

- **URL**: `/ws/global`
- **Purpose**: System-wide monitoring and administration
- **Features**: System status, connection monitoring, global announcements

## Message Structure

### Core Message Format

All WebSocket messages follow this standardized structure:

```json
{
  "type": "message_type",
  "id": "unique-message-id",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "protocol_version": "1.0",
  "priority": "normal",
  "client_id": "client-identifier",
  "session_id": "optional-session-id",
  "simulation_id": "optional-simulation-id",
  "data": {
    // Message-specific payload
  },
  "error": {
    // Error information if applicable
  },
  "compression": "none",
  "encrypted": false,
  "correlation_id": "optional-correlation-id",
  "reply_to": "optional-message-id",
  "expires_at": "optional-expiration-timestamp"
}
```

### Required Fields

- **type**: Message type identifier (see Message Types section)
- **client_id**: Unique client connection identifier
- **timestamp**: ISO 8601 timestamp when message was created

### Optional Fields

- **id**: Unique message identifier (auto-generated if not provided)
- **protocol_version**: Protocol version (defaults to "1.0")
- **priority**: Message priority level (critical, high, normal, low)
- **session_id**: Session identifier for multi-tab support
- **simulation_id**: Target simulation identifier
- **data**: Message payload data
- **error**: Error information for error messages
- **compression**: Compression method used for payload
- **encrypted**: Whether the payload is encrypted
- **correlation_id**: For request-response message correlation
- **reply_to**: ID of message this is replying to
- **expires_at**: Message expiration timestamp

## Message Categories

Messages are organized into logical categories:

### CONNECTION

Connection lifecycle management

- `connection_established`
- `connection_terminated`
- `connection_info`
- `connection_stats`

### AUTHENTICATION

User authentication and authorization

- `auth_request`
- `auth_success`
- `auth_failed`
- `auth_refresh`
- `auth_logout`

### SUBSCRIPTION

Subscription management for real-time updates

- `subscribe`
- `unsubscribe`
- `subscription_confirmed`
- `unsubscription_confirmed`
- `subscription_list`
- `subscription_error`

### SIMULATION

Simulation control and management

- `simulation_start`
- `simulation_stop`
- `simulation_pause`
- `simulation_resume`
- `simulation_reset`
- `simulation_config`
- `simulation_status`

### DATA

Real-time data streaming

- `simulation_update`
- `performance_update`
- `status_update`
- `batch_update`
- `snapshot_update`
- `metrics_update`

### ERROR

Error reporting and handling

- `error`
- `warning`
- `validation_error`
- `rate_limit_error`

### HEARTBEAT

Connection health monitoring

- `ping`
- `pong`

### SYSTEM

System management and monitoring

- `system_status`
- `system_shutdown`
- `system_maintenance`

## Message Priority Levels

Messages are prioritized for processing order:

- **CRITICAL**: System errors, shutdowns (processed immediately)
- **HIGH**: Authentication, connection events (high priority queue)
- **NORMAL**: Regular data updates (standard processing)
- **LOW**: Heartbeats, status updates (background processing)

## Payload Schemas

### Authentication Payload

```json
{
  "api_key": "your-api-key",
  "client_info": {
    "name": "Client Application",
    "version": "1.0.0",
    "platform": "web"
  },
  "refresh_token": "optional-refresh-token"
}
```

### Subscription Payload

```json
{
  "simulation_id": "simulation-identifier",
  "subscription_type": "simulation", // "simulation", "performance", "all"
  "filters": {
    "data_types": ["population", "fitness"],
    "generation_interval": 10
  }
}
```

### Simulation Control Payload

```json
{
  "simulation_id": "simulation-identifier",
  "parameters": {
    "population_size": 1000,
    "mutation_rate": 0.001,
    "selection_pressure": 0.5
  },
  "config": {
    "max_generations": 100,
    "real_time": true
  }
}
```

### Simulation Data Payload

```json
{
  "simulation_id": "simulation-identifier",
  "generation": 42,
  "timestamp": "2024-01-01T12:00:00.000Z",
  "population_data": {
    "total_population": 950,
    "resistant_count": 150,
    "sensitive_count": 800
  },
  "fitness_data": {
    "average_fitness": 0.75,
    "fitness_std": 0.12,
    "max_fitness": 0.95
  },
  "mutation_data": {
    "mutation_events": 5,
    "mutation_rate": 0.001
  },
  "performance_metrics": {
    "generation_time": 0.5,
    "memory_usage": 125000
  }
}
```

### Performance Data Payload

```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "cpu_usage": 45.2,
  "memory_usage": 128000000,
  "network_latency": 15.5,
  "active_connections": 12,
  "message_rate": 150.0
}
```

### Error Payload

```json
{
  "error_code": "SIMULATION_ERROR",
  "error_message": "Simulation failed to start",
  "details": {
    "reason": "Invalid parameters",
    "parameter": "population_size",
    "expected": "> 0"
  },
  "timestamp": "2024-01-01T12:00:00.000Z",
  "severity": "high"
}
```

### Batch Update Payload

```json
{
  "updates": [
    {
      "generation": 1,
      "population": 1000,
      "resistant": 50
    },
    {
      "generation": 2,
      "population": 995,
      "resistant": 55
    }
  ],
  "batch_id": "batch-identifier",
  "total_batches": 5,
  "batch_index": 0
}
```

## Connection Flow

### 1. Connection Establishment

```
Client -> Server: WebSocket connection request
Server -> Client: connection_established message
```

### 2. Authentication (Optional)

```
Client -> Server: auth_request message
Server -> Client: auth_success or auth_failed message
```

### 3. Subscription Management

```
Client -> Server: subscribe message
Server -> Client: subscription_confirmed message
```

### 4. Data Streaming

```
Server -> Client: simulation_update messages (real-time)
Server -> Client: performance_update messages (periodic)
```

### 5. Simulation Control

```
Client -> Server: simulation_start message
Server -> Client: status_update message
Server -> Client: simulation_update messages (continuous)
```

### 6. Heartbeat Monitoring

```
Server -> Client: ping message (every 30 seconds)
Client -> Server: pong message (response)
```

### 7. Connection Termination

```
Client/Server -> Peer: connection_terminated message
WebSocket connection closed
```

## Error Handling

### Error Message Format

All errors follow the standardized error payload format with:

- **error_code**: Specific error identifier
- **error_message**: Human-readable error description
- **details**: Additional error context
- **severity**: Error severity level

### Common Error Codes

- `AUTH_FAILED`: Authentication failure
- `INVALID_MESSAGE`: Malformed message
- `SUBSCRIPTION_ERROR`: Subscription management failure
- `SIMULATION_ERROR`: Simulation operation failure
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `MESSAGE_TOO_LARGE`: Message exceeds size limits
- `PROTOCOL_VERSION_MISMATCH`: Unsupported protocol version

## Validation Rules

### Message Validation

- Messages must be valid JSON
- Required fields must be present
- Field types must match specification
- Message size must not exceed limits

### Payload Validation

- Payloads are validated against specific schemas
- Field constraints are enforced (min/max values, string lengths)
- Enum values must be from allowed sets
- Nested objects must follow sub-schemas

### Client ID Validation

- Must be 8-128 characters
- Only alphanumeric characters, hyphens, and underscores
- Must be unique per connection

### Simulation ID Validation

- Must be 10-100 characters
- Only alphanumeric characters, hyphens, and underscores
- Must reference existing simulation

## Compression Support

Large payloads can be compressed using:

- **none**: No compression (default)
- **gzip**: Gzip compression
- **deflate**: Deflate compression

Compression is indicated in the `compression` field.

## Versioning Strategy

### Protocol Versioning

- Current version: 1.0
- Version format: MAJOR.MINOR
- Breaking changes increment MAJOR version
- New features increment MINOR version

### Compatibility

- Server supports multiple protocol versions
- Clients must specify protocol version
- Deprecated versions are phased out gradually

## Security Considerations

### Authentication

- API key authentication required for sensitive operations
- Session management with optional refresh tokens
- Rate limiting enforced per client

### Message Security

- Input validation and sanitization
- Message size limits to prevent DoS
- Client ID sanitization to prevent injection

### Connection Security

- WebSocket over TLS (WSS) in production
- Connection timeout and cleanup
- Heartbeat monitoring for connection health

## Implementation Examples

### JavaScript Client

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/simulation");

// Send authentication
ws.send(
  JSON.stringify({
    type: "auth_request",
    client_id: "my-client-id",
    data: {
      api_key: "your-api-key",
    },
  })
);

// Subscribe to simulation
ws.send(
  JSON.stringify({
    type: "subscribe",
    client_id: "my-client-id",
    simulation_id: "sim-123",
    data: {
      simulation_id: "sim-123",
      subscription_type: "simulation",
    },
  })
);

// Handle messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log("Received:", message.type, message.data);
};
```

### Python Client

```python
import asyncio
import websockets
import json

async def client():
    uri = "ws://localhost:8000/ws/simulation"
    async with websockets.connect(uri) as websocket:
        # Send authentication
        auth_message = {
            "type": "auth_request",
            "client_id": "my-client-id",
            "data": {
                "api_key": "your-api-key"
            }
        }
        await websocket.send(json.dumps(auth_message))

        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['type']}")
```

## Testing and Debugging

### Message Validation

Use the protocol validator to check message format:

```python
from schemas.websocket_protocol import ProtocolValidator

is_valid, error = ProtocolValidator.validate_message_format(message_data)
if not is_valid:
    print(f"Validation error: {error}")
```

### Protocol Documentation

Generate protocol schema for testing tools:

```python
from schemas.websocket_protocol import export_protocol_schema

schema_json = export_protocol_schema()
with open('protocol_schema.json', 'w') as f:
    f.write(schema_json)
```

## Future Extensions

### Planned Features

- Binary message support for large data transfers
- Message acknowledgment system
- Connection resumption after disconnects
- Client-side message filtering
- Encryption for sensitive data

### Version 1.1 Roadmap

- Enhanced batch processing
- Real-time collaboration features
- Advanced subscription filters
- Performance optimizations

---

This protocol specification ensures reliable, secure, and efficient WebSocket communication for the bacterial simulation system while providing room for future enhancements and extensions.
