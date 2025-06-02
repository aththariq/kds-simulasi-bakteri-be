"""
Enhanced WebSocket routes for real-time simulation updates.
Task 8.1: WebSocket Connection Setup - Comprehensive implementation.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from typing import Dict, List, Optional, Any
import json
import uuid
import asyncio
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

# Import MessageType and WebSocketProtocolMessage from the protocol schema
from schemas.websocket_protocol import MessageType, WebSocketProtocolMessage, MessageFactory
from services.simulation_service import SimulationService
from services.reconnection_service import get_reconnection_manager, ReconnectionState
from services.websocket_error_handler import ErrorHandler, ErrorCode, ErrorSeverity, ErrorCategory
from utils.auth import verify_api_key_websocket
from services.websocket_optimizations import (
    get_optimized_websocket_service,
    MessageOptimizationLevel
)

router = APIRouter(prefix="/ws", tags=["WebSocket"])
logger = logging.getLogger(__name__)

# Global simulation service instance
simulation_service = SimulationService()


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """Simplified WebSocket message structure for backward compatibility."""
    type: MessageType
    timestamp: str
    client_id: str
    simulation_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        message_dict = asdict(self)
        message_dict['type'] = self.type.value
        return json.dumps(message_dict)
    
    @classmethod
    def from_json(cls, json_str: str, client_id: str) -> 'WebSocketMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        # Handle both 'data' and 'payload' field names for frontend compatibility
        message_data = data.get('data') or data.get('payload')
        
        # Convert string message type to MessageType enum
        message_type_str = data.get('type', '').upper()
        try:
            message_type = MessageType(message_type_str)
        except ValueError:
            # Try lowercase version for backward compatibility
            try:
                message_type = MessageType(data.get('type'))
            except ValueError:
                logger.warning(f"Unknown message type: {data.get('type')}")
                message_type = MessageType.ERROR
        
        return cls(
            type=message_type,
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            client_id=client_id,
            simulation_id=data.get('simulation_id'),
            data=message_data,
            error=data.get('error')
        )
    
    def to_protocol_message(self) -> WebSocketProtocolMessage:
        """Convert to protocol message format."""
        return WebSocketProtocolMessage(
            type=self.type,
            timestamp=self.timestamp,
            client_id=self.client_id,
            simulation_id=self.simulation_id,
            data=self.data,
            error={"error_message": self.error, "error_code": "GENERAL_ERROR"} if self.error else None
        )


@dataclass
class ClientConnection:
    """WebSocket client connection information."""
    client_id: str
    websocket: WebSocket
    state: ConnectionState
    connected_at: datetime
    last_activity: datetime
    subscriptions: List[str]
    authenticated: bool = False
    user_id: Optional[str] = None
    heartbeat_count: int = 0
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'client_id': self.client_id,
            'state': self.state.value,
            'connected_at': self.connected_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'subscriptions': self.subscriptions,
            'authenticated': self.authenticated,
            'user_id': self.user_id,
            'heartbeat_count': self.heartbeat_count
        }


class EnhancedConnectionManager:
    """Enhanced WebSocket connection manager with authentication and lifecycle management."""
    
    def __init__(self):
        self.active_connections: Dict[str, ClientConnection] = {}
        self.simulation_subscribers: Dict[str, List[str]] = {}
        self.heartbeat_interval: float = 30.0  # seconds
        self.connection_timeout: float = 300.0  # 5 minutes
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_started: bool = False
        
        # Lazy initialization for heavy dependencies
        self._reconnection_manager = None
        self._error_handler = None
        self._callbacks_registered = False
        
        # Don't start heartbeat automatically during import
        # It will be started when first connection is made
    
    @property
    def reconnection_manager(self):
        """Lazy initialization of reconnection manager."""
        if self._reconnection_manager is None:
            self._reconnection_manager = get_reconnection_manager()
            if not self._callbacks_registered:
                self._reconnection_manager.register_reconnection_callback(self._handle_reconnection_state_change)
                self._reconnection_manager.register_message_replay_callback(self._handle_message_replay)
                self._callbacks_registered = True
        return self._reconnection_manager
    
    @property
    def error_handler(self):
        """Lazy initialization of error handler."""
        if self._error_handler is None:
            self._error_handler = ErrorHandler()
        return self._error_handler
    
    def _start_heartbeat_monitor(self):
        """Start the heartbeat monitoring task."""
        if not self._heartbeat_started and (self._heartbeat_task is None or self._heartbeat_task.done()):
            try:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
                self._heartbeat_started = True
                logger.info("Heartbeat monitor started")
            except RuntimeError:
                # No event loop running, will start later when needed
                logger.debug("No event loop running, heartbeat monitor will start later")
    
    async def _heartbeat_monitor(self):
        """Monitor client connections and send heartbeat pings."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                current_time = datetime.now()
                
                # Check for inactive connections and send heartbeats
                disconnected_clients = []
                for client_id, connection in self.active_connections.items():
                    time_since_activity = (current_time - connection.last_activity).total_seconds()
                    
                    if time_since_activity > self.connection_timeout:
                        # Connection timed out
                        logger.warning(f"Connection {client_id} timed out after {time_since_activity}s")
                        disconnected_clients.append(client_id)
                    elif time_since_activity > self.heartbeat_interval:
                        # Send heartbeat
                        await self._send_heartbeat(client_id)
                
                # Clean up timed out connections
                for client_id in disconnected_clients:
                    await self.disconnect(client_id, reason="timeout")
                    
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _send_heartbeat(self, client_id: str):
        """Send heartbeat ping to a client."""
        try:
            connection = self.active_connections[client_id]
            message = WebSocketMessage(
                type=MessageType.PING,
                timestamp=datetime.now().isoformat(),
                client_id=client_id
            )
            await connection.websocket.send_text(message.to_json())
            connection.heartbeat_count += 1
            logger.debug(f"Sent heartbeat to client {client_id}")
        except Exception as e:
            logger.error(f"Failed to send heartbeat to {client_id}: {e}")
            await self.disconnect(client_id, reason="heartbeat_failed")
    
    def _handle_reconnection_state_change(self, state: ReconnectionState):
        """Handle reconnection state changes."""
        logger.info(f"Reconnection state changed to: {state.value}")
        # Could notify clients about server reconnection status here
    
    def _handle_message_replay(self, client_id: str, messages: List[Dict[str, Any]]):
        """Handle message replay after reconnection."""
        logger.info(f"Replaying {len(messages)} messages for client {client_id}")
        # Messages would be replayed when client reconnects
        # This is handled by the reconnection service
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """Accept a new WebSocket connection and return client ID."""
        if not client_id:
            client_id = str(uuid.uuid4())
        
        await websocket.accept()
        
        # Start heartbeat monitor if not already started
        if not self._heartbeat_started:
            self._start_heartbeat_monitor()
        
        # Check for existing session recovery
        session_state = await self.reconnection_manager.handle_successful_connection(client_id)
        
        # Create connection record
        connection = ClientConnection(
            client_id=client_id,
            websocket=websocket,
            state=ConnectionState.CONNECTED,
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            subscriptions=[]
        )
        
        # Restore session state if available
        if session_state:
            connection.subscriptions = list(session_state.subscriptions)
            logger.info(f"Restored session for client {client_id} with {len(session_state.subscriptions)} subscriptions")
        
        self.active_connections[client_id] = connection
        
        # Send connection established message
        message = WebSocketMessage(
            type=MessageType.CONNECTION_ESTABLISHED,
            timestamp=datetime.now().isoformat(),
            client_id=client_id,
            data={
                "server_info": "Bacterial Simulation WebSocket Server",
                "protocol_version": "1.0",
                "heartbeat_interval": self.heartbeat_interval,
                "supported_features": [
                    "simulation_streaming",
                    "real_time_updates",
                    "authentication",
                    "subscription_management",
                    "automatic_reconnection",
                    "session_recovery"
                ],
                "session_recovered": session_state is not None,
                "restored_subscriptions": connection.subscriptions if session_state else []
            }
        )
        
        await websocket.send_text(message.to_json())
        logger.info(f"New WebSocket connection established: {client_id}")
        
        return client_id
    
    async def disconnect(self, client_id: str, reason: str = "unknown"):
        """Disconnect a WebSocket client."""
        if client_id not in self.active_connections:
            return
            
        connection = self.active_connections[client_id]
        connection.state = ConnectionState.DISCONNECTING
        
        # Save session state for potential reconnection
        subscriptions = set(connection.subscriptions)
        auth_token = getattr(connection, 'auth_token', None)
        await self.reconnection_manager.save_session(
            client_id=client_id,
            auth_token=auth_token,
            subscriptions=subscriptions
        )
        
        # Handle disconnection in reconnection manager
        await self.reconnection_manager.handle_disconnect(client_id, reason)
        
        try:
            # Send disconnection message
            message = WebSocketMessage(
                type=MessageType.CONNECTION_TERMINATED,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                data={"reason": reason}
            )
            await connection.websocket.send_text(message.to_json())
        except:
            pass  # Connection might already be closed
        
        # Remove from active connections
        del self.active_connections[client_id]
        
        # Remove from all simulation subscriptions
        for simulation_id in list(self.simulation_subscribers.keys()):
            if client_id in self.simulation_subscribers[simulation_id]:
                self.simulation_subscribers[simulation_id].remove(client_id)
                if not self.simulation_subscribers[simulation_id]:
                    del self.simulation_subscribers[simulation_id]
        
        logger.info(f"WebSocket connection disconnected: {client_id} (reason: {reason})")
    
    async def authenticate_client(self, client_id: str, api_key: str) -> bool:
        """Authenticate a WebSocket client."""
        if client_id not in self.active_connections:
            return False
        
        try:
            # Verify API key (this would integrate with your auth system)
            # For now, we'll use a simple check
            user_id = await verify_api_key_websocket(api_key)
            
            connection = self.active_connections[client_id]
            connection.authenticated = True
            connection.user_id = user_id
            connection.state = ConnectionState.AUTHENTICATED
            connection.update_activity()
            
            # Send authentication success
            message = WebSocketMessage(
                type=MessageType.AUTH_SUCCESS,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                data={"user_id": user_id}
            )
            await connection.websocket.send_text(message.to_json())
            
            logger.info(f"Client {client_id} authenticated as user {user_id}")
            return True
            
        except Exception as e:
            # Send authentication failure
            message = WebSocketMessage(
                type=MessageType.AUTH_FAILED,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                error=str(e)
            )
            await self.active_connections[client_id].websocket.send_text(message.to_json())
            logger.warning(f"Authentication failed for client {client_id}: {e}")
            return False
    
    async def subscribe_to_simulation(self, client_id: str, simulation_id: str) -> bool:
        """Subscribe a client to simulation updates."""
        if client_id not in self.active_connections:
            return False
        
        connection = self.active_connections[client_id]
        
        # Add to simulation subscribers
        if simulation_id not in self.simulation_subscribers:
            self.simulation_subscribers[simulation_id] = []
        if client_id not in self.simulation_subscribers[simulation_id]:
            self.simulation_subscribers[simulation_id].append(client_id)
        
        # Add to client's subscriptions
        if simulation_id not in connection.subscriptions:
            connection.subscriptions.append(simulation_id)
        
        connection.update_activity()
        
        # Send subscription confirmation
        message = WebSocketMessage(
            type=MessageType.SUBSCRIPTION_CONFIRMED,
            timestamp=datetime.now().isoformat(),
            client_id=client_id,
            simulation_id=simulation_id,
            data={"subscriber_count": len(self.simulation_subscribers[simulation_id])}
        )
        await connection.websocket.send_text(message.to_json())
        
        logger.info(f"Client {client_id} subscribed to simulation {simulation_id}")
        return True
    
    async def unsubscribe_from_simulation(self, client_id: str, simulation_id: str) -> bool:
        """Unsubscribe a client from simulation updates."""
        if client_id not in self.active_connections:
            return False
        
        connection = self.active_connections[client_id]
        
        # Remove from simulation subscribers
        if simulation_id in self.simulation_subscribers:
            if client_id in self.simulation_subscribers[simulation_id]:
                self.simulation_subscribers[simulation_id].remove(client_id)
                if not self.simulation_subscribers[simulation_id]:
                    del self.simulation_subscribers[simulation_id]
        
        # Remove from client's subscriptions
        if simulation_id in connection.subscriptions:
            connection.subscriptions.remove(simulation_id)
        
        connection.update_activity()
        
        # Send unsubscription confirmation
        message = WebSocketMessage(
            type=MessageType.UNSUBSCRIPTION_CONFIRMED,
            timestamp=datetime.now().isoformat(),
            client_id=client_id,
            simulation_id=simulation_id
        )
        await connection.websocket.send_text(message.to_json())
        
        logger.info(f"Client {client_id} unsubscribed from simulation {simulation_id}")
        return True
    
    async def send_to_client(self, client_id: str, message: WebSocketMessage) -> bool:
        """Send a message to a specific client."""
        if client_id not in self.active_connections:
            # Client is disconnected, queue message for potential replay
            message_dict = {
                "type": message.type.value,
                "timestamp": message.timestamp,
                "client_id": message.client_id,
                "simulation_id": message.simulation_id,
                "data": message.data,
                "error": message.error
            }
            queued = self.reconnection_manager.queue_message(client_id, message_dict)
            if queued:
                logger.debug(f"Queued message for disconnected client {client_id}")
            return False
        
        try:
            connection = self.active_connections[client_id]
            await connection.websocket.send_text(message.to_json())
            connection.update_activity()
            return True
        except Exception as e:
            logger.error(f"Failed to send message to client {client_id}: {e}")
            # Queue message before disconnecting
            message_dict = {
                "type": message.type.value,
                "timestamp": message.timestamp,
                "client_id": message.client_id,
                "simulation_id": message.simulation_id,
                "data": message.data,
                "error": message.error
            }
            self.reconnection_manager.queue_message(client_id, message_dict)
            await self.disconnect(client_id, reason="send_failed")
            return False
    
    async def broadcast_to_simulation(self, simulation_id: str, message: WebSocketMessage) -> int:
        """Broadcast a message to all clients subscribed to a simulation."""
        if simulation_id not in self.simulation_subscribers:
            return 0
        
        successful_sends = 0
        failed_clients = []
        
        for client_id in self.simulation_subscribers[simulation_id].copy():
            success = await self.send_to_client(client_id, message)
            if success:
                successful_sends += 1
            else:
                failed_clients.append(client_id)
        
        # Clean up failed clients
        for client_id in failed_clients:
            await self.disconnect(client_id, reason="broadcast_failed")
        
        return successful_sends
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "authenticated_connections": sum(1 for c in self.active_connections.values() if c.authenticated),
            "total_subscriptions": sum(len(subs) for subs in self.simulation_subscribers.values()),
            "active_simulations": len(self.simulation_subscribers),
            "connections": [conn.to_dict() for conn in self.active_connections.values()]
        }


# Global enhanced connection manager
manager = EnhancedConnectionManager()


@router.websocket("/simulation")
async def simulation_websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None, description="Optional client ID for reconnection")
):
    """Enhanced WebSocket endpoint for simulation data with optimization support."""
    try:
        # Connect with optimization support
        final_client_id = await manager.connect(websocket, client_id)
        optimization_service = get_optimized_websocket_service()
        
        # Set default optimization level for new clients
        optimization_service.set_client_optimization_level(
            final_client_id, 
            MessageOptimizationLevel.COMPRESSED
        )
        
        logger.info(f"Client {final_client_id} connected to simulation WebSocket with optimization enabled")
        
        # Send connection confirmation with optimization info
        connection_message = WebSocketMessage(
            type=MessageType.CONNECTION_ESTABLISHED,
            timestamp=datetime.now().isoformat(),
            client_id=final_client_id,
            data={
                "server_info": {
                    "version": "1.0",
                    "supported_features": [
                        "real_time_updates", 
                        "heartbeat", 
                        "reconnection",
                        "optimization",
                        "compression",
                        "delta_updates"
                    ],
                    "optimization_enabled": optimization_service.optimization_enabled,
                    "optimization_level": optimization_service.get_client_optimization_level(final_client_id).value
                },
                "client_id": final_client_id,
                "reconnection_id": manager.reconnection_manager.generate_reconnection_id(final_client_id)
            }
        )
        
        # Send using optimized service
        await optimization_service.send_optimized_message(
            websocket, 
            final_client_id, 
            asdict(connection_message)
        )
          # Start message handling loop
        async for message in websocket.iter_text():
            print(f"=== RECEIVED RAW MESSAGE ===")
            print(f"Client: {final_client_id}")
            print(f"Raw message: {message}")
            try:
                # Parse incoming message
                try:
                    parsed_message = WebSocketMessage.from_json(message, final_client_id)
                    print(f"Parsed message type: {parsed_message.type}")
                    print(f"Parsed message data: {parsed_message.data}")
                except Exception as parse_error:
                    print(f"PARSE ERROR: {parse_error}")
                    logger.error(f"Failed to parse message from {final_client_id}: {parse_error}")
                    await manager.send_to_client(
                        final_client_id,
                        WebSocketMessage(
                            type=MessageType.ERROR,
                            timestamp=datetime.now().isoformat(),
                            client_id=final_client_id,
                            error=f"Invalid message format: {parse_error}"
                        )
                    )
                    continue
                
                # Handle optimization-specific messages
                if parsed_message.type == MessageType.SYSTEM_STATUS and parsed_message.data:
                    if parsed_message.data.get('action') == 'set_optimization_level':
                        level_str = parsed_message.data.get('level', 'compressed')
                        try:
                            level = MessageOptimizationLevel(level_str)
                            optimization_service.set_client_optimization_level(final_client_id, level)
                            
                            # Confirm optimization level change
                            response = WebSocketMessage(
                                type=MessageType.SYSTEM_STATUS,
                                timestamp=datetime.now().isoformat(),
                                client_id=final_client_id,
                                data={
                                    "action": "optimization_level_updated",
                                    "level": level.value,
                                    "message": f"Optimization level set to {level.value}"
                                }
                            )
                            await optimization_service.send_optimized_message(
                                websocket,
                                final_client_id,
                                asdict(response)
                            )
                            continue
                        except ValueError:
                            logger.warning(f"Invalid optimization level requested: {level_str}")
                
                # Process regular messages
                await handle_websocket_message(final_client_id, parsed_message)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error from {final_client_id}: {e}")
                await manager.send_to_client(
                    final_client_id,
                    WebSocketMessage(
                        type=MessageType.ERROR,
                        timestamp=datetime.now().isoformat(),
                        client_id=final_client_id,
                        error="Invalid JSON format"
                    )
                )
            except Exception as e:
                logger.error(f"Error processing message from {final_client_id}: {e}")
                await manager.error_handler.handle_websocket_error(final_client_id, e)
    
    except WebSocketDisconnect:
        logger.info(f"Client {final_client_id} disconnected")
        await manager.disconnect(final_client_id, "client_disconnected")
        optimization_service.cleanup_client(final_client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {final_client_id}: {e}")
        await manager.error_handler.handle_websocket_error(final_client_id, e)
        optimization_service.cleanup_client(final_client_id)


async def handle_websocket_message(client_id: str, message: WebSocketMessage):
    """Handle incoming WebSocket messages based on type."""
    print(f"=== WEBSOCKET MESSAGE HANDLER CALLED ===")
    print(f"Client ID: {client_id}")
    print(f"Message type: {message.type}")
    print(f"Message simulation_id: {message.simulation_id}")
    print(f"Message data: {message.data}")
    
    if message.type == MessageType.AUTH_REQUEST:
        # Handle authentication
        api_key = message.data.get("api_key") if message.data else None
        if api_key:
            await manager.authenticate_client(client_id, api_key)
        else:
            error_msg = WebSocketMessage(
                type=MessageType.AUTH_FAILED,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                error="API key required for authentication"
            )
            await manager.send_to_client(client_id, error_msg)
    
    elif message.type == MessageType.SUBSCRIBE:
        # Handle subscription
        if message.simulation_id:
            await manager.subscribe_to_simulation(client_id, message.simulation_id)
        else:
            error_msg = WebSocketMessage(
                type=MessageType.ERROR,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                error="Simulation ID required for subscription"
            )
            await manager.send_to_client(client_id, error_msg)
    
    elif message.type == MessageType.UNSUBSCRIBE:
        # Handle unsubscription
        if message.simulation_id:
            await manager.unsubscribe_from_simulation(client_id, message.simulation_id)
        else:
            error_msg = WebSocketMessage(
                type=MessageType.ERROR,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                error="Simulation ID required for unsubscription"
            )
            await manager.send_to_client(client_id, error_msg)
    
    elif message.type == MessageType.SIMULATION_START:
        # Handle simulation start request
        logger.info(f"Processing SIMULATION_START message for client {client_id}")
        logger.info(f"Message simulation_id: {message.simulation_id}")
        logger.info(f"Message data: {message.data}")
        
        if message.simulation_id and message.data:
            # Check if client is authenticated (optional requirement)
            connection = manager.active_connections.get(client_id)
            if connection and (not hasattr(connection, 'authenticated') or True):  # Allow non-authenticated for now
                try:
                    # Create simulation with parameters from the message
                    simulation_params = message.data
                    logger.info(f"Creating simulation with params: {simulation_params}")
                    
                    # Create the simulation first
                    simulation_result = simulation_service.create_simulation(
                        simulation_id=message.simulation_id,
                        initial_population_size=simulation_params.get('initial_population_size', 1000),
                        mutation_rate=simulation_params.get('mutation_rate', 0.01),
                        selection_pressure=simulation_params.get('selection_pressure', 0.5),
                        antibiotic_concentration=simulation_params.get('antibiotic_concentration', 1.0),
                        simulation_time=simulation_params.get('simulation_time', 100)
                    )
                    
                    logger.info(f"Created simulation {message.simulation_id} successfully")
                    logger.info(f"Simulation result: {simulation_result}")
                    
                    # Start streaming simulation updates
                    logger.info(f"Starting simulation stream for {message.simulation_id}")
                    asyncio.create_task(stream_simulation_updates(message.simulation_id, client_id))                    
                    response_msg = WebSocketMessage(
                        type=MessageType.STATUS_UPDATE,
                        timestamp=datetime.now().isoformat(),
                        client_id=client_id,
                        simulation_id=message.simulation_id,
                        data={
                            "status": "simulation_started",
                            "simulation_id": message.simulation_id,
                            "parameters": simulation_params
                        }
                    )
                    await manager.send_to_client(client_id, response_msg)
                    
                except Exception as e:
                    logger.error(f"Failed to create/start simulation {message.simulation_id}: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    error_msg = WebSocketMessage(
                        type=MessageType.ERROR,
                        timestamp=datetime.now().isoformat(),
                        client_id=client_id,
                        simulation_id=message.simulation_id,
                        error=f"Failed to start simulation: {str(e)}"
                    )
                    await manager.send_to_client(client_id, error_msg)
            else:
                error_msg = WebSocketMessage(
                    type=MessageType.ERROR,
                    timestamp=datetime.now().isoformat(),
                    client_id=client_id,
                    error="Authentication required to start simulations"
                )
                await manager.send_to_client(client_id, error_msg)
        else:
            error_msg = WebSocketMessage(
                type=MessageType.ERROR,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                error="simulation_id and simulation parameters required"
            )
            await manager.send_to_client(client_id, error_msg)
    
    elif message.type == MessageType.PONG:
        # Handle heartbeat response
        connection = manager.active_connections.get(client_id)
        if connection:
            connection.update_activity()
            logger.debug(f"Received heartbeat response from client {client_id}")
    
    else:
        # Unknown message type
        error_msg = WebSocketMessage(
            type=MessageType.ERROR,
            timestamp=datetime.now().isoformat(),
            client_id=client_id,
            error=f"Unknown message type: {message.type.value}"
        )
        await manager.send_to_client(client_id, error_msg)


async def stream_simulation_updates(simulation_id: str, client_id: str):
    """
    Stream simulation updates to WebSocket clients.
    Enhanced with error handling and performance monitoring.
    """
    try:
        logger.info(f"Starting simulation stream for {simulation_id} to client {client_id}")
        
        async for progress_data in simulation_service.run_simulation_async(simulation_id):
            # Create update message
            update_message = WebSocketMessage(
                type=MessageType.SIMULATION_UPDATE,
                timestamp=datetime.now().isoformat(),
                client_id=client_id,
                simulation_id=simulation_id,
                data=progress_data
            )
            
            # Send to requesting client
            await manager.send_to_client(client_id, update_message)
            
            # Broadcast to all subscribers
            await manager.broadcast_to_simulation(simulation_id, update_message)
            
            # Send spatial-specific updates if spatial data is available
            if "spatial_data" in progress_data:
                spatial_message = WebSocketMessage(
                    type=MessageType.SIMULATION_UPDATE,
                    timestamp=datetime.now().isoformat(),
                    client_id=client_id,
                    simulation_id=simulation_id,
                    data={
                        "type": "spatial_update",
                        **progress_data["spatial_data"]
                    }
                )
                
                # Broadcast spatial data to all simulation subscribers
                await manager.broadcast_to_simulation(simulation_id, spatial_message)
            
    except Exception as e:
        logger.error(f"Error in simulation stream {simulation_id}: {e}")
        error_message = WebSocketMessage(
            type=MessageType.ERROR,
            timestamp=datetime.now().isoformat(),
            client_id=client_id,
            simulation_id=simulation_id,
            error=f"Simulation streaming error: {str(e)}"
        )
        await manager.send_to_client(client_id, error_message)


@router.websocket("/global")
async def global_websocket_endpoint(websocket: WebSocket):
    """
    Global WebSocket endpoint for system-wide updates and monitoring.
    """
    client_id = await manager.connect(websocket)
    
    try:
        while True:
            raw_data = await websocket.receive_text()
            
            try:
                message = WebSocketMessage.from_json(raw_data, client_id)
                
                # Handle global commands
                if message.type == MessageType.AUTH_REQUEST:
                    api_key = message.data.get("api_key") if message.data else None
                    if api_key:
                        await manager.authenticate_client(client_id, api_key)
                
                # Echo message for testing
                echo_message = WebSocketMessage(
                    type=MessageType.STATUS_UPDATE,
                    timestamp=datetime.now().isoformat(),
                    client_id=client_id,
                    data={"echo": message.data, "connection_stats": manager.get_connection_stats()}
                )
                await manager.send_to_client(client_id, echo_message)
                
            except json.JSONDecodeError:
                error_message = WebSocketMessage(
                    type=MessageType.ERROR,
                    timestamp=datetime.now().isoformat(),
                    client_id=client_id,
                    error="Invalid JSON format"
                )
                await manager.send_to_client(client_id, error_message)
                
    except WebSocketDisconnect:
        await manager.disconnect(client_id, reason="client_disconnect")


@router.websocket("/spatial/{simulation_id}")
async def spatial_websocket_endpoint(
    websocket: WebSocket,
    simulation_id: str,
    client_id: Optional[str] = Query(None, description="Optional client ID for reconnection")
):
    """
    WebSocket endpoint specifically for spatial/Petri dish visualization data.
    
    Provides real-time updates of:
    - Bacterial positions and movements
    - Spatial grid statistics
    - Antibiotic zone data
    - Resistance status visualization data
    """
    
    # Establish connection
    client_id = await manager.connect(websocket, client_id)
    
    try:
        # Subscribe to simulation updates
        await manager.subscribe_to_simulation(client_id, simulation_id)
        
        # Send initial spatial data if simulation exists
        if simulation_id in simulation_service.active_simulations:
            sim_data = simulation_service.active_simulations[simulation_id]
            
            if "spatial_grid" in sim_data:
                # Send initial spatial state
                initial_message = WebSocketMessage(
                    type=MessageType.SIMULATION_UPDATE,
                    timestamp=datetime.now().isoformat(),
                    client_id=client_id,
                    simulation_id=simulation_id,
                    data={
                        "type": "spatial_initialization",
                        "grid_dimensions": [sim_data["spatial_grid"].width, sim_data["spatial_grid"].height],
                        "cell_size": sim_data["spatial_grid"].cell_size,
                        "boundary_condition": sim_data["spatial_grid"].boundary_condition.value,
                        "current_generation": sim_data.get("current_generation", 0)
                    }
                )
                await websocket.send_text(initial_message.to_json())
        
        while True:
            # Listen for client messages
            raw_data = await websocket.receive_text()
            
            try:
                message = WebSocketMessage.from_json(raw_data, client_id)
                
                if message.type == MessageType.PING:
                    # Respond to ping
                    pong_message = WebSocketMessage(
                        type=MessageType.PONG,
                        timestamp=datetime.now().isoformat(),
                        client_id=client_id,
                        simulation_id=simulation_id
                    )
                    await websocket.send_text(pong_message.to_json())
                
                elif message.data and message.data.get("type") == "get_spatial_data":
                    # Send current spatial data
                    if simulation_id in simulation_service.active_simulations:
                        sim_data = simulation_service.active_simulations[simulation_id]
                        if "spatial_data" in sim_data.get("results", {}):
                            spatial_response = WebSocketMessage(
                                type=MessageType.SIMULATION_UPDATE,
                                timestamp=datetime.now().isoformat(),
                                client_id=client_id,
                                simulation_id=simulation_id,
                                data={
                                    "type": "spatial_update",
                                    **sim_data["results"]["spatial_data"]
                                }
                            )
                            await websocket.send_text(spatial_response.to_json())
                
            except json.JSONDecodeError as e:
                error_message = WebSocketMessage(
                    type=MessageType.ERROR,
                    timestamp=datetime.now().isoformat(),
                    client_id=client_id,
                    error=f"Invalid JSON format: {str(e)}"
                )
                await websocket.send_text(error_message.to_json())
                
    except WebSocketDisconnect:
        await manager.disconnect(client_id, reason="spatial_client_disconnect")
    except Exception as e:
        logger.error(f"Error in spatial WebSocket for {simulation_id}: {e}")
        await manager.disconnect(client_id, reason="spatial_connection_error")


@router.websocket("/performance")
async def performance_websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None, description="Optional client ID for reconnection")
):
    """WebSocket endpoint for real-time performance monitoring with optimization."""
    manager = get_connection_manager()
    optimization_service = get_optimized_websocket_service()
    actual_client_id = await manager.connect(websocket, client_id)
    
    # Set performance monitoring optimization level
    optimization_service.set_client_optimization_level(
        actual_client_id, 
        MessageOptimizationLevel.DELTA
    )
    
    try:
        # Import performance profiler
        from utils.performance_profiler import get_profiler
        profiler = get_profiler()
        
        logger.info(f"Performance monitoring client {actual_client_id} connected with optimization")
        
        # Send initial connection message with optimization info
        welcome_message = WebSocketMessage(
            type=MessageType.CONNECTION_ESTABLISHED,
            timestamp=datetime.now().isoformat(),
            client_id=actual_client_id,
            data={
                "message": "Connected to performance monitoring",
                "optimization_enabled": optimization_service.optimization_enabled,
                "optimization_level": optimization_service.get_client_optimization_level(actual_client_id).value,
                "features": [
                    "real_time_system_metrics",
                    "function_performance_tracking", 
                    "performance_alerts",
                    "baseline_notifications",
                    "optimized_transmission",
                    "delta_compression"
                ]
            }
        )
        
        # Send using optimized service
        await optimization_service.send_optimized_message(
            websocket, 
            actual_client_id, 
            asdict(welcome_message)
        )
        
        # Register WebSocket callback with profiler
        async def performance_callback(message: Dict[str, Any]):
            """Callback for performance updates with optimization."""
            try:
                perf_message = WebSocketMessage(
                    type=MessageType.PERFORMANCE_UPDATE,
                    timestamp=datetime.now().isoformat(),
                    client_id=actual_client_id,
                    data=message
                )
                
                # Use delta compression for performance updates
                await optimization_service.send_optimized_message(
                    websocket,
                    actual_client_id,
                    asdict(perf_message),
                    use_delta=True,
                    use_batching=False
                )
            except Exception as e:
                logger.error(f"Failed to send performance update: {e}")
        
        # Subscribe to performance updates
        profiler.subscribe_websocket(performance_callback)
        
        # Start task for periodic system metrics
        async def send_system_metrics():
            """Send periodic system metrics updates with optimization."""
            while True:
                try:
                    # Get recent system metrics
                    system_metrics = profiler.realtime_collector.get_recent_metrics(1)
                    if system_metrics:
                        metrics_message = WebSocketMessage(
                            type=MessageType.PERFORMANCE_UPDATE,
                            timestamp=datetime.now().isoformat(),
                            client_id=actual_client_id,
                            data={
                                "type": "system_metrics",
                                "metrics": system_metrics[0]
                            }
                        )
                        
                        # Use delta compression for frequent metrics
                        await optimization_service.send_optimized_message(
                            websocket,
                            actual_client_id,
                            asdict(metrics_message),
                            use_delta=True
                        )
                    
                    # Send performance summary every 10 seconds
                    await asyncio.sleep(10)
                    
                    # Get performance summary
                    summary = profiler.get_performance_summary()
                    summary_message = WebSocketMessage(
                        type=MessageType.PERFORMANCE_UPDATE,
                        timestamp=datetime.now().isoformat(),
                        client_id=actual_client_id,
                        simulation_id="performance_monitoring",
                        data={
                            "type": "performance_summary",
                            "summary": summary
                        }
                    )
                    
                    # Use delta compression for summary updates
                    await optimization_service.send_optimized_message(
                        websocket,
                        actual_client_id,
                        asdict(summary_message),
                        use_delta=True
                    )
                    
                    await asyncio.sleep(20)  # 30 second total interval
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error sending system metrics: {e}")
                    await asyncio.sleep(5)
        
        # Start system metrics task
        metrics_task = asyncio.create_task(send_system_metrics())
        
        # Listen for client messages
        try:
            while True:
                data = await websocket.receive_text()
                
                try:
                    # Check if it's a binary message with optimization
                    if data.startswith('WSOP'):
                        # This would be a binary optimized message
                        # For now, we'll handle text messages only
                        logger.debug(f"Received potential binary message from {actual_client_id}")
                        continue
                    
                    message = WebSocketMessage.from_json(data, actual_client_id)
                    logger.debug(f"Performance monitoring message from {actual_client_id}: {message.type}")
                    
                    # Handle optimization control messages
                    if message.type == MessageType.SYSTEM_STATUS and message.data:
                        action = message.data.get('action')
                        
                        if action == 'set_optimization_level':
                            level_str = message.data.get('level', 'delta')
                            try:
                                level = MessageOptimizationLevel(level_str)
                                optimization_service.set_client_optimization_level(actual_client_id, level)
                                
                                # Confirm change
                                response = WebSocketMessage(
                                    type=MessageType.SYSTEM_STATUS,
                                    timestamp=datetime.now().isoformat(),
                                    client_id=actual_client_id,
                                    data={
                                        "action": "optimization_level_updated",
                                        "level": level.value,
                                        "message": f"Performance monitoring optimization set to {level.value}"
                                    }
                                )
                                await optimization_service.send_optimized_message(
                                    websocket,
                                    actual_client_id,
                                    asdict(response)
                                )
                                continue
                            except ValueError:
                                logger.warning(f"Invalid optimization level: {level_str}")
                        
                        elif action == 'get_optimization_stats':
                            # Send optimization statistics
                            stats = optimization_service.get_performance_summary()
                            response = WebSocketMessage(
                                type=MessageType.PERFORMANCE_UPDATE,
                                timestamp=datetime.now().isoformat(),
                                client_id=actual_client_id,
                                data={
                                    "type": "optimization_stats",
                                    "stats": stats
                                }
                            )
                            await optimization_service.send_optimized_message(
                                websocket,
                                actual_client_id,
                                asdict(response)
                            )
                            continue
                    
                    # Handle specific performance monitoring commands
                    if message.type == MessageType.SUBSCRIBE and message.data:
                        subscription_type = message.data.get("subscription_type")
                        
                        if subscription_type == "function_metrics":
                            function_name = message.data.get("function_name")
                            if function_name:
                                # Send current metrics for the function
                                if function_name in profiler.metrics_storage:
                                    metrics = profiler.metrics_storage[function_name][-10:]  # Last 10 metrics
                                    response = WebSocketMessage(
                                        type=MessageType.PERFORMANCE_UPDATE,
                                        timestamp=datetime.now().isoformat(),
                                        client_id=actual_client_id,
                                        data={
                                            "type": "function_metrics",
                                            "function_name": function_name,
                                            "metrics": [m.to_dict() for m in metrics]
                                        }
                                    )
                                    await optimization_service.send_optimized_message(
                                        websocket,
                                        actual_client_id,
                                        asdict(response)
                                    )
                        
                        elif subscription_type == "alerts":
                            # Send recent alerts
                            recent_alerts = [
                                alert.to_dict() for alert in profiler.alerts
                                if alert.timestamp > datetime.now() - timedelta(hours=24)
                            ]
                            response = WebSocketMessage(
                                type=MessageType.PERFORMANCE_UPDATE,
                                timestamp=datetime.now().isoformat(),
                                client_id=actual_client_id,
                                data={
                                    "type": "recent_alerts",
                                    "alerts": recent_alerts
                                }
                            )
                            await optimization_service.send_optimized_message(
                                websocket,
                                actual_client_id,
                                asdict(response)
                            )
                    
                    elif message.type == MessageType.PING:
                        # Respond to ping
                        pong_message = WebSocketMessage(
                            type=MessageType.PONG,
                            timestamp=datetime.now().isoformat(),
                            client_id=actual_client_id
                        )
                        await optimization_service.send_optimized_message(
                            websocket,
                            actual_client_id,
                            asdict(pong_message)
                        )
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error from performance client {actual_client_id}: {e}")
                    error_message = WebSocketMessage(
                        type=MessageType.ERROR,
                        timestamp=datetime.now().isoformat(),
                        client_id=actual_client_id,
                        error="Invalid JSON format"
                    )
                    await optimization_service.send_optimized_message(
                        websocket,
                        actual_client_id,
                        asdict(error_message)
                    )
                except Exception as e:
                    logger.error(f"Error processing performance monitoring message: {e}")
                    await manager.error_handler.handle_websocket_error(actual_client_id, e)
        
        finally:
            # Clean up
            metrics_task.cancel()
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass
            
            # Unsubscribe from profiler
            try:
                profiler.unsubscribe_websocket(performance_callback)
            except Exception as e:
                logger.error(f"Error unsubscribing from profiler: {e}")
        
    except WebSocketDisconnect:
        logger.info(f"Performance monitoring client {actual_client_id} disconnected")
        await manager.disconnect(actual_client_id, "performance_client_disconnect")
        optimization_service.cleanup_client(actual_client_id)
    except Exception as e:
        logger.error(f"Error in performance monitoring WebSocket for {actual_client_id}: {e}")
        await manager.error_handler.handle_websocket_error(actual_client_id, e)
        optimization_service.cleanup_client(actual_client_id)


def get_connection_manager() -> EnhancedConnectionManager:
    """Get the global connection manager instance."""
    return manager 