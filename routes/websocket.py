"""
WebSocket routes for real-time simulation updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import uuid
import asyncio
from services.simulation_service import SimulationService

router = APIRouter(prefix="/ws", tags=["WebSocket"])

# Global simulation service instance
simulation_service = SimulationService()

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.simulation_subscribers: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from all simulation subscriptions
        for simulation_id in list(self.simulation_subscribers.keys()):
            if client_id in self.simulation_subscribers[simulation_id]:
                self.simulation_subscribers[simulation_id].remove(client_id)
                if not self.simulation_subscribers[simulation_id]:
                    del self.simulation_subscribers[simulation_id]

    def subscribe_to_simulation(self, client_id: str, simulation_id: str):
        """Subscribe a client to simulation updates."""
        if simulation_id not in self.simulation_subscribers:
            self.simulation_subscribers[simulation_id] = []
        if client_id not in self.simulation_subscribers[simulation_id]:
            self.simulation_subscribers[simulation_id].append(client_id)

    def unsubscribe_from_simulation(self, client_id: str, simulation_id: str):
        """Unsubscribe a client from simulation updates."""
        if simulation_id in self.simulation_subscribers:
            if client_id in self.simulation_subscribers[simulation_id]:
                self.simulation_subscribers[simulation_id].remove(client_id)
                if not self.simulation_subscribers[simulation_id]:
                    del self.simulation_subscribers[simulation_id]

    async def send_personal_message(self, message: str, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception:
                self.disconnect(client_id)

    async def broadcast_to_simulation(self, message: str, simulation_id: str):
        """Broadcast a message to all clients subscribed to a simulation."""
        if simulation_id in self.simulation_subscribers:
            disconnected_clients = []
            for client_id in self.simulation_subscribers[simulation_id]:
                try:
                    await self.send_personal_message(message, client_id)
                except Exception:
                    disconnected_clients.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected_clients:
                self.disconnect(client_id)

# Global connection manager
manager = ConnectionManager()

@router.websocket("/simulation")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time simulation updates.
    
    Clients can subscribe to simulation updates by sending messages with:
    {
        "action": "subscribe",
        "simulation_id": "uuid-here"
    }
    
    Or unsubscribe with:
    {
        "action": "unsubscribe", 
        "simulation_id": "uuid-here"
    }
    """
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)
    
    # Send connection confirmation
    await websocket.send_text(json.dumps({
        "type": "connection",
        "message": "Connected to simulation WebSocket",
        "client_id": client_id
    }))
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                action = message.get("action")
                simulation_id = message.get("simulation_id")
                
                if action == "subscribe" and simulation_id:
                    manager.subscribe_to_simulation(client_id, simulation_id)
                    await websocket.send_text(json.dumps({
                        "type": "subscription",
                        "message": f"Subscribed to simulation {simulation_id}",
                        "simulation_id": simulation_id
                    }))
                    
                elif action == "unsubscribe" and simulation_id:
                    manager.unsubscribe_from_simulation(client_id, simulation_id)
                    await websocket.send_text(json.dumps({
                        "type": "unsubscription",
                        "message": f"Unsubscribed from simulation {simulation_id}",
                        "simulation_id": simulation_id
                    }))
                    
                elif action == "run_simulation" and simulation_id:
                    # Start streaming simulation updates via WebSocket
                    asyncio.create_task(stream_simulation_updates(simulation_id, client_id))
                    await websocket.send_text(json.dumps({
                        "type": "simulation_started",
                        "message": f"Started streaming simulation {simulation_id}",
                        "simulation_id": simulation_id
                    }))
                    
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid action or missing simulation_id"
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)


async def stream_simulation_updates(simulation_id: str, client_id: str):
    """
    Stream simulation updates to a specific WebSocket client.
    
    Args:
        simulation_id: ID of the simulation to run
        client_id: ID of the WebSocket client to send updates to
    """
    try:
        async for progress_data in simulation_service.run_simulation_async(simulation_id):
            # Send update to the specific client
            message = json.dumps({
                "type": "simulation_update",
                "simulation_id": simulation_id,
                "data": progress_data
            })
            await manager.send_personal_message(message, client_id)
            
            # Also broadcast to any other subscribers
            await manager.broadcast_to_simulation(message, simulation_id)
            
    except Exception as e:
        error_message = json.dumps({
            "type": "simulation_error",
            "simulation_id": simulation_id,
            "error": str(e)
        })
        await manager.send_personal_message(error_message, client_id)


@router.websocket("/global")
async def global_websocket_endpoint(websocket: WebSocket):
    """
    Global WebSocket endpoint for system-wide updates.
    """
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)
    
    await websocket.send_text(json.dumps({
        "type": "connection",
        "message": "Connected to global WebSocket",
        "client_id": client_id
    }))
    
    try:
        while True:
            # Keep connection alive and handle any global messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle global commands if needed
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "message": f"Received: {message}"
                }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(client_id) 