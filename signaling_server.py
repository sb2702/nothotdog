#!/usr/bin/env python3
"""
Simple WebRTC signaling server using WebSockets
Handles SDP offer/answer exchange between browser and Python client
"""

import asyncio
import websockets
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalingServer:
    def __init__(self):
        self.clients = {}  # Store connected clients
        
    async def register_client(self, websocket, client_type):
        """Register a new client (browser or python)"""
        client_id = f"{client_type}_{len(self.clients)}"
        self.clients[client_id] = {
            'websocket': websocket,
            'type': client_type,
            'id': client_id
        }
        logger.info(f"Client registered: {client_id}")
        return client_id
    
    async def unregister_client(self, client_id):
        """Remove client from registry"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Client unregistered: {client_id}")
    
    async def forward_message(self, sender_id, message):
        """Forward signaling messages between clients"""
        sender_type = self.clients[sender_id]['type']
        
        # Forward to the other type of client
        target_type = 'python' if sender_type == 'browser' else 'browser'
        
        for client_id, client_info in self.clients.items():
            if client_info['type'] == target_type and client_id != sender_id:
                try:
                    await client_info['websocket'].send(json.dumps(message))
                    logger.info(f"Forwarded {message.get('type')} from {sender_id} to {client_id}")
                except websockets.exceptions.ConnectionClosed:
                    await self.unregister_client(client_id)
    
    async def handle_client(self, websocket):
        """Handle WebSocket connection from client"""
        client_id = None
        try:
            # Wait for registration message
            registration = await websocket.recv()
            reg_data = json.loads(registration)
            
            if reg_data.get('type') != 'register':
                await websocket.send(json.dumps({'error': 'First message must be registration'}))
                return
            
            client_type = reg_data.get('client_type')  # 'browser' or 'python'
            if client_type not in ['browser', 'python']:
                await websocket.send(json.dumps({'error': 'client_type must be browser or python'}))
                return
            
            client_id = await self.register_client(websocket, client_type)
            
            # Send registration confirmation
            await websocket.send(json.dumps({
                'type': 'registered',
                'client_id': client_id
            }))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type in ['offer', 'answer', 'ice-candidate']:
                        # Forward WebRTC signaling messages
                        await self.forward_message(client_id, data)
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            if client_id:
                await self.unregister_client(client_id)

async def main():
    """Start the signaling server"""
    server = SignalingServer()
    
    # Start WebSocket server
    start_server = websockets.serve(
        server.handle_client,
        "localhost", 
        8765,
        ping_interval=20,
        ping_timeout=10
    )
    
    logger.info("WebRTC Signaling Server starting on ws://localhost:8765")
    
    async with start_server:
        # Keep the server running
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")