
import asyncio
import websockets
import json
import requests
import uuid

API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

async def test_websocket():
    # 1. Create Session
    print("Creating session...")
    try:
        resp = requests.post(f"{API_URL}/api/sessions")
        resp.raise_for_status()
        session_data = resp.json()
        session_id = session_data["session_id"]
        print(f"Session created: {session_id}")
    except Exception as e:
        print(f"Failed to create session: {e}")
        return

    # 2. Connect to WebSocket
    uri = f"{WS_URL}/{session_id}"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for connection message (if any)
            # In ConnectionManager, it sends "connected" message immediately.
            # Let's see if we get it or if we just send.
            
            # Send "hi"
            msg = "hi"
            print(f"Sending: {msg}")
            await websocket.send(msg)
            
            # Receive first response
            response = await websocket.recv()
            print(f"Received: {response}")
            
            data = json.loads(response)
            if data.get("type") == "connected":
                 print("Received connected message, waiting for agent responses...")
                 
                 for _ in range(5):
                     try:
                         response = await websocket.recv()
                         print(f"Received: {response}")
                     except asyncio.TimeoutError:
                         break

            print("Test Passed!")
    except Exception as e:
        print(f"WebSocket test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
