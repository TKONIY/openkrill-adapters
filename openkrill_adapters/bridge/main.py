"""Bridge entrypoint — WebSocket client that connects to OpenKrill server.

Usage:
    openkrill-bridge start --server https://krill.example.com --token brg-xxx

Bridge lifecycle:
    1. Connect to server via WebSocket
    2. Authenticate with bridge token
    3. Listen for requests (bridge.request)
    4. Spawn CLI subprocess per request, stream output back (bridge.stream)
    5. Report completion (bridge.done) or error (bridge.error)
    6. Send heartbeat every 30s
"""
