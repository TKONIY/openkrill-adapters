"""Agent Bridge — lightweight daemon for proxying local CLI tools to OpenKrill server.

The Bridge runs on a user's work machine, connects to the central server via WebSocket,
and spawns CLI subprocesses (Claude Code, Codex, etc.) to handle requests.
"""
