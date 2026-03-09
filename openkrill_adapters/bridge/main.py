"""Bridge daemon — runs on developer's local machine.

Connects to the OpenKrill server via WebSocket, receives requests,
spawns CLI subprocesses, and streams output back.

Usage:
    python -m openkrill_adapters.bridge.main \
        --server ws://localhost:8000/ws/bridge \
        --token <jwt-token> \
        --agent-id <agent-uuid> \
        --command claude --args "-p"

Protocol:
    1. Connect to server WebSocket
    2. Send auth: {"type": "auth", "token": "...", "agent_id": "..."}
    3. Receive requests: {"type": "bridge.request", "request_id": "...", "messages": [...]}
    4. Spawn subprocess with last user message as input
    5. Send response: {"type": "bridge.response", "request_id": "...", "content": "..."}
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess

import websockets

from openkrill_adapters.bridge.session_manager import SessionManager

logger = logging.getLogger(__name__)


async def handle_request(
    ws: websockets.WebSocketClientProtocol,
    data: dict,
    command: str,
    args: str,
) -> None:
    """Handle a bridge request by spawning a CLI subprocess."""
    request_id = data.get("request_id", "")
    messages = data.get("messages", [])

    # Extract system context and user message
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.error",
                    "request_id": request_id,
                    "error": "No user message found",
                }
            )
        )
        return

    prompt = user_messages[-1].get("content", "")
    # Prepend system context (group chat info, etc.) to the prompt
    if system_parts:
        system_context = "\n\n".join(system_parts)
        prompt = f"[System Context]\n{system_context}\n\n[User Message]\n{prompt}"

    # Build command — use "--" to prevent prompt being parsed as flags
    cmd_parts = [command] + args.split() + ["--", prompt]
    logger.info("Running: %s ...", command)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            logger.error("CLI error (exit %d): %s", proc.returncode, error_msg)
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.error",
                        "request_id": request_id,
                        "error": f"CLI exited with code {proc.returncode}: {error_msg[:500]}",
                    }
                )
            )
            return

        content = stdout.decode("utf-8", errors="replace").strip()
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.response",
                    "request_id": request_id,
                    "content": content,
                    "content_type": "text",
                }
            )
        )
        logger.info("Response sent (%d chars)", len(content))

    except TimeoutError:
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.error",
                    "request_id": request_id,
                    "error": "CLI subprocess timed out (300s)",
                }
            )
        )
    except Exception as e:
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.error",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
        )


async def handle_session_send(
    ws: websockets.WebSocketClientProtocol,
    data: dict,
    session_manager: SessionManager,
) -> None:
    """Handle a session send request — stream Claude Code output back."""
    request_id = data.get("request_id", "")
    agent_id = data.get("agent_id", "")
    prompt = data.get("prompt", "")
    messages = data.get("messages", [])

    # Support both direct prompt and messages-based input
    if not prompt and messages:
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") == "user"]
        if user_messages:
            prompt = user_messages[-1].get("content", "")
            # Prepend system context (group chat info, etc.) to the prompt
            if system_parts:
                system_context = "\n\n".join(system_parts)
                prompt = f"[System Context]\n{system_context}\n\n[User Message]\n{prompt}"

    if not prompt:
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.error",
                    "request_id": request_id,
                    "error": "No prompt provided",
                }
            )
        )
        return

    try:
        async for event in session_manager.send_message(agent_id, prompt):
            event_type = event.get("type", "")

            if event_type == "error":
                await ws.send(
                    json.dumps(
                        {
                            "type": "bridge.error",
                            "request_id": request_id,
                            "error": event.get("error", "Unknown error"),
                        }
                    )
                )
                return

            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type", "")

                if delta_type == "text_delta":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "bridge.stream",
                                "request_id": request_id,
                                "chunk_type": "text",
                                "content": delta.get("text", ""),
                            }
                        )
                    )
                elif delta_type == "thinking_delta":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "bridge.stream",
                                "request_id": request_id,
                                "chunk_type": "thinking",
                                "content": delta.get("thinking", ""),
                            }
                        )
                    )

        # Stream complete
        session_id = session_manager.get_session_id(agent_id)
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.stream.end",
                    "request_id": request_id,
                    "session_id": session_id or "",
                }
            )
        )
        logger.info("Session stream complete for request %s", request_id)

    except Exception as e:
        logger.exception("Error in session send for request %s", request_id)
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.error",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
        )


async def handle_session_interrupt(
    ws: websockets.WebSocketClientProtocol,
    data: dict,
    session_manager: SessionManager,
) -> None:
    """Handle a session interrupt request."""
    request_id = data.get("request_id", "")
    agent_id = data.get("agent_id", "")

    interrupted = await session_manager.interrupt(agent_id)
    await ws.send(
        json.dumps(
            {
                "type": "bridge.session.interrupted",
                "request_id": request_id,
                "interrupted": interrupted,
            }
        )
    )
    logger.info("Session interrupt for agent %s: %s", agent_id, interrupted)


async def handle_session_query(
    ws: websockets.WebSocketClientProtocol,
    data: dict,
    agent_id: str,
    session_manager: SessionManager | None,
) -> None:
    """Handle a session status query — report current session info."""
    request_id = data.get("request_id", "")

    session = None
    if session_manager:
        sid = session_manager.get_session_id(agent_id)
        has_active_proc = bool(sid and sid in session_manager._active_procs)
        session = {
            "session_id": sid or "",
            "active": sid is not None,
            "processing": has_active_proc,
        }

    await ws.send(
        json.dumps(
            {
                "type": "bridge.session.info",
                "request_id": request_id,
                "session": session,
            }
        )
    )


async def run_bridge(
    server_url: str,
    token: str,
    agent_id: str,
    command: str,
    args: str,
    session_mode: bool = False,
) -> None:
    """Main bridge loop — connect, authenticate, handle requests."""
    session_manager: SessionManager | None = None
    if session_mode:
        session_manager = SessionManager(command=command)
        logger.info("Session mode enabled")

    while True:
        try:
            logger.info("Connecting to %s ...", server_url)
            async with websockets.connect(server_url) as ws:
                # Authenticate
                await ws.send(
                    json.dumps(
                        {
                            "type": "auth",
                            "token": token,
                            "agent_id": agent_id,
                        }
                    )
                )
                logger.info("Bridge connected and authenticated (agent=%s)", agent_id)

                # Heartbeat task
                async def heartbeat() -> None:
                    while True:
                        await asyncio.sleep(30)
                        await ws.send(json.dumps({"type": "ping"}))

                heartbeat_task = asyncio.create_task(heartbeat())

                try:
                    async for raw in ws:
                        data = json.loads(raw)
                        msg_type = data.get("type", "")

                        if msg_type == "bridge.request":
                            # Handle in background to allow concurrent requests
                            asyncio.create_task(handle_request(ws, data, command, args))
                        elif msg_type == "bridge.session.send" and session_manager:
                            asyncio.create_task(handle_session_send(ws, data, session_manager))
                        elif msg_type == "bridge.session.interrupt" and session_manager:
                            asyncio.create_task(handle_session_interrupt(ws, data, session_manager))
                        elif msg_type == "bridge.session.query":
                            asyncio.create_task(
                                handle_session_query(ws, data, agent_id, session_manager)
                            )
                        elif msg_type == "pong":
                            pass
                        else:
                            logger.debug("Unknown message type: %s", msg_type)
                finally:
                    heartbeat_task.cancel()

        except Exception as e:
            logger.warning("Connection lost: %s. Reconnecting in 5s...", e)
            await asyncio.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenKrill Bridge Daemon")
    parser.add_argument("--server", required=True, help="WebSocket URL (ws://host:port/ws/bridge)")
    parser.add_argument(
        "--token",
        default=os.environ.get("OPENKRILL_BRIDGE_TOKEN", ""),
        help="JWT auth token (or set OPENKRILL_BRIDGE_TOKEN env var)",
    )
    parser.add_argument("--agent-id", required=True, help="Agent UUID to serve")
    parser.add_argument("--command", default="claude", help="CLI command (default: claude)")
    parser.add_argument(
        "--args",
        default="-p",
        help='CLI args, use = syntax for dash args: --args="-p" (default: -p)',
    )
    parser.add_argument(
        "--session-mode",
        action="store_true",
        help="Enable session management and streaming",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.token:
        parser.error("--token is required (or set OPENKRILL_BRIDGE_TOKEN env var)")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(
        run_bridge(
            server_url=args.server,
            token=args.token,
            agent_id=args.agent_id,
            command=args.command,
            args=args.args,
            session_mode=args.session_mode,
        )
    )


if __name__ == "__main__":
    main()
