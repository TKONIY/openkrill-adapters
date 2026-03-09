"""OpenKrill skill for Claude Code — connect current session to OpenKrill.

This is a Claude Code custom slash command (/openkrill) that starts the
bridge daemon in session mode, connecting the current Claude Code session
to an OpenKrill agent.

Usage inside Claude Code:
    /openkrill connect --agent-id <uuid>
    /openkrill connect --agent-id <uuid> --server ws://myserver:8000/ws/bridge

Setup:
    Add to your Claude Code settings (CLAUDE.md or .claude/settings.json):
    {
      "skills": {
        "openkrill": {
          "command": "python -m openkrill_adapters.bridge.skill"
        }
      }
    }

    Set environment variables:
    - OPENKRILL_BRIDGE_TOKEN: JWT token for authentication
    - OPENKRILL_SERVER: WebSocket URL (default: ws://localhost:8000/ws/bridge)
"""

import argparse
import asyncio
import logging
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect this Claude Code session to OpenKrill"
    )
    subparsers = parser.add_subparsers(dest="action")

    connect_parser = subparsers.add_parser("connect", help="Start bridge daemon in session mode")
    connect_parser.add_argument("--agent-id", required=True, help="Agent UUID in OpenKrill")
    connect_parser.add_argument(
        "--server",
        default=os.environ.get("OPENKRILL_SERVER", "ws://localhost:8000/ws/bridge"),
        help="OpenKrill WebSocket URL",
    )
    connect_parser.add_argument(
        "--token",
        default=os.environ.get("OPENKRILL_BRIDGE_TOKEN", ""),
        help="JWT auth token (or set OPENKRILL_BRIDGE_TOKEN env var)",
    )

    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        sys.exit(1)

    if args.action == "connect":
        if not args.token:
            print("Error: --token required or set OPENKRILL_BRIDGE_TOKEN env var")
            sys.exit(1)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # Import and run the bridge in session mode
        from openkrill_adapters.bridge.main import run_bridge

        print(f"Connecting to OpenKrill as agent {args.agent_id}...")
        print(f"Server: {args.server}")
        print("Session mode: enabled (persistent context)")
        print("Press Ctrl+C to disconnect.\n")

        try:
            asyncio.run(
                run_bridge(
                    server_url=args.server,
                    token=args.token,
                    agent_id=args.agent_id,
                    command="claude",
                    args="-p",
                    session_mode=True,
                )
            )
        except KeyboardInterrupt:
            print("\nDisconnected from OpenKrill.")


if __name__ == "__main__":
    main()
