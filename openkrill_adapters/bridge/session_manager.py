"""Session manager for persistent Claude Code sessions on the bridge."""

import asyncio
import json
import logging
import os
import subprocess
import uuid

logger = logging.getLogger(__name__)


class SessionProcess:
    """Tracks a running Claude Code process for a session."""

    def __init__(self, session_id: str, proc: asyncio.subprocess.Process):
        self.session_id = session_id
        self.proc = proc


class SessionManager:
    """Manages Claude Code sessions -- create, send, interrupt."""

    def __init__(
        self,
        command: str = "claude",
        initial_session_id: str | None = None,
        agent_id: str | None = None,
    ):
        self._command = command
        self._sessions: dict[str, str] = {}  # agent_session_key -> claude_session_id
        self._active_procs: dict[str, asyncio.subprocess.Process] = {}  # session_id -> running proc

        # Pre-populate with an existing session (e.g. user's local Claude Code session)
        if initial_session_id and agent_id:
            self._sessions[agent_id] = initial_session_id
            logger.info(
                "Pre-loaded session %s for agent %s",
                initial_session_id[:8],
                agent_id[:8],
            )

    def get_or_create_session_id(self, agent_id: str) -> tuple[str, bool]:
        """Get existing session_id or create a new one. Returns (session_id, is_new)."""
        if agent_id in self._sessions:
            return self._sessions[agent_id], False
        session_id = str(uuid.uuid4())
        self._sessions[agent_id] = session_id
        return session_id, True

    async def send_message(self, agent_id: str, prompt: str):
        """Send a message to a Claude Code session, yielding stream-json events.

        Async generator that yields parsed JSON events from Claude Code's
        stream-json output.
        """
        session_id, is_new = self.get_or_create_session_id(agent_id)

        # Build command
        cmd = [self._command, "-p", "--verbose", "--output-format", "stream-json"]
        if is_new:
            cmd.extend(["--session-id", session_id])
        else:
            cmd.extend(["--resume", session_id])
        cmd.extend(["--", prompt])

        logger.info(
            "Session %s: running %s %s",
            session_id[:8],
            self._command,
            "new" if is_new else "resume",
        )

        # Build a clean env — remove CLAUDECODE to avoid nested-session detection
        clean_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=clean_env,
        )
        self._active_procs[session_id] = proc

        try:
            # Read stdout line by line (stream-json is newline-delimited JSON)
            while proc.stdout:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=300)
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                try:
                    event = json.loads(line_str)
                    yield event
                except json.JSONDecodeError:
                    logger.debug("Non-JSON line from claude: %s", line_str[:200])

            await proc.wait()
            if proc.returncode != 0:
                stderr_bytes = await proc.stderr.read() if proc.stderr else b""
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                logger.error(
                    "Session %s exited with code %d: %s",
                    session_id[:8],
                    proc.returncode,
                    stderr_text[:500],
                )
                yield {
                    "type": "error",
                    "error": f"Claude exited with code {proc.returncode}: {stderr_text[:500]}",
                }
        except TimeoutError:
            proc.kill()
            yield {"type": "error", "error": "Session timed out (300s)"}
        finally:
            self._active_procs.pop(session_id, None)

    async def interrupt(self, agent_id: str) -> bool:
        """Send SIGINT to the active Claude Code process for this session."""
        session_id = self._sessions.get(agent_id)
        if not session_id:
            return False
        proc = self._active_procs.get(session_id)
        if not proc:
            return False
        try:
            proc.send_signal(2)  # SIGINT
            return True
        except Exception:
            return False

    def destroy_session(self, agent_id: str) -> bool:
        """Remove session tracking (Claude Code's internal state persists on disk)."""
        session_id = self._sessions.pop(agent_id, None)
        if session_id:
            proc = self._active_procs.pop(session_id, None)
            if proc:
                proc.kill()
            return True
        return False

    def get_session_id(self, agent_id: str) -> str | None:
        return self._sessions.get(agent_id)
