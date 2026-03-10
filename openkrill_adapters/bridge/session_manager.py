"""Session manager for persistent Claude Code sessions on the bridge."""

import asyncio
import json
import logging
import os
import pty
import uuid

logger = logging.getLogger(__name__)


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

        Claude Code requires a TTY to produce output in --print mode, so we
        use a pseudo-terminal (PTY) to run the subprocess.
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

        # Build a clean env — remove all Claude Code env vars to avoid nested-session detection
        # Claude CLI checks CLAUDECODE and CLAUDE_CODE_ENTRYPOINT
        _claude_env_keys = {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"}
        clean_env = {k: v for k, v in os.environ.items() if k not in _claude_env_keys}

        # Use a PTY because Claude Code requires a TTY to produce output.
        # We create a master/slave PTY pair. The subprocess writes to the slave
        # (which looks like a terminal), and we read from the master fd.
        master_fd, slave_fd = pty.openpty()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=slave_fd,
            stderr=asyncio.subprocess.PIPE,
            stdin=slave_fd,
            env=clean_env,
        )
        # Close slave fd in parent — the child process has it
        os.close(slave_fd)

        self._active_procs[session_id] = proc

        try:
            loop = asyncio.get_event_loop()
            buffer = b""

            while True:
                try:
                    # Read from PTY master fd asynchronously
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(None, os.read, master_fd, 65536),
                        timeout=300,
                    )
                except TimeoutError:
                    proc.kill()
                    yield {"type": "error", "error": "Session timed out (300s)"}
                    return
                except OSError:
                    # PTY closed — process exited
                    break

                if not chunk:
                    break

                buffer += chunk

                # Process complete lines
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)
                    line_str = line_bytes.decode("utf-8", errors="replace").strip()
                    # Strip ANSI escape codes and carriage returns
                    line_str = _strip_ansi(line_str)
                    if not line_str:
                        continue
                    try:
                        event = json.loads(line_str)
                        yield event
                    except json.JSONDecodeError:
                        logger.debug("Non-JSON line from claude: %s", line_str[:200])

            # Process any remaining buffer
            if buffer:
                line_str = buffer.decode("utf-8", errors="replace").strip()
                line_str = _strip_ansi(line_str)
                if line_str:
                    try:
                        event = json.loads(line_str)
                        yield event
                    except json.JSONDecodeError:
                        logger.debug("Non-JSON trailing data: %s", line_str[:200])

            await proc.wait()
            stderr_bytes = await proc.stderr.read() if proc.stderr else b""
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
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
            elif stderr_text:
                logger.warning(
                    "Session %s exited 0 but stderr: %s",
                    session_id[:8],
                    stderr_text[:500],
                )
        finally:
            os.close(master_fd)
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


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and carriage returns from text."""
    import re

    # Remove ANSI escape sequences
    ansi_re = re.compile(r"\x1b\[[^a-zA-Z]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b\[[\?]?[0-9;]*[a-zA-Z]")
    text = ansi_re.sub("", text)
    # Remove carriage returns
    text = text.replace("\r", "")
    return text.strip()
