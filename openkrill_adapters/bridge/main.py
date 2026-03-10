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

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess

import websockets

from openkrill_adapters.bridge.session_manager import SessionManager

# ---------------------------------------------------------------------------
# Constants for file system / git handlers
# ---------------------------------------------------------------------------

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".mypy_cache", ".ruff_cache"}

_MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB

_MAX_DEPTH = 5

_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".sql": "sql",
    ".graphql": "graphql",
    ".xml": "xml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".env": "dotenv",
    ".dockerfile": "dockerfile",
    ".tf": "hcl",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".vue": "vue",
    ".svelte": "svelte",
}

_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pyc", ".pyo", ".class", ".jar",
    ".bin", ".dat", ".db", ".sqlite",
}  # fmt: skip

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_path(requested: str, working_dir: str) -> str | None:
    """Resolve *requested* under *working_dir* and return the real path.

    Returns ``None`` if the resolved path escapes the working directory.
    """
    if os.path.isabs(requested):
        resolved = os.path.realpath(requested)
    else:
        resolved = os.path.realpath(os.path.join(working_dir, requested))
    wd_real = os.path.realpath(working_dir)
    if not (resolved == wd_real or resolved.startswith(wd_real + os.sep)):
        return None
    return resolved


def _is_binary(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in _BINARY_EXTENSIONS


def _detect_language(path: str) -> str:
    _, ext = os.path.splitext(path)
    if ext.lower() in _LANG_MAP:
        return _LANG_MAP[ext.lower()]
    name = os.path.basename(path).lower()
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    return "text"


async def _run_cmd(cmd: list[str], cwd: str, timeout: float = 10) -> tuple[int, str, str]:
    """Run a command asynchronously and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except FileNotFoundError:
        return (127, "", f"Command not found: {cmd[0]}")
    except TimeoutError:
        return (1, "", "Command timed out")


# ---------------------------------------------------------------------------
# File system handlers
# ---------------------------------------------------------------------------


async def _git_tracked_files(working_dir: str) -> set[str] | None:
    """Return the set of git-tracked + untracked-but-not-ignored files.

    Returns None if *working_dir* is not a git repo.
    """
    rc1, tracked_out, _ = await _run_cmd(["git", "ls-files"], working_dir)
    if rc1 != 0:
        return None
    rc2, untracked_out, _ = await _run_cmd(
        ["git", "ls-files", "--others", "--exclude-standard"], working_dir
    )
    files: set[str] = set()
    for line in tracked_out.splitlines():
        if line.strip():
            files.add(line.strip())
    if rc2 == 0:
        for line in untracked_out.splitlines():
            if line.strip():
                files.add(line.strip())
    return files


def _build_tree_os(
    root: str,
    current_depth: int,
    max_depth: int,
) -> list[dict]:
    """Walk the filesystem, skipping ignored directories."""
    nodes: list[dict] = []
    try:
        entries = sorted(os.scandir(root), key=lambda e: (not e.is_dir(), e.name))
    except PermissionError:
        return nodes

    for entry in entries:
        if entry.name in _SKIP_DIRS:
            continue
        if entry.is_dir(follow_symlinks=False):
            children: list[dict] = []
            if current_depth < max_depth:
                children = _build_tree_os(entry.path, current_depth + 1, max_depth)
            nodes.append(
                {
                    "name": entry.name,
                    "type": "directory",
                    "path": entry.path,
                    "children": children,
                }
            )
        elif entry.is_file(follow_symlinks=False):
            node: dict = {
                "name": entry.name,
                "type": "file",
                "path": entry.path,
            }
            if _is_binary(entry.path):
                node["binary"] = True
            nodes.append(node)
    return nodes


def _build_tree_from_git(
    file_list: set[str],
    working_dir: str,
    base_rel: str,
    max_depth: int,
) -> list[dict]:
    """Build a tree from a flat list of git-tracked relative paths."""
    # Filter to only paths under base_rel
    prefix = base_rel.rstrip("/") + "/" if base_rel and base_rel != "." else ""
    relevant: list[str] = []
    for f in file_list:
        if prefix:
            if f.startswith(prefix):
                relevant.append(f[len(prefix) :])
        else:
            relevant.append(f)

    # Build nested dict first
    tree_dict: dict = {}
    for rel in relevant:
        parts = rel.split("/")
        # Skip if exceeds max depth
        if len(parts) > max_depth + 1:
            # Still add parent dirs up to max_depth
            parts = parts[: max_depth + 1]
        node = tree_dict
        for i, part in enumerate(parts):
            if part not in node:
                node[part] = {}
            if i < len(parts) - 1:
                node = node[part]

    # Convert to list of nodes
    base_path = os.path.join(working_dir, base_rel) if base_rel and base_rel != "." else working_dir

    def _dict_to_nodes(d: dict, parent_path: str, depth: int) -> list[dict]:
        nodes: list[dict] = []
        for name in sorted(d.keys(), key=lambda n: (not bool(d[n]), n)):
            full = os.path.join(parent_path, name)
            children_dict = d[name]
            if children_dict:
                # It's a directory (has children)
                children: list[dict] = []
                if depth < max_depth:
                    children = _dict_to_nodes(children_dict, full, depth + 1)
                nodes.append(
                    {
                        "name": name,
                        "type": "directory",
                        "path": full,
                        "children": children,
                    }
                )
            else:
                # Could be a file or an empty dir — check filesystem
                node_item: dict = {
                    "name": name,
                    "type": "file",
                    "path": full,
                }
                if _is_binary(full):
                    node_item["binary"] = True
                nodes.append(node_item)
        return nodes

    return _dict_to_nodes(tree_dict, base_path, 1)


async def handle_files_list(
    ws: websockets.WebSocketClientProtocol,
    data: dict,
    working_dir: str,
) -> None:
    """Handle bridge.files.list — return a file tree."""
    request_id = data.get("request_id", "")
    req_path = data.get("path", ".")
    depth = min(data.get("depth", 3), _MAX_DEPTH)

    try:
        resolved = _validate_path(req_path, working_dir)
        if resolved is None:
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.files.tree",
                        "request_id": request_id,
                        "error": "Path outside working directory",
                    }
                )
            )
            return

        if not os.path.isdir(resolved):
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.files.tree",
                        "request_id": request_id,
                        "error": f"Not a directory: {req_path}",
                    }
                )
            )
            return

        # Try git-based listing first
        git_files = await _git_tracked_files(working_dir)
        if git_files is not None:
            wd_real = os.path.realpath(working_dir)
            resolved_real = os.path.realpath(resolved)
            base_rel = os.path.relpath(resolved_real, wd_real)
            if base_rel == ".":
                base_rel = ""
            tree = _build_tree_from_git(git_files, wd_real, base_rel, depth)
        else:
            tree = _build_tree_os(resolved, 1, depth)

        await ws.send(
            json.dumps(
                {
                    "type": "bridge.files.tree",
                    "request_id": request_id,
                    "tree": tree,
                }
            )
        )
        logger.info("Files tree sent (%d top-level entries)", len(tree))

    except Exception as e:
        logger.exception("Error in handle_files_list")
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.files.tree",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
        )


async def handle_files_read(
    ws: websockets.WebSocketClientProtocol,
    data: dict,
    working_dir: str,
) -> None:
    """Handle bridge.files.read — return file content."""
    request_id = data.get("request_id", "")
    req_path = data.get("path", "")

    try:
        if not req_path:
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.files.content",
                        "request_id": request_id,
                        "error": "No path specified",
                    }
                )
            )
            return

        resolved = _validate_path(req_path, working_dir)
        if resolved is None:
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.files.content",
                        "request_id": request_id,
                        "error": "Path outside working directory",
                    }
                )
            )
            return

        if not os.path.isfile(resolved):
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.files.content",
                        "request_id": request_id,
                        "error": f"File not found: {req_path}",
                    }
                )
            )
            return

        file_size = os.path.getsize(resolved)
        if file_size > _MAX_FILE_SIZE:
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.files.content",
                        "request_id": request_id,
                        "path": resolved,
                        "error": f"File too large ({file_size} bytes, max {_MAX_FILE_SIZE})",
                    }
                )
            )
            return

        if _is_binary(resolved):
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.files.content",
                        "request_id": request_id,
                        "path": resolved,
                        "binary": True,
                        "size": file_size,
                    }
                )
            )
            return

        # Read text file
        try:
            with open(resolved, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.files.content",
                        "request_id": request_id,
                        "path": resolved,
                        "error": f"Cannot read file: {e}",
                    }
                )
            )
            return

        language = _detect_language(resolved)
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.files.content",
                    "request_id": request_id,
                    "path": resolved,
                    "content": content,
                    "language": language,
                }
            )
        )
        logger.info("File content sent: %s (%d chars)", resolved, len(content))

    except Exception as e:
        logger.exception("Error in handle_files_read")
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.files.content",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
        )


# ---------------------------------------------------------------------------
# Git handlers
# ---------------------------------------------------------------------------


async def handle_git_log(
    ws: websockets.WebSocketClientProtocol,
    data: dict,
    working_dir: str,
) -> None:
    """Handle bridge.git.log — return commit history."""
    request_id = data.get("request_id", "")
    limit = min(data.get("limit", 50), 500)

    try:
        rc, stdout, stderr = await _run_cmd(
            [
                "git",
                "log",
                "--format=%H%x00%h%x00%s%x00%an%x00%aI%x00%P%x00%D",
                f"-n{limit}",
            ],
            working_dir,
        )

        if rc != 0:
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.git.log.result",
                        "request_id": request_id,
                        "error": (
                            "Not a git repository"
                            if "not a git" in stderr.lower()
                            else stderr.strip()
                        ),
                    }
                )
            )
            return

        commits = []
        for line in stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("\x00")
            if len(parts) < 7:
                continue
            full_hash, short_hash, message, author, date, parents_str, refs_str = parts
            parents = parents_str.split() if parents_str.strip() else []
            refs = [r.strip() for r in refs_str.split(",") if r.strip()] if refs_str.strip() else []
            commits.append(
                {
                    "hash": full_hash,
                    "short_hash": short_hash,
                    "message": message,
                    "author": author,
                    "date": date,
                    "parents": parents,
                    "refs": refs,
                }
            )

        await ws.send(
            json.dumps(
                {
                    "type": "bridge.git.log.result",
                    "request_id": request_id,
                    "commits": commits,
                }
            )
        )
        logger.info("Git log sent (%d commits)", len(commits))

    except Exception as e:
        logger.exception("Error in handle_git_log")
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.git.log.result",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
        )


def _parse_diff_output(diff_text: str, numstat_text: str) -> list[dict]:
    """Parse unified diff output + numstat into per-file entries."""
    # Parse numstat for additions/deletions
    stats: dict[str, tuple[int, int]] = {}
    for line in numstat_text.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            adds = int(parts[0]) if parts[0] != "-" else 0
            dels = int(parts[1]) if parts[1] != "-" else 0
            path = parts[2]
            # Handle renames: "old => new" or "{old => new}/path"
            if " => " in path:
                path = path.split(" => ")[-1].rstrip("}")
                if "{" in parts[2]:
                    # e.g. src/{old.py => new.py}
                    prefix = parts[2].split("{")[0]
                    path = prefix + path
            stats[path] = (adds, dels)

    # Split diff into per-file chunks
    files: list[dict] = []
    current_diff_lines: list[str] = []
    current_path = ""
    current_status = "modified"

    def _flush() -> None:
        if current_path:
            adds, dels = stats.get(current_path, (0, 0))
            files.append(
                {
                    "path": current_path,
                    "status": current_status,
                    "additions": adds,
                    "deletions": dels,
                    "diff": "\n".join(current_diff_lines),
                }
            )

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            _flush()
            current_diff_lines = [line]
            # Extract path: diff --git a/path b/path
            parts = line.split(" b/", 1)
            current_path = parts[1] if len(parts) > 1 else ""
            current_status = "modified"
        elif line.startswith("new file mode"):
            current_status = "added"
            current_diff_lines.append(line)
        elif line.startswith("deleted file mode"):
            current_status = "deleted"
            current_diff_lines.append(line)
        elif line.startswith("rename from") or line.startswith("rename to"):
            current_status = "renamed"
            current_diff_lines.append(line)
        else:
            current_diff_lines.append(line)

    _flush()
    return files


async def handle_git_diff(
    ws: websockets.WebSocketClientProtocol,
    data: dict,
    working_dir: str,
) -> None:
    """Handle bridge.git.diff — return diff information."""
    request_id = data.get("request_id", "")
    target = data.get("target", "uncommitted")
    commit_hash = data.get("hash", "")
    from_hash = data.get("from_hash", "")
    to_hash = data.get("to_hash", "")

    try:
        if target == "uncommitted":
            # Check if there are any commits
            rc_check, _, _ = await _run_cmd(["git", "rev-parse", "HEAD"], working_dir)
            if rc_check != 0:
                diff_args = ["git", "diff"]
                numstat_args = ["git", "diff", "--numstat"]
            else:
                diff_args = ["git", "diff", "HEAD"]
                numstat_args = ["git", "diff", "HEAD", "--numstat"]
        elif target == "commit":
            if not commit_hash:
                await ws.send(
                    json.dumps(
                        {
                            "type": "bridge.git.diff.result",
                            "request_id": request_id,
                            "error": "No commit hash specified",
                        }
                    )
                )
                return
            diff_args = ["git", "diff", f"{commit_hash}~1", commit_hash]
            numstat_args = ["git", "diff", f"{commit_hash}~1", commit_hash, "--numstat"]
        elif target == "range":
            if not from_hash or not to_hash:
                await ws.send(
                    json.dumps(
                        {
                            "type": "bridge.git.diff.result",
                            "request_id": request_id,
                            "error": "from_hash and to_hash required for range diff",
                        }
                    )
                )
                return
            diff_args = ["git", "diff", from_hash, to_hash]
            numstat_args = ["git", "diff", from_hash, to_hash, "--numstat"]
        else:
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.git.diff.result",
                        "request_id": request_id,
                        "error": f"Unknown diff target: {target}",
                    }
                )
            )
            return

        rc_diff, diff_out, diff_err = await _run_cmd(diff_args, working_dir, timeout=30)
        rc_stat, stat_out, _ = await _run_cmd(numstat_args, working_dir, timeout=30)

        if rc_diff != 0:
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.git.diff.result",
                        "request_id": request_id,
                        "error": diff_err.strip() or "git diff failed",
                    }
                )
            )
            return

        files = _parse_diff_output(diff_out, stat_out if rc_stat == 0 else "")

        await ws.send(
            json.dumps(
                {
                    "type": "bridge.git.diff.result",
                    "request_id": request_id,
                    "files": files,
                }
            )
        )
        logger.info("Git diff sent (%d files)", len(files))

    except Exception as e:
        logger.exception("Error in handle_git_diff")
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.git.diff.result",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
        )


async def handle_git_branches(
    ws: websockets.WebSocketClientProtocol,
    data: dict,
    working_dir: str,
) -> None:
    """Handle bridge.git.branches — return branch list."""
    request_id = data.get("request_id", "")

    try:
        rc_cur, current_out, _ = await _run_cmd(["git", "branch", "--show-current"], working_dir)
        rc_all, branches_out, stderr = await _run_cmd(
            ["git", "branch", "-a", "--format=%(refname:short)"], working_dir
        )

        if rc_all != 0:
            await ws.send(
                json.dumps(
                    {
                        "type": "bridge.git.branches.result",
                        "request_id": request_id,
                        "error": (
                            "Not a git repository"
                            if "not a git" in stderr.lower()
                            else stderr.strip()
                        ),
                    }
                )
            )
            return

        current = current_out.strip() if rc_cur == 0 else ""
        branches = [b.strip() for b in branches_out.strip().splitlines() if b.strip()]

        await ws.send(
            json.dumps(
                {
                    "type": "bridge.git.branches.result",
                    "request_id": request_id,
                    "current": current,
                    "branches": branches,
                }
            )
        )
        logger.info("Git branches sent (%d branches, current=%s)", len(branches), current)

    except Exception as e:
        logger.exception("Error in handle_git_branches")
        await ws.send(
            json.dumps(
                {
                    "type": "bridge.git.branches.result",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
        )


# ---------------------------------------------------------------------------
# CLI request handler
# ---------------------------------------------------------------------------


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

            # Claude Code stream-json format: "assistant" event with full message
            if event_type == "assistant":
                message = event.get("message", {})
                content_blocks = message.get("content", [])
                for block in content_blocks:
                    block_type = block.get("type", "")
                    if block_type == "text" and block.get("text"):
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "bridge.stream",
                                    "request_id": request_id,
                                    "chunk_type": "text",
                                    "content": block["text"],
                                }
                            )
                        )
                    elif block_type == "thinking" and block.get("thinking"):
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "bridge.stream",
                                    "request_id": request_id,
                                    "chunk_type": "thinking",
                                    "content": block["thinking"],
                                }
                            )
                        )
                    elif block_type == "tool_use":
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "bridge.stream",
                                    "request_id": request_id,
                                    "chunk_type": "tool_use",
                                    "content": "",
                                    "metadata": {
                                        "tool_name": block.get("name", ""),
                                        "tool_use_id": block.get("id", ""),
                                        "tool_input": block.get("input", {}),
                                    },
                                }
                            )
                        )

            # Claude Code "result" event with tool_results and usage
            elif event_type == "result":
                tool_results = event.get("tool_results", [])
                for tr in tool_results:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "bridge.stream",
                                "request_id": request_id,
                                "chunk_type": "tool_result",
                                "content": "",
                                "metadata": {
                                    "tool_use_id": tr.get("tool_use_id", ""),
                                    "tool_name": tr.get("name", ""),
                                    "result": tr.get("content", ""),
                                    "is_error": tr.get("is_error", False),
                                },
                            }
                        )
                    )
                # Extract usage from result event
                result_usage = event.get("usage", {})
                if result_usage:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "bridge.stream",
                                "request_id": request_id,
                                "chunk_type": "usage",
                                "content": "",
                                "metadata": {
                                    "input_tokens": result_usage.get("input_tokens", 0),
                                    "output_tokens": result_usage.get("output_tokens", 0),
                                    "model": event.get("model", ""),
                                },
                            }
                        )
                    )

            # Anthropic API format (content_block_delta) — keep for compatibility
            elif event_type == "content_block_delta":
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
    initial_session_id: str = "",
    working_dir: str = "",
) -> None:
    """Main bridge loop — connect, authenticate, handle requests."""
    if not working_dir:
        working_dir = os.getcwd()
    working_dir = os.path.realpath(working_dir)

    session_manager: SessionManager | None = None
    if session_mode:
        session_manager = SessionManager(
            command=command,
            initial_session_id=initial_session_id or None,
            agent_id=agent_id if initial_session_id else None,
        )
        if initial_session_id:
            logger.info("Session mode enabled (resuming session %s)", initial_session_id[:8])
        else:
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
                        # File system protocol
                        elif msg_type == "bridge.files.list":
                            asyncio.create_task(handle_files_list(ws, data, working_dir))
                        elif msg_type == "bridge.files.read":
                            asyncio.create_task(handle_files_read(ws, data, working_dir))
                        # Git protocol
                        elif msg_type == "bridge.git.log":
                            asyncio.create_task(handle_git_log(ws, data, working_dir))
                        elif msg_type == "bridge.git.diff":
                            asyncio.create_task(handle_git_diff(ws, data, working_dir))
                        elif msg_type == "bridge.git.branches":
                            asyncio.create_task(handle_git_branches(ws, data, working_dir))
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
    parser.add_argument(
        "--session-id",
        default="",
        help="Existing Claude Code session ID to resume (implies --session-mode)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.token:
        parser.error("--token is required (or set OPENKRILL_BRIDGE_TOKEN env var)")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --session-id implies --session-mode
    session_mode = args.session_mode or bool(args.session_id)

    asyncio.run(
        run_bridge(
            server_url=args.server,
            token=args.token,
            agent_id=args.agent_id,
            command=args.command,
            args=args.args,
            session_mode=session_mode,
            initial_session_id=args.session_id,
        )
    )


if __name__ == "__main__":
    main()
