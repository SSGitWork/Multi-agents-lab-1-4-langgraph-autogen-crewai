"""
tools/code_tools.py  –  File I/O and sandboxed code-execution tools.

These are the same tools you built in Lab 1.2, provided here as a
pre-wired module so you can focus on the framework comparison in Lab 1.4.

Each framework skeleton imports the raw Python functions from here and
wraps them in its own tool/function format.

Students do NOT modify this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Workspace directory – all file operations are sandboxed here.
# ---------------------------------------------------------------------------
WORKSPACE = Path(tempfile.mkdtemp(prefix="lab14_workspace_"))


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------
def read_file(path: str) -> str:
    """Read a file from the sandbox workspace.

    Args:
        path: Relative path inside the workspace directory.

    Returns:
        File contents as a string, or an error message prefixed with "ERROR:".
    """
    target = WORKSPACE / path
    try:
        return target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: file '{path}' not found in workspace."
    except Exception as exc:
        return f"ERROR: could not read '{path}': {exc}"


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------
def write_file(path: str, content: str) -> str:
    """Write content to a file in the sandbox workspace.

    Creates any intermediate directories that do not exist.

    Args:
        path:    Relative path inside the workspace directory.
        content: Text to write.

    Returns:
        A success message including the line count, or an error message.
    """
    target = WORKSPACE / path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        line_count = len(content.splitlines())
        return f"File written successfully: '{path}' ({line_count} lines)."
    except Exception as exc:
        return f"ERROR: could not write '{path}': {exc}"


# ---------------------------------------------------------------------------
# exec_python
# ---------------------------------------------------------------------------
def exec_python(code: str, timeout: int = 10) -> str:
    """Execute a Python snippet in a subprocess sandbox.

    Args:
        code:    Python source code to run.
        timeout: Maximum execution time in seconds (default 10).

    Returns:
        Combined stdout + stderr, truncated to 2 000 characters.
        Prefixed with "ERROR:" if the process times out or raises.
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        if len(output) > 2000:
            output = output[:2000] + "\n...[truncated]"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: execution timed out after {timeout} seconds."
    except Exception as exc:
        return f"ERROR: subprocess failed: {exc}"


# ---------------------------------------------------------------------------
# JSON schemas – used by OpenAI function-calling and LangGraph tool nodes.
# ---------------------------------------------------------------------------
READ_FILE_SCHEMA: dict = {
    "name": "read_file",
    "description": "Read a file from the sandbox workspace.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the workspace.",
            }
        },
        "required": ["path"],
    },
}

WRITE_FILE_SCHEMA: dict = {
    "name": "write_file",
    "description": "Write content to a file in the sandbox workspace.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
}

EXEC_PYTHON_SCHEMA: dict = {
    "name": "exec_python",
    "description": (
        "Execute a Python code snippet in a sandboxed subprocess. "
        "Returns stdout + stderr, truncated to 2 000 characters."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            }
        },
        "required": ["code"],
    },
}

ALL_SCHEMAS: list[dict] = [READ_FILE_SCHEMA, WRITE_FILE_SCHEMA, EXEC_PYTHON_SCHEMA]


# ---------------------------------------------------------------------------
# Dispatcher – given a tool name and JSON-encoded arguments, call the
# right function and return its string result.
# ---------------------------------------------------------------------------
TOOL_REGISTRY: dict[str, object] = {
    "read_file": read_file,
    "write_file": write_file,
    "exec_python": exec_python,
}


def dispatch(tool_name: str, arguments: str | dict) -> str:
    """Call a tool by name, parsing JSON arguments if supplied as a string.

    Args:
        tool_name:  One of "read_file", "write_file", "exec_python".
        arguments:  Tool arguments as a JSON string or a plain dict.

    Returns:
        The tool's return value (always a string).
    """
    if tool_name not in TOOL_REGISTRY:
        return f"ERROR: unknown tool '{tool_name}'."
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    return TOOL_REGISTRY[tool_name](**args)
