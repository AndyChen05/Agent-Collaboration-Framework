import ast
import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Tool schemas — OpenAI/DeepSeek format: {"type": "function", "function": {...}}
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    # ── File operations ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full contents of a file. "
                "Use this to inspect existing files before modifying them. "
                "For large files, use search_in_file first to locate the section you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file, creating it if it does not exist and overwriting if it does. "
                "Use append_to_file if you want to add to an existing file without destroying its contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write to."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_to_file",
            "description": (
                "Append text to the end of an existing file. "
                "Creates the file if it does not exist. "
                "Use this instead of write_file when you want to add content without overwriting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "content": {"type": "string", "description": "Text to append."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file at the given path. Fails if the path does not exist or is a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to delete."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List the files and folders inside a directory. "
                "Use this to understand what already exists before reading or writing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to '.' (current directory).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": (
                "Search for lines matching a pattern (plain text or regex) in a file. "
                "Returns matching lines with their line numbers. "
                "Use this to verify that a function or class exists, or to find a specific section."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to search in."},
                    "pattern": {"type": "string", "description": "Plain text or regex pattern to search for."},
                },
                "required": ["path", "pattern"],
            },
        },
    },
    # ── Execution ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python code snippet inline and return stdout and stderr. "
                "Use this for quick calculations, data manipulation, or logic that does not need a separate file. "
                "Use run_shell to run an existing .py file or pytest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to execute."},
                    "timeout": {"type": "number", "description": "Max seconds to wait. Default: 10."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run any shell command and return its stdout and stderr. "
                "Pass the command as a list of strings: [\"git\", \"status\"] or [\"python\", \"hello.py\"]. "
                "Use this for git, pip, file execution, and any other shell operations. "
                "Prefer run_tests for running pytest — it has better defaults."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command as a list of tokens, e.g. [\"git\", \"log\", \"--oneline\", \"-5\"].",
                    },
                    "timeout": {"type": "number", "description": "Max seconds to wait. Default: 15."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run pytest on a file or directory and return the full test output. "
                "Always runs with -v (verbose) so individual test results are visible. "
                "Use this whenever you need to execute a test suite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to a test file or directory containing tests. Default: '.'",
                    },
                    "timeout": {"type": "number", "description": "Max seconds to wait. Default: 30."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_syntax",
            "description": (
                "Validate Python syntax without executing the code. "
                "Returns 'Syntax OK' or the exact SyntaxError with line number. "
                "Use this as a fast pre-check before running or writing a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to validate."},
                },
                "required": ["code"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

async def execute_read_file(inputs: dict) -> str:
    path = Path(inputs["path"])
    if not path.exists():
        return f"Error: file not found at '{path}'"
    if not path.is_file():
        return f"Error: '{path}' is a directory, not a file"
    return path.read_text(encoding="utf-8")


async def execute_write_file(inputs: dict) -> str:
    path = Path(inputs["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inputs["content"], encoding="utf-8")
    return f"Written {len(inputs['content'])} characters to '{path}'"


async def execute_append_to_file(inputs: dict) -> str:
    path = Path(inputs["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(inputs["content"])
    return f"Appended {len(inputs['content'])} characters to '{path}'"


async def execute_delete_file(inputs: dict) -> str:
    path = Path(inputs["path"])
    if not path.exists():
        return f"Error: '{path}' does not exist"
    if not path.is_file():
        return f"Error: '{path}' is a directory — use a shell command to remove directories"
    path.unlink()
    return f"Deleted '{path}'"


async def execute_list_directory(inputs: dict) -> str:
    path = Path(inputs.get("path", "."))
    if not path.exists():
        return f"Error: path '{path}' does not exist"
    if not path.is_dir():
        return f"Error: '{path}' is a file, not a directory"
    entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
    if not entries:
        return "(empty directory)"
    lines = []
    for entry in entries:
        tag = "FILE" if entry.is_file() else "DIR "
        lines.append(f"  [{tag}] {entry.name}")
    return "\n".join(lines)


async def execute_search_in_file(inputs: dict) -> str:
    path = Path(inputs["path"])
    if not path.exists():
        return f"Error: file not found at '{path}'"
    pattern = inputs["pattern"]
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"
    matches = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if regex.search(line):
            matches.append(f"  L{i}: {line}")
    if not matches:
        return f"No matches for '{pattern}' in '{path}'"
    return f"Found {len(matches)} match(es) for '{pattern}' in '{path}':\n" + "\n".join(matches)


async def execute_run_python(inputs: dict) -> str:
    code = inputs["code"]
    timeout = inputs.get("timeout", 10)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        parts = []
        if stdout:
            parts.append(f"STDOUT:\n{stdout.decode('utf-8', errors='replace')}")
        if stderr:
            parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
        if proc.returncode != 0:
            parts.append(f"Exit code: {proc.returncode}")
        return "\n".join(parts) if parts else "(no output)"
    except asyncio.TimeoutError:
        return f"Error: execution timed out after {timeout}s"
    finally:
        os.unlink(tmp_path)


async def execute_run_shell(inputs: dict) -> str:
    command = inputs["command"]
    timeout = inputs.get("timeout", 15)
    if not isinstance(command, list) or not command:
        return "Error: 'command' must be a non-empty list of strings"
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        parts = []
        if stdout:
            parts.append(f"STDOUT:\n{stdout.decode('utf-8', errors='replace')}")
        if stderr:
            parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
        if proc.returncode != 0:
            parts.append(f"Exit code: {proc.returncode}")
        return "\n".join(parts) if parts else "(no output)"
    except asyncio.TimeoutError:
        return f"Error: command timed out after {timeout}s"
    except FileNotFoundError:
        return f"Error: command not found: '{command[0]}'"


async def execute_run_tests(inputs: dict) -> str:
    path = inputs.get("path", ".")
    timeout = inputs.get("timeout", 30)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pytest", path, "-v",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        parts = []
        if stdout:
            parts.append(stdout.decode("utf-8", errors="replace"))
        if stderr:
            parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
        return "\n".join(parts) if parts else "(no output)"
    except asyncio.TimeoutError:
        return f"Error: tests timed out after {timeout}s"


async def execute_check_syntax(inputs: dict) -> str:
    code = inputs["code"]
    try:
        ast.parse(code)
        return "Syntax OK"
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}\n  {e.text}"


# ---------------------------------------------------------------------------
# Registry — maps tool name → executor
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, callable] = {
    "read_file":       execute_read_file,
    "write_file":      execute_write_file,
    "append_to_file":  execute_append_to_file,
    "delete_file":     execute_delete_file,
    "list_directory":  execute_list_directory,
    "search_in_file":  execute_search_in_file,
    "run_python":      execute_run_python,
    "run_shell":       execute_run_shell,
    "run_tests":       execute_run_tests,
    "check_syntax":    execute_check_syntax,
}
