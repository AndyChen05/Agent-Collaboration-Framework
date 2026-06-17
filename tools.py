import asyncio
import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Tool schemas — OpenAI/DeepSeek format: {"type": "function", "function": {...}}
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file at the given path. "
                "Use this to inspect existing files before modifying them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    }
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
                "Write content to a file at the given path. "
                "Creates the file if it does not exist; overwrites if it does."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write into the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List the files and folders inside a directory. "
                "Use this to understand what already exists before reading or writing files."
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
            "name": "run_python",
            "description": (
                "Execute a Python code snippet in a subprocess and return its stdout and stderr. "
                "Use this to run scripts, test logic, or verify that written code works correctly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Maximum seconds to wait before killing the process. Default: 10.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Executors — unchanged: pure Python, no SDK dependency
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


async def execute_run_python(inputs: dict) -> str:
    code = inputs["code"]
    timeout = inputs.get("timeout", 10)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python", tmp_path,
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


# ---------------------------------------------------------------------------
# Registry — maps tool name → executor
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, callable] = {
    "read_file": execute_read_file,
    "write_file": execute_write_file,
    "list_directory": execute_list_directory,
    "run_python": execute_run_python,
}
