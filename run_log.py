"""
Tracks tool errors during a run for post-run reflection.
tools.py appends here on non-zero exits; reflect.py reads it after the run.
"""

from dataclasses import dataclass


@dataclass
class ToolError:
    tool: str
    summary: str   # what was attempted (command string, file path, etc.)
    error: str     # the error output / stderr


_errors: list[ToolError] = []


def record(tool: str, summary: str, error: str) -> None:
    _errors.append(ToolError(tool=tool, summary=summary, error=error))


def get_errors() -> list[ToolError]:
    return list(_errors)


def reset() -> None:
    _errors.clear()
