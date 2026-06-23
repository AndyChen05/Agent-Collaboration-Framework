"""
Tracks files written by the agent during a run.
tools.py records every write here; main.py saves and loads the manifest for --clean.
"""

from pathlib import Path

_written: set[str] = set()


def record(path: str | Path) -> None:
    _written.add(str(Path(path).resolve()))


def top_level_dirs() -> list[Path]:
    """Return the unique top-level directories (relative to cwd) of all recorded paths."""
    cwd = Path.cwd()
    roots: set[Path] = set()
    for p in _written:
        try:
            rel = Path(p).relative_to(cwd)
            roots.add(cwd / rel.parts[0])
        except ValueError:
            roots.add(Path(p))
    return sorted(roots)


def all_paths() -> list[Path]:
    return sorted(Path(p) for p in _written)


def reset() -> None:
    _written.clear()
