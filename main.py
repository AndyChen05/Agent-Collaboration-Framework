import argparse
import asyncio
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv must run before any agent/critic/orchestrator imports,
# because those modules create API clients at module level.
load_dotenv(Path(__file__).parent / ".env")

import json
import run_log
import run_manifest
import reflect
import token_tracker
from orchestrator import run_with_oversight
from meta_orchestrator import run_with_meta_oversight
from dag_orchestrator import DAGNode, run_dag

MANIFEST_FILE = Path(__file__).parent / ".last_run_manifest.json"

# ── Mode switches ─────────────────────────────────────────────────────────────
# USE_DAG = True   → DAG orchestrator (parallel waves, smaller per-node context)
# USE_DAG = False, USE_META = True  → meta-orchestrator (LLM decides retry strategy)
# USE_DAG = False, USE_META = False → plain sequential actor-critic loop
USE_DAG = True
USE_META = False

# ── Monolithic task (used when USE_DAG = False) ───────────────────────────────
TASK = """
Build a Python package called text_utils with the following structure:

text_utils/
    __init__.py          — exports all public functions
    analyzer.py          — word_count(text), char_count(text), sentence_count(text), top_words(text, n=5)
    formatter.py         — wrap_text(text, width=80), title_case(text), clean_whitespace(text)
tests/
    test_analyzer.py     — at least 6 pytest tests covering all 4 analyzer functions
    test_formatter.py    — at least 6 pytest tests covering all 3 formatter functions

Requirements:
- Use check_syntax on every module before writing it to disk
- All functions must include type hints and a one-line docstring
- word_count returns int (number of words), char_count returns int (number of non-whitespace chars),
  sentence_count returns int (split on '.', '!', '?'), top_words returns list of (word, count) tuples
- wrap_text wraps at given width using textwrap, title_case capitalizes each word,
  clean_whitespace collapses multiple spaces/newlines into single space and strips
- Run the full test suite with run_tests once everything is written
- All 12+ tests must pass
- After confirming tests pass, delete any __pycache__ directories created during testing
- Add text_utils/ and tests/ to .gitignore (append to the existing .gitignore file)
"""

# ── DAG task decomposition (used when USE_DAG = True) ────────────────────────
# Wave 1 (parallel): analyzer_module | formatter_module
# Wave 2 (parallel): init_module | test_analyzer | test_formatter
# Wave 3 (serial):   run_and_cleanup
DAG_NODES = [
    DAGNode(
        "analyzer_module",
        "Write text_utils/analyzer.py with these four functions:\n"
        "  word_count(text: str) -> int       — number of words (split on whitespace)\n"
        "  char_count(text: str) -> int       — number of non-whitespace characters\n"
        "  sentence_count(text: str) -> int   — split on '.', '!', '?'; count non-empty segments\n"
        "  top_words(text: str, n: int = 5) -> list[tuple[str, int]]  — most frequent words via Counter\n"
        "Requirements: use check_syntax before writing; all functions need type hints and a one-line docstring.",
    ),
    DAGNode(
        "formatter_module",
        "Write text_utils/formatter.py with these three functions:\n"
        "  wrap_text(text: str, width: int = 80) -> str   — use textwrap.fill\n"
        "  title_case(text: str) -> str                    — capitalize each word\n"
        "  clean_whitespace(text: str) -> str              — collapse whitespace to single space and strip\n"
        "Requirements: use check_syntax before writing; all functions need type hints and a one-line docstring.",
    ),
    DAGNode(
        "init_module",
        "Write text_utils/__init__.py that imports and re-exports all 7 public functions "
        "(word_count, char_count, sentence_count, top_words from analyzer; "
        "wrap_text, title_case, clean_whitespace from formatter) via __all__.\n"
        "Use check_syntax before writing.",
        depends_on=["analyzer_module", "formatter_module"],
    ),
    DAGNode(
        "test_analyzer",
        "Write tests/test_analyzer.py with at least 6 pytest tests covering all 4 functions "
        "in text_utils/analyzer.py (word_count, char_count, sentence_count, top_words).\n"
        "Use check_syntax before writing.",
        depends_on=["analyzer_module"],
    ),
    DAGNode(
        "test_formatter",
        "Write tests/test_formatter.py with at least 6 pytest tests covering all 3 functions "
        "in text_utils/formatter.py (wrap_text, title_case, clean_whitespace).\n"
        "Use check_syntax before writing.",
        depends_on=["formatter_module"],
    ),
    DAGNode(
        "run_and_cleanup",
        "Run the full test suite with run_tests on the tests/ directory. "
        "All 12+ tests must pass.\n"
        "After confirming tests pass:\n"
        "  1. Delete any __pycache__ directories (use run_shell or run_python with shutil)\n"
        "  2. Append 'text_utils/' and 'tests/' to .gitignore if not already present",
        depends_on=["init_module", "test_analyzer", "test_formatter"],
    ),
]


async def main():
    token_tracker.reset()
    run_manifest.reset()
    run_log.reset()

    if USE_DAG:
        print("\n" + "="*60)
        print("  MODE: DAG orchestration")
        print("="*60)
        results = await run_dag(DAG_NODES)
        token_tracker.print_summary()

        print(f"\n{'='*60}")
        print("FINAL OUTCOME — DAG")
        print("="*60)
        all_passed = all(r.get("verdict", {}).get("passed", False) for r in results.values())
        print(f"All nodes passed : {all_passed}")
        for name, outcome in results.items():
            passed = outcome.get("verdict", {}).get("passed", False)
            rounds = outcome.get("rounds", "?")
            print(f"  [{('✓' if passed else '✗')}] {name} ({rounds} round(s))")

    else:
        if USE_META:
            print("\n" + "="*60)
            print("  MODE: Meta-orchestrator")
            print("="*60)
            outcome = await run_with_meta_oversight(TASK)
        else:
            print("\n" + "="*60)
            print("  MODE: Sequential orchestration")
            print("="*60)
            outcome = await run_with_oversight(TASK)
        token_tracker.print_summary()

        verdict = outcome["verdict"]
        print(f"\n{'='*60}")
        print("FINAL OUTCOME — Sequential")
        print("="*60)
        print(f"Passed : {verdict.get('passed')}")
        print(f"Rounds : {outcome['rounds']}")
        print(f"Checks : {verdict.get('checks', {})}")
        if "note" in outcome:
            print(f"Note   : {outcome['note']}")
        print(f"\n--- Actor's final result ---\n{outcome['result']}")

    # Save manifest so --clean knows what to remove next time
    paths = [str(p) for p in run_manifest.all_paths()]
    MANIFEST_FILE.write_text(json.dumps(paths, indent=2), encoding="utf-8")

    # Post-run reflection — runs after token summary, so cost report stays clean
    await reflect.run()


def handle_clean() -> None:
    if not MANIFEST_FILE.exists():
        print("No manifest found — run the agent at least once first.")
        sys.exit(0)

    paths = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if not paths:
        print("Manifest is empty — nothing to delete.")
        sys.exit(0)

    # Collapse to unique top-level dirs/files so the prompt is readable
    cwd = Path.cwd()
    roots: set[Path] = set()
    for p in paths:
        try:
            rel = Path(p).relative_to(cwd)
            roots.add(cwd / rel.parts[0])
        except ValueError:
            roots.add(Path(p))

    existing = sorted(r for r in roots if r.exists())
    if not existing:
        print("Nothing to delete — files are already gone.")
        sys.exit(0)

    print("The following will be permanently deleted:")
    for r in existing:
        print(f"  {r.relative_to(cwd)}")
    answer = input("Confirm delete? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)

    for r in existing:
        if r.is_dir():
            shutil.rmtree(r)
        else:
            r.unlink()
        print(f"  Deleted {r.relative_to(cwd)}")
    MANIFEST_FILE.unlink(missing_ok=True)
    print("Done.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete files from the last run (recorded in .last_run_manifest.json).",
    )
    args = parser.parse_args()

    if args.clean:
        handle_clean()

    asyncio.run(main())
