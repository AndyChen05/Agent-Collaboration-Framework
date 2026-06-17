import asyncio
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv must run before any agent/critic/orchestrator imports,
# because those modules create API clients at module level.
load_dotenv(Path(__file__).parent / ".env")

from orchestrator import run_with_oversight

# ── Change this task to try different things ──────────────────────────────────
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
"""

async def main():
    outcome = await run_with_oversight(TASK)
    verdict = outcome["verdict"]
    print(f"\n{'='*60}")
    print("FINAL OUTCOME")
    print('='*60)
    print(f"Passed : {verdict.get('passed')}")
    print(f"Rounds : {outcome['rounds']}")
    print(f"Checks : {verdict.get('checks', {})}")
    if "note" in outcome:
        print(f"Note   : {outcome['note']}")
    print(f"\n--- Actor's final result ---\n{outcome['result']}")

if __name__ == "__main__":
    asyncio.run(main())
