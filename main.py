import asyncio
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv must run before any agent/critic/orchestrator imports,
# because those modules create API clients at module level.
load_dotenv(Path(__file__).parent / ".env")

from orchestrator import run_with_oversight

# ── Change this task to try different things ──────────────────────────────────
TASK = """
Create a file called calculator.py with four functions:
  - add(a, b)      — returns a + b
  - subtract(a, b) — returns a - b
  - multiply(a, b) — returns a * b
  - divide(a, b)   — returns a / b, raises ValueError if b is 0

All functions must accept both int and float. Include type hints.

Then create test_calculator.py with at least 8 pytest tests covering:
  1. add with integers
  2. add with floats
  3. subtract
  4. multiply
  5. divide normal case
  6. divide by zero raises ValueError
  7. multiply by negative number
  8. chaining (e.g. add result fed into multiply)

Run the tests with: python -m pytest test_calculator.py -v
All 8 tests must pass. Show the full pytest output.
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
