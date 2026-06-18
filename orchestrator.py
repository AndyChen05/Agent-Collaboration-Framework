from agent import run_agent
from critic import run_critic, CRITIC_ENABLED

MAX_ROUNDS = 3


async def run_with_oversight(task: str) -> dict:
    """
    Actor-critic loop with evidence-based feedback:

      1. Actor (run_agent) attempts the task.
      2. Critic (run_critic) independently verifies using tools — not just reading text.
      3. If passed: done.
      4. If failed: build a revision prompt that includes the actual check failures
         (expected vs actual values from real execution), then re-run the actor.
      5. Give up after MAX_ROUNDS.

    The key communication improvement over a score system: the actor receives
    concrete evidence ("expected 'Hello, World!\\n', got 'hello world\\n'") instead
    of a vague grade ("7/10"). That evidence comes from the critic actually running
    the code — not from reading the actor's claims.
    """
    current_task = task
    last_result = ""
    last_verdict = {}

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'━'*60}")
        print(f"  ROUND {round_num} / {MAX_ROUNDS}")
        print(f"{'━'*60}")

        # ── Actor ─────────────────────────────────────────────────────────────
        last_result = await run_agent(current_task)

        # ── Critic ────────────────────────────────────────────────────────────
        if not CRITIC_ENABLED:
            last_verdict = {"passed": True, "checks": {}, "errors": [],
                            "feedback": "Critic disabled.", "suggestions": ""}
        else:
            print(f"\n--- Critic reviewing round {round_num} ---")
            last_verdict = await run_critic(task, last_result)

        passed = last_verdict.get("passed", False)
        checks = last_verdict.get("checks", {})
        errors = last_verdict.get("errors", [])
        feedback = last_verdict.get("feedback", "")
        suggestions = last_verdict.get("suggestions", "")

        print(f"Verdict  : {'PASS ✓' if passed else 'FAIL ✗'}")
        for check_name, check_passed in checks.items():
            mark = "✓" if check_passed else "✗"
            print(f"  [{mark}] {check_name}")
        for err in errors:
            print(f"      expected : {err.get('expected')!r}")
            print(f"      actual   : {err.get('actual')!r}")
        print(f"Feedback : {feedback}")

        if passed:
            print(f"\nApproved after {round_num} round(s).")
            return {"result": last_result, "verdict": last_verdict, "rounds": round_num}

        if round_num < MAX_ROUNDS:
            print(f"\nFeeding evidence back to actor for round {round_num + 1}...")

            # Build a revision prompt with actual evidence from the critic's execution
            error_lines = "\n".join(
                f"  - {e.get('check')}: "
                f"expected {e.get('expected')!r}, "
                f"got {e.get('actual')!r}"
                for e in errors
            ) or "  (no specific error details)"

            current_task = (
                f"{task}\n\n"
                f"--- REVISION REQUIRED (round {round_num} failed independent verification) ---\n"
                f"A separate QA agent ran your code and found these failures:\n"
                f"{error_lines}\n\n"
                f"QA feedback : {feedback}\n"
                f"What to fix : {suggestions}"
            )

    print(f"\nMax rounds ({MAX_ROUNDS}) reached without passing.")
    return {
        "result": last_result,
        "verdict": last_verdict,
        "rounds": MAX_ROUNDS,
        "note": "did not pass after max rounds",
    }
