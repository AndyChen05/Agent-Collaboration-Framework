import asyncio
import json
import os
from typing import Literal
import httpx


from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from agent import run_agent
from critic import run_critic
import critic
import token_tracker

MODEL = "deepseek-chat"
MAX_ROUNDS = 5  # meta gets more rounds since it can choose smarter actions

client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=10.0),
    ),
)


class MetaDecision(BaseModel):
    action: Literal["retry", "decompose", "accept", "escalate"]
    reasoning: str
    retry_guidance: str = ""
    subtasks: list[str] = []


META_SYSTEM_PROMPT = """\
You are a meta-orchestrator for a multi-agent AI system. An AI actor attempted a task, \
a critic verified it, and it failed. Your job is to decide the smartest next step.

Choose ONE action:

"retry"     — The actor made a fixable mistake. Give specific, actionable guidance on what
              to do differently. Not vague encouragement — concrete instructions.

"decompose" — The task is too broad or the actor is taking the wrong approach entirely.
              Break it into 2-4 independent, concrete subtasks that together solve the original.

"accept"    — The critic is being overly strict. The core work is correct and the failures
              are on trivial formatting or edge cases that don't matter in practice.

"escalate"  — Genuinely impossible given the constraints, or failed too many times with no
              sign of progress. Last resort.

Respond ONLY with a valid JSON object:
{
  "action": "retry" | "decompose" | "accept" | "escalate",
  "reasoning": "why you chose this action (1-2 sentences)",
  "retry_guidance": "concrete instructions for the actor — what exactly to do differently",
  "subtasks": ["subtask 1", "subtask 2"]
}
"""


async def decide(
    task: str,
    result: str,
    verdict: dict,
    rounds_used: int,
    history: list[dict],
) -> MetaDecision:
    errors = verdict.get("errors", [])
    error_lines = "\n".join(
        f"  - {e.get('check')}: expected {e.get('expected')!r}, got {e.get('actual')!r}"
        for e in errors
    ) or "  (no specific error details)"

    history_lines = "\n".join(
        f"  Round {h['round']}: action={h['action']}, guidance={h.get('guidance', '')}"
        for h in history
    ) or "  (first failure)"

    user_message = (
        f"TASK:\n{task}\n\n"
        f"ACTOR RESULT (first 1000 chars):\n{result[:1000]}\n\n"
        f"CRITIC VERDICT:\n"
        f"  Checks : {verdict.get('checks', {})}\n"
        f"  Errors :\n{error_lines}\n"
        f"  Feedback   : {verdict.get('feedback', '')}\n"
        f"  Suggestions: {verdict.get('suggestions', '')}\n\n"
        f"ROUNDS USED: {rounds_used} / {MAX_ROUNDS}\n"
        f"PREVIOUS DECISIONS:\n{history_lines}"
    )

    response = await client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": META_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    token_tracker.record(MODEL, response.usage)

    raw = response.choices[0].message.content
    try:
        return MetaDecision(**json.loads(raw))
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"  [meta] Parse error: {e} — defaulting to retry")
        return MetaDecision(
            action="retry",
            reasoning="parse error in meta response",
            retry_guidance=verdict.get("suggestions", ""),
        )


async def run_with_meta_oversight(task: str) -> dict:
    current_task = task
    last_result = ""
    last_verdict: dict = {}
    history: list[dict] = []

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'━'*60}")
        print(f"  META ROUND {round_num} / {MAX_ROUNDS}")
        print(f"{'━'*60}")

        # ── Actor ─────────────────────────────────────────────────────────────
        last_result = await run_agent(current_task)

        # ── Critic ────────────────────────────────────────────────────────────
        if not critic.CRITIC_ENABLED:
            last_verdict = {"passed": True, "checks": {}, "errors": [],
                            "feedback": "Critic disabled.", "suggestions": ""}
        else:
            print(f"\n--- Critic reviewing round {round_num} ---")
            last_verdict = await run_critic(task, last_result)

        passed = last_verdict.get("passed", False)
        checks = last_verdict.get("checks", {})
        errors = last_verdict.get("errors", [])
        feedback = last_verdict.get("feedback", "")

        print(f"Verdict  : {'PASS ✓' if passed else 'FAIL ✗'}")
        for name, ok in checks.items():
            print(f"  [{'✓' if ok else '✗'}] {name}")
        for err in errors:
            print(f"      expected : {err.get('expected')!r}")
            print(f"      actual   : {err.get('actual')!r}")
        print(f"Feedback : {feedback}")

        if passed:
            print(f"\nApproved after {round_num} round(s).")
            return {"result": last_result, "verdict": last_verdict, "rounds": round_num}

        if round_num == MAX_ROUNDS:
            break

        # ── Meta decision ─────────────────────────────────────────────────────
        print(f"\n--- Meta-orchestrator deciding ---")
        decision = await decide(task, last_result, last_verdict, round_num, history)
        print(f"  Action    : {decision.action}")
        print(f"  Reasoning : {decision.reasoning}")

        history.append({
            "round": round_num,
            "action": decision.action,
            "guidance": decision.retry_guidance or decision.reasoning,
        })

        if decision.action == "accept":
            print("\nMeta-orchestrator accepted result despite critic failure.")
            return {"result": last_result, "verdict": last_verdict, "rounds": round_num,
                    "note": "accepted by meta-orchestrator"}

        if decision.action == "escalate":
            print("\nMeta-orchestrator escalated — human review required.")
            return {"result": last_result, "verdict": last_verdict, "rounds": round_num,
                    "note": "escalated — human review required"}

        if decision.action == "decompose":
            print(f"\n  Decomposing into {len(decision.subtasks)} subtask(s):")
            for i, st in enumerate(decision.subtasks, 1):
                print(f"    {i}. {st}")
            subtask_results = await asyncio.gather(
                *[run_agent(st) for st in decision.subtasks]
            )
            combined = "\n\n".join(
                f"=== Subtask {i+1} result ===\n{r}"
                for i, r in enumerate(subtask_results)
            )
            current_task = (
                f"{task}\n\n"
                f"--- SUBTASK CONTEXT ---\n{combined}\n\n"
                f"The subtasks above are complete. Now finish the full original task."
            )

        else:  # retry
            if decision.retry_guidance:
                print(f"  Guidance  : {decision.retry_guidance}")
            error_lines = "\n".join(
                f"  - {e.get('check')}: expected {e.get('expected')!r}, got {e.get('actual')!r}"
                for e in errors
            ) or "  (no specific error details)"
            current_task = (
                f"{task}\n\n"
                f"--- REVISION REQUIRED (round {round_num} failed) ---\n"
                f"Meta-orchestrator guidance: {decision.retry_guidance}\n\n"
                f"Critic evidence:\n{error_lines}\n"
                f"Critic feedback: {feedback}"
            )

    print(f"\nMax rounds ({MAX_ROUNDS}) reached without passing.")
    return {
        "result": last_result,
        "verdict": last_verdict,
        "rounds": MAX_ROUNDS,
        "note": "did not pass after max rounds",
    }
