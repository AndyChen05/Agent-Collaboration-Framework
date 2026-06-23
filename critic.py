import json
import os
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tools import TOOL_SCHEMAS, TOOL_REGISTRY
import token_tracker

# Set False to skip adversarial verification (saves tokens during DAG development).
# Set True to re-enable full two-phase critic verification.
CRITIC_ENABLED = True

# ── Verdict schema ────────────────────────────────────────────────────────────
# This is the PROTOCOL between critic and orchestrator — always this shape,
# regardless of what task was being evaluated. Task content varies; this doesn't.

class ErrorDetail(BaseModel):
    check: str
    expected: str = ""
    actual: str = ""

class CriticVerdict(BaseModel):
    passed: bool
    checks: dict[str, bool] = {}
    errors: list[ErrorDetail] = []
    feedback: str
    suggestions: str = ""

# Check platform.deepseek.com/api_keys for available model IDs.
# Known valid IDs: "deepseek-chat" (V3), "deepseek-reasoner" (R1).
# If "deepseek-v4-pro" exists as a newer model, swap it in here.
MODEL = "deepseek-chat"

client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=10.0),
        proxy=None,
    ),
)

# Critic observes — it must not modify, delete, append, or run arbitrary shell commands
_CRITIC_EXCLUDED = {"write_file", "append_to_file", "delete_file", "run_shell"}
CRITIC_TOOL_SCHEMAS = [s for s in TOOL_SCHEMAS if s["function"]["name"] not in _CRITIC_EXCLUDED]
CRITIC_TOOL_REGISTRY = {k: v for k, v in TOOL_REGISTRY.items() if k not in _CRITIC_EXCLUDED}

CRITIC_SYSTEM_PROMPT = """\
You are an adversarial QA engineer. An AI agent has attempted to complete a task.
Your job is to independently verify whether it actually succeeded.

Rules:
- Do NOT trust any claims the agent made. Verify everything yourself with your tools.
- Run the code yourself. Read the files yourself. Check the actual output.
- Be paranoid about exact format: spaces, punctuation, capitalization all matter.
- Your mindset is adversarial. The builder was optimistic. You are not.

Use your tools to gather evidence, then wait for the verdict prompt.
"""

VERDICT_PROMPT = """\
Based on your investigation above, output your final verdict as a JSON object:
{
  "passed": true or false,
  "checks": {
    "name_of_each_thing_you_verified": true or false
  },
  "errors": [
    {"check": "name of failed check", "expected": "what it should be", "actual": "what it actually was"}
  ],
  "feedback": "one paragraph: what the agent did right and what it got wrong",
  "suggestions": "specific things the agent must fix, empty string if passed"
}
Output ONLY the JSON object. No other text.
"""


async def _execute_tools(tool_calls: list) -> list[dict]:
    results = []
    for tool_call in tool_calls:
        name = tool_call.function.name
        inputs = json.loads(tool_call.function.arguments)
        print(f"  [critic tool] {name}({inputs})")
        executor = CRITIC_TOOL_REGISTRY.get(name)
        if executor is None:
            content = f"Error: unknown tool '{name}'"
        else:
            try:
                content = await executor(inputs)
            except Exception as e:
                content = f"Tool raised exception: {e}"
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"  [critic result] {preview}\n")
        results.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": content,
        })
    return results


async def run_critic(task: str, actor_result: str) -> dict:
    """
    Two-phase verification:

    Phase 1 — investigation loop:
        Critic uses tools to gather real evidence. It runs the code itself,
        reads the files itself, checks actual output vs expected output.
        This is separate from the builder's loop: the builder verified that
        its own code runs; the critic verifies that the output matches the spec.

    Phase 2 — verdict extraction:
        After investigation, a separate call with response_format=json_object
        extracts the structured verdict. Two phases exist so the critic cannot
        confuse "I believe it worked" with "I ran it and checked the output."
    """
    investigation_prompt = (
        f"ORIGINAL TASK:\n{task.strip()}\n\n"
        f"AGENT'S CLAIMED RESULT:\n{actor_result.strip()}\n\n"
        "Investigate whether the agent actually completed the task correctly. "
        "Use your tools — do not take the agent's word for it."
    )

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": investigation_prompt},
    ]

    # ── Phase 1: investigation loop ───────────────────────────────────────────
    print("\n--- Critic investigating ---")
    for _ in range(10):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=CRITIC_TOOL_SCHEMAS,
        )
        token_tracker.record(MODEL, response.usage)
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            messages.append(choice.message.model_dump())
            break

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message.model_dump())
            tool_results = await _execute_tools(choice.message.tool_calls)
            messages.extend(tool_results)

    # ── Phase 2: verdict extraction ───────────────────────────────────────────
    messages.append({"role": "user", "content": VERDICT_PROMPT})
    verdict_response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    token_tracker.record(MODEL, verdict_response.usage)

    raw = verdict_response.choices[0].message.content
    try:
        verdict = CriticVerdict(**json.loads(raw))
        return verdict.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        # json.JSONDecodeError: model returned malformed JSON despite json_object mode
        # ValidationError: JSON parsed but missing/wrong-typed fields (e.g. passed is a string not bool)
        return CriticVerdict(
            passed=False,
            feedback=f"Critic verdict failed schema validation: {e}",
            suggestions="Internal error — retry.",
        ).model_dump()
