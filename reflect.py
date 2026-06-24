"""
Post-run reflection: reviews tool errors and updates lessons.json.

lessons.json format:
  {
    "permanent": ["confirmed constraint — injected into every actor prompt"],
    "temporary": [
      {"text": "suspected constraint", "runs": 0}
    ]
  }

Temporary lessons auto-promote to permanent after surviving 2 runs without
being contradicted. The LLM can also promote explicitly when an error recurs.

Triggered from main.py after token_tracker.print_summary() so it doesn't pollute the cost report.
"""

import json
import os
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

import run_log
import token_tracker

MODEL = "deepseek-v4-pro"
LESSONS_FILE = Path(__file__).parent / "lessons.json"
PROMOTE_AFTER_RUNS = 2

client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=10.0),
    ),
)


class ReflectionOutput(BaseModel):
    new_temporary: list[str] = []
    promote_to_permanent: list[str] = []


REFLECT_SYSTEM_PROMPT = """\
You are a learning component for a multi-agent AI system running on a specific machine.
Your job is to review tool errors that occurred during the last run and extract environment
or platform constraints worth remembering for future runs.

You will receive:
- Tool errors: what failed and how
- Existing temporary lessons: patterns seen once before (unconfirmed)
- Existing permanent lessons: confirmed constraints already in use

Your output (JSON only):
{
  "new_temporary": [
    "one-line constraint to add as a suspected lesson (if not already in temp or permanent)"
  ],
  "promote_to_permanent": [
    "exact text of a temporary lesson to promote — only if this run confirms it again"
  ]
}

Rules:
- Only extract ENVIRONMENT or PLATFORM constraints — things that are true of this machine/OS
  regardless of the task. E.g. "mkdir is not available as subprocess on Windows",
  "python must be invoked as sys.executable, not 'python'".
- Do NOT log coding mistakes, logic errors, or task-specific failures.
- Do NOT add a lesson that is already in permanent or temporary lists.
- If no errors reveal platform constraints, return {"new_temporary": [], "promote_to_permanent": []}.
- Keep each lesson under 120 characters, factual and actionable.
"""


def load_lessons() -> dict:
    if LESSONS_FILE.exists():
        data = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # migrate old flat format to two-tier
            return {"permanent": data, "temporary": []}
        # migrate temporary from list-of-strings to list-of-dicts
        temp = data.get("temporary", [])
        if temp and isinstance(temp[0], str):
            data["temporary"] = [{"text": t, "runs": 0} for t in temp]
        return data
    return {"permanent": [], "temporary": []}


def save_lessons(lessons: dict) -> None:
    LESSONS_FILE.write_text(json.dumps(lessons, indent=2), encoding="utf-8")


def _lesson_texts(lessons: dict) -> list[str]:
    return [l["text"] if isinstance(l, dict) else l for l in lessons["temporary"]]


async def run() -> None:
    lessons = load_lessons()
    changed = False

    # ── Auto-promote lessons that survived PROMOTE_AFTER_RUNS clean runs ──────
    to_promote = [
        l for l in lessons["temporary"]
        if isinstance(l, dict) and l.get("runs", 0) >= PROMOTE_AFTER_RUNS
    ]
    for lesson in to_promote:
        lessons["temporary"].remove(lesson)
        lessons["permanent"].append(lesson["text"])
        print(f"\n  [reflect] Promoted to permanent: {lesson['text']}")
        changed = True

    # Increment run count for all surviving temporary lessons
    for lesson in lessons["temporary"]:
        if isinstance(lesson, dict):
            lesson["runs"] = lesson.get("runs", 0) + 1
            changed = True

    # ── LLM-based reflection for new lessons (only when errors exist) ─────────
    errors = run_log.get_errors()
    if errors:
        error_lines = "\n".join(
            f"- [{e.tool}] attempted: {e.summary!r} → error: {e.error[:200]}"
            for e in errors
        )
        temp_lines = "\n".join(f"- {t}" for t in _lesson_texts(lessons)) or "(none)"
        perm_lines = "\n".join(f"- {l}" for l in lessons["permanent"]) or "(none)"

        user_message = (
            f"TOOL ERRORS THIS RUN:\n{error_lines}\n\n"
            f"EXISTING TEMPORARY LESSONS:\n{temp_lines}\n\n"
            f"EXISTING PERMANENT LESSONS:\n{perm_lines}"
        )

        response = await client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        token_tracker.record(MODEL, response.usage)

        raw = response.choices[0].message.content
        try:
            output = ReflectionOutput(**json.loads(raw))
        except Exception as e:
            print(f"\n  [reflect] Parse error: {e} — skipping LLM lesson update")
            output = ReflectionOutput()

        existing_texts = set(_lesson_texts(lessons)) | set(lessons["permanent"])
        for lesson in output.new_temporary:
            if lesson and lesson not in existing_texts:
                lessons["temporary"].append({"text": lesson, "runs": 0})
                print(f"\n  [reflect] New temporary lesson: {lesson}")
                changed = True

        for lesson_text in output.promote_to_permanent:
            match = next(
                (l for l in lessons["temporary"]
                 if (isinstance(l, dict) and l["text"] == lesson_text) or l == lesson_text),
                None,
            )
            if match:
                lessons["temporary"].remove(match)
                lessons["permanent"].append(lesson_text)
                print(f"\n  [reflect] Promoted to permanent (LLM confirmed): {lesson_text}")
                changed = True

    if changed:
        save_lessons(lessons)
        print(f"  [reflect] lessons.json updated ({len(lessons['permanent'])} permanent, "
              f"{len(lessons['temporary'])} temporary)")
    else:
        print("\n  [reflect] No lesson changes this run.")
