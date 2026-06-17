import asyncio
import json
import os
from openai import AsyncOpenAI
from tools import TOOL_SCHEMAS, TOOL_REGISTRY

MODEL = "deepseek-v4-pro"   # swap to "deepseek-reasoner" for the R1 reasoning model
MAX_ITERATIONS = 20

client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def _extract_final_text(message) -> str:
    return message.content or "(agent finished with no text response)"


async def _execute_tools(tool_calls: list) -> list[dict]:
    """
    Run every tool call the model requested and return a list of
    tool-result messages in OpenAI format (role: "tool").
    """
    tool_result_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_input = json.loads(tool_call.function.arguments)

        print(f"  [tool] {tool_name}({tool_input})")

        executor = TOOL_REGISTRY.get(tool_name)
        if executor is None:
            result_content = f"Error: unknown tool '{tool_name}'"
        else:
            try:
                result_content = await executor(tool_input)
            except Exception as e:
                result_content = f"Tool raised an exception: {e}"

        preview = result_content[:300] + "..." if len(result_content) > 300 else result_content
        print(f"  [result] {preview}\n")

        tool_result_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_content,
        })

    return tool_result_messages


async def run_agent(task: str) -> str:
    """
    Agentic loop for DeepSeek (OpenAI-compatible API).

    Flow per iteration:
      1. Call the model with current messages + tool schemas
      2. finish_reason == "stop"       → done, return text
      3. finish_reason == "tool_calls" → execute tools, append results, loop
    """
    messages = [{"role": "user", "content": task}]
    iteration = 0

    print(f"\n{'='*60}")
    print(f"TASK: {task.strip()}")
    print(f"{'='*60}\n")

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        response = await client.chat.completions.create(
            model=MODEL,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            messages=messages,
        )

        choice = response.choices[0]
        print(f"Finish reason: {choice.finish_reason}")

        # ── Done ──────────────────────────────────────────────────────────
        if choice.finish_reason == "stop":
            return _extract_final_text(choice.message)
    
        # ── Tool calls ────────────────────────────────────────────────────
        if choice.finish_reason == "tool_calls":
            # reasoning_content exists on deepseek-reasoner (R1) but not deepseek-chat (V3)
            reasoning = getattr(choice.message, "reasoning_content", None)
            if reasoning:
                print(f"  [reasoning] {reasoning[:500]}")
            # content is often None with deepseek-chat when calling tools
            print(f"  [thinking]  {choice.message.content or '(none)'}")
            messages.append(choice.message.model_dump())

            # Execute every requested tool and append results
            tool_results = await _execute_tools(choice.message.tool_calls)
            messages.extend(tool_results)
            continue

        # ── Unexpected ────────────────────────────────────────────────────
        print(f"Unexpected finish reason: {choice.finish_reason!r} — stopping.")
        break

    return f"Agent stopped after {iteration} iterations without completing the task."
