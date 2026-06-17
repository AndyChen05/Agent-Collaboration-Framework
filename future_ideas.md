# Future Ideas & Architecture Reference
# Multiagent Orchestration Project

---

## Current System (as of June 2026)

```
main.py → orchestrator.py → agent.py (actor) → tools.py
                          → critic.py (verifier) → tools.py (read-only subset)
```

- Actor: DeepSeek V3 (`deepseek-chat`), optimistic builder, all 4 tools
- Critic: DeepSeek V3 (`deepseek-chat`), adversarial verifier, 3 tools (no write_file)
- Orchestrator: pure Python loop, no LLM inside it, MAX_ROUNDS=3
- Communication: one-way via orchestrator — critic's JSON errors formatted into actor's next task prompt
- Actor and critic never talk directly

---

## Environment Design (what it means)

Environment design = engineering everything AROUND the model instead of changing the model through prompts.

Three layers:

### 1. Tool Design (what actions agents can take)
**Granular tools** — instead of one big tool, many small precise ones:

| Coarse (bad) | Granular (good) |
|---|---|
| `run_code(code)` | `run_file(path)`, `run_inline(code)`, `run_tests(test_path)`, `check_syntax(code)` |
| `manage_file(op, path)` | `read_file(path)`, `write_file(path, content)`, `delete_file(path)`, `move_file(src, dst)` |

Why it matters: granular tools give the model precise vocabulary. It can't accidentally delete a file when it only has a `run_file` tool. Tool boundaries = behavioral guardrails.

### 2. Sandboxed Execution (where code runs safely)
A sandbox is an isolated environment where agent code runs without affecting anything outside it.

**Current state:** our `run_python` tool runs code directly on your machine. If the agent writes `import os; os.rmdir(".")`, it runs on your actual filesystem.

**Sandboxed state:** agent code runs in a container (Docker) or cloud VM (E2B). If it crashes, corrupts files, or does something dangerous — the sandbox absorbs it. Your machine is untouched.

```
Current:  agent code → your machine's Python → your real filesystem
Sandboxed: agent code → Docker container → isolated virtual filesystem
                                         ↑ disposable, reset per task
```

**E2B** (e2b.dev) is a service that gives you a cloud Python sandbox via API — you send code, it runs it isolated, returns stdout/stderr. No Docker setup needed.

### 3. Structured Output Validation (forcing correct output shape)
Instead of trusting the model to return the right format, validate with code:

```python
# Prompt engineering (weak — hope it follows instructions)
"Please return a JSON with fields: passed, checks, errors"

# Environment design (strong — crash if it doesn't)
from pydantic import BaseModel
class CriticVerdict(BaseModel):
    passed: bool
    checks: dict[str, bool]
    errors: list[dict]
    feedback: str

verdict = CriticVerdict(**json.loads(raw))  # raises if schema wrong
```

Pydantic is the standard library for this in Python.

---

## Future Architecture Patterns

### Pipeline Pattern (NEXT PRIORITY)
Actor finishes segment → critic reviews segment → while actor starts next segment.

```
Time:  |--actor seg1--|--actor seg2--|--actor seg3--|
                      |--critic1--|  |--critic2--|  |--critic3--|
```

Requires: actor must emit discrete checkpoint objects, not one final string.
How: actor returns a list of `Segment(name, content, type)` objects. Orchestrator triggers `asyncio.gather(actor(next_seg), critic(prev_seg))` at each boundary.

Files to change: `agent.py` (return segments, not string), `orchestrator.py` (gather logic), create `segment.py` (dataclass).

---

### DAG Orchestration (Directed Acyclic Graph)
Tasks as nodes, dependencies as edges. Independent nodes run in parallel via `asyncio.gather()`.

```
         ┌─ write_login ─────────────────────┐
START ───┤                                    ├──► run_integration_tests ──► DONE
         └─ write_database ─► write_schema ──┘
```

- `write_login` and `write_database` have no dependency → run in parallel
- `write_schema` must wait for `write_database`
- `run_integration_tests` must wait for both branches to finish

```python
# Python implementation of the above DAG
login_result, db_result = await asyncio.gather(
    run_agent("write the login module"),
    run_agent("write the database module"),
)
schema_result = await run_agent(f"write schema based on: {db_result}")
integration = await run_agent(f"run integration tests for: {login_result} + {schema_result}")
```

Reference: look at Prefect, Airflow, or GitHub Actions for DAG mental model.
Files to create: `dag.py` (DAGNode, DAGRunner classes), update `orchestrator.py`.

---

### Shared Blackboard
A shared state object all agents read and write. Removes direct agent-to-agent data passing.

```python
blackboard = {
    "task": "build hello.py",
    "status": "in_progress",
    "files_created": [],
    "test_results": {},
    "critic_verdict": None,
    "round": 1,
}
```

Actor writes `files_created`, `test_results`. Critic reads both, writes `critic_verdict`. Orchestrator reads everything to decide next step. No agent needs to know who else exists.

Benefits: decoupled agents, inspectable state, easy to add new agents (just have them read/write the board).
Local: Python dict. Distributed: Redis.
Files to create: `blackboard.py`, update all agents to read/write it.

---

### Dynamic Routing (Specialist Agents)
A fast classifier decides which specialist handles each subtask.

```
Task arrives → Classifier LLM → "this is a database task" → Database Agent
                              → "this is a frontend task" → Frontend Agent
                              → "this is a security task" → Security Agent
```

The classifier uses a cheap, fast model (haiku-tier). Specialist agents are tuned/prompted for their domain.

This is the pattern you said you understood — it's exactly this. The orchestrator doesn't need to know all possible task types upfront; the classifier figures it out dynamically.

Files to create: `classifier.py`, `agents/database.py`, `agents/frontend.py`, `agents/security.py`, update `orchestrator.py` with routing logic.

---

### Speculative Execution
Run N actors on the same task in parallel. Critic picks the best.

```python
results = await asyncio.gather(
    run_agent(task),  # actor 1
    run_agent(task),  # actor 2 — same task, different random seed
    run_agent(task),  # actor 3
)
# critic evaluates all 3, picks highest quality
best = await run_selector(task, results)
```

Cost: 3× API calls. Benefit: critic picks from 3 attempts instead of retrying 3 times sequentially.
When to use: quality matters more than cost, task has high variance in output quality.

---

### Event Bus (Pub/Sub)
Agents publish events. Other agents subscribe. Fully decoupled — no agent calls another directly.

```python
# Actor publishes, doesn't know critic exists
await bus.publish("task_complete", {"result": result})

# Critic subscribes, doesn't know actor exists
@bus.subscribe("task_complete")
async def on_task_complete(event):
    verdict = await run_critic(event["task"], event["result"])
    await bus.publish("critique_done", verdict)

# Orchestrator subscribes to both
@bus.subscribe("critique_done")
async def on_critique(event):
    if not event["passed"]:
        await bus.publish("task_retry", {...})
```

Local: `asyncio.Queue`. Distributed: Redis pub/sub, Kafka, RabbitMQ.
This is how you'd architect a system where agents are fully independent services.

---

### Meta-Agent Orchestrator
Replace the hardcoded Python loop with an LLM that decides what to do next.

```
Current orchestrator (dumb loop):
  if critic_passed: return
  if rounds < 3: retry actor
  else: give up

Meta-agent orchestrator (LLM decides):
  LLM sees: task, actor_result, critic_verdict, rounds_used, history
  LLM decides: "retry" | "decompose" | "escalate" | "accept" | "try_different_actor"
```

**Can it spawn agents?** Yes — the meta-orchestrator can dynamically create new agents based on what it decides:

```python
decision = await meta_llm.decide(context)
if decision == "decompose":
    subtasks = await meta_llm.split(task)
    results = await asyncio.gather(*[run_agent(s) for s in subtasks])
elif decision == "escalate":
    await notify_human(context)
elif decision == "specialist":
    return await run_specialist_agent(decision.specialist_type, task)
```

The meta-orchestrator is itself an agent. It makes decisions that change the structure of the system at runtime. This is what LangGraph and AutoGen implement.

Files to create: `meta_orchestrator.py`, replace `orchestrator.py` routing logic with LLM calls.

---

### Human-in-the-Loop
Orchestrator pauses at defined checkpoints, waits for human approval before continuing.

```python
if requires_human_review(verdict):
    print("Waiting for human approval...")
    approval = input("Approve? (y/n): ")
    if approval != "y":
        return escalate(task, verdict)
```

Production version: webhook, Slack message, email, dashboard button.
Critical for: irreversible actions, production deployments, high-stakes decisions.

---

## Collaboration Logic Rule

**Orchestrator.py is the single source of truth for agent collaboration.**

Any change to HOW agents work together goes in `orchestrator.py`:
- Adding a third agent
- Making two agents parallel
- Changing retry logic
- Adding routing logic
- Adding human checkpoints

Actor and critic stay focused on their single job. They don't know about each other.
The orchestrator is the only component that knows both exist.

---

## Tech Stack to Explore (by category)

| Category | Local/Simple | Production/Distributed |
|---|---|---|
| Sandboxing | subprocess + tempfile (current) | Docker, E2B |
| Output validation | json.loads + try/except (current) | Pydantic |
| Shared state | Python dict | Redis, SQLite |
| Event bus | asyncio.Queue | Redis pub/sub, Kafka |
| DAG runner | custom Python | Prefect, Airflow |
| Persistent memory | json file | Chroma (vector DB), Pinecone |
| Monitoring | print() | Langfuse, OpenTelemetry |
| Model routing | if/else | classifier agent + specialist agents |

---

## Model Notes

- `deepseek-chat` = DeepSeek V3 (fast, cheap, default)
- `deepseek-reasoner` = DeepSeek R1 (step-by-step reasoning, better for evaluation/critic)
- Verify any new model IDs at platform.deepseek.com before using
- Claude as actor: requires Anthropic API credits (separate from Claude Pro subscription)
- For multi-model: Claude as actor + DeepSeek as critic gives genuine diversity of perspective
