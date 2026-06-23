"""
DAG orchestrator — runs a task graph where independent nodes execute in parallel
via asyncio.gather(), and dependent nodes receive prior results as context.

Each node goes through the full run_with_oversight() loop (actor + optional critic).
The messages list is local to each node, so context size stays bounded regardless
of total task complexity.

Usage:
    from dag_orchestrator import DAGNode, run_dag

    nodes = [
        DAGNode("a", "Do X"),
        DAGNode("b", "Do Y"),
        DAGNode("c", "Do Z using X and Y", depends_on=["a", "b"]),
    ]
    results = await run_dag(nodes)
"""

import asyncio
from dataclasses import dataclass, field
from orchestrator import run_with_oversight


@dataclass
class DAGNode:
    name: str
    task: str
    depends_on: list[str] = field(default_factory=list)


async def run_dag(nodes: list[DAGNode]) -> dict[str, dict]:
    """
    Execute a DAG of tasks, parallelising independent nodes at each wave.

    Returns a dict mapping node name → run_with_oversight() outcome dict.
    Stops early if a node fails (passed=False after all rounds).
    """
    results: dict[str, dict] = {}
    completed: set[str] = set()
    wave = 0

    while len(completed) < len(nodes):
        wave += 1

        ready = [
            n for n in nodes
            if n.name not in completed
            and all(dep in completed for dep in n.depends_on)
        ]

        if not ready:
            stuck = [n.name for n in nodes if n.name not in completed]
            print(f"\nDAG: no runnable nodes — possible cycle or missing dependency. Stuck: {stuck}")
            break

        print(f"\n{'━'*60}")
        print(f"  DAG WAVE {wave} — {len(ready)} node(s) running in parallel:")
        for n in ready:
            deps = f" (needs: {', '.join(n.depends_on)})" if n.depends_on else ""
            print(f"    • {n.name}{deps}")
        print(f"{'━'*60}")

        # Inject prior node results into tasks that have dependencies
        full_tasks = []
        for node in ready:
            if node.depends_on:
                context_parts = [
                    f"=== {dep} completed ===\n{results[dep].get('result', '')}"
                    for dep in node.depends_on
                ]
                context = "\n\n".join(context_parts)
                full_task = (
                    f"Context from completed prior tasks:\n{context}\n\n"
                    f"{'─'*40}\n"
                    f"YOUR TASK:\n{node.task}"
                )
            else:
                full_task = node.task
            full_tasks.append(full_task)

        # Run this wave in parallel
        outcomes = await asyncio.gather(
            *[run_with_oversight(task) for task in full_tasks]
        )

        for node, outcome in zip(ready, outcomes):
            results[node.name] = outcome
            completed.add(node.name)
            passed = outcome.get("verdict", {}).get("passed", False)
            status = "PASS ✓" if passed else "FAIL ✗"
            print(f"\n  [{status}] {node.name} — {outcome.get('rounds', '?')} round(s)")

            if not passed:
                print(f"  DAG stopping: node '{node.name}' did not pass verification.")
                return results

    print(f"\nDAG complete — {len(completed)}/{len(nodes)} nodes finished in {wave} wave(s).")
    return results
