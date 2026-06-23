"""
Tracks token usage across all API calls in a run, keyed by model.
DeepSeek returns prompt_cache_hit_tokens and prompt_cache_miss_tokens
as extra fields on the usage object alongside the standard prompt_tokens
and completion_tokens.

Usage:
    import token_tracker
    token_tracker.record(MODEL, response.usage)
    token_tracker.print_summary()
    token_tracker.reset()   # call before each run if reusing the process
"""

_stats: dict[str, dict] = {}


def record(model: str, usage) -> None:
    if model not in _stats:
        _stats[model] = {"cache_hit": 0, "cache_miss": 0, "output": 0, "calls": 0}
    s = _stats[model]
    s["calls"] += 1

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    # DeepSeek stores non-standard fields in model_extra
    extra = getattr(usage, "model_extra", {}) or {}
    cache_hit = extra.get("prompt_cache_hit_tokens", 0) or 0
    cache_miss = prompt_tokens - cache_hit

    s["cache_hit"] += cache_hit
    s["cache_miss"] += cache_miss
    s["output"] += completion_tokens


def reset() -> None:
    _stats.clear()


def _pct(part: int, total: int) -> str:
    return f"{part / total * 100:.1f}%" if total > 0 else "n/a"


def print_summary() -> None:
    if not _stats:
        print("  (no token usage recorded)")
        return

    # ── Raw counts ────────────────────────────────────────────────────────────
    col = 14
    print(f"\n{'─'*68}")
    print("  TOKEN USAGE")
    print(f"{'─'*68}")
    print(f"  {'Model':<22} {'Calls':>5}  {'CacheHit':>{col}}  {'CacheMiss':>{col}}  {'Output':>{col}}")
    print(f"  {'─'*22} {'─'*5}  {'─'*col}  {'─'*col}  {'─'*col}")

    totals = {"calls": 0, "cache_hit": 0, "cache_miss": 0, "output": 0}
    for model, s in _stats.items():
        print(
            f"  {model:<22} {s['calls']:>5}  "
            f"{s['cache_hit']:>{col},}  "
            f"{s['cache_miss']:>{col},}  "
            f"{s['output']:>{col},}"
        )
        for k in totals:
            totals[k] += s[k]

    if len(_stats) > 1:
        print(f"  {'─'*22} {'─'*5}  {'─'*col}  {'─'*col}  {'─'*col}")
        print(
            f"  {'TOTAL':<22} {totals['calls']:>5}  "
            f"{totals['cache_hit']:>{col},}  "
            f"{totals['cache_miss']:>{col},}  "
            f"{totals['output']:>{col},}"
        )

    # ── Efficiency matrix ─────────────────────────────────────────────────────
    # Cache hit rate  = cache_hit  / (cache_hit + cache_miss)  — higher is cheaper
    # Input paid rate = cache_miss / (cache_hit + cache_miss)  — lower is cheaper
    # Output ratio    = output     / (cache_hit + cache_miss)  — work done per input token paid
    print(f"\n  EFFICIENCY")
    print(f"  {'─'*22} {'─'*10}  {'─'*10}  {'─'*14}")
    print(f"  {'Model':<22} {'HitRate':>10}  {'PaidRate':>10}  {'Output/Input':>14}")
    print(f"  {'─'*22} {'─'*10}  {'─'*10}  {'─'*14}")

    all_rows = list(_stats.items())
    if len(_stats) > 1:
        all_rows.append(("TOTAL", totals))

    for model, s in all_rows:
        total_input = s["cache_hit"] + s["cache_miss"]
        hit_rate  = _pct(s["cache_hit"],  total_input)
        paid_rate = _pct(s["cache_miss"], total_input)
        out_ratio = _pct(s["output"],     s["cache_miss"]) if s["cache_miss"] > 0 else "n/a"
        print(f"  {model:<22} {hit_rate:>10}  {paid_rate:>10}  {out_ratio:>14}")

    print(f"{'─'*68}")
    print("  HitRate  = tokens served from cache / total input  (higher → cheaper)")
    print("  PaidRate = tokens billed at full price / total input (lower → cheaper)")
    print("  Output/Input(paid) = output tokens per paid input token (higher → more work per ¥)")
    print(f"{'─'*68}")
