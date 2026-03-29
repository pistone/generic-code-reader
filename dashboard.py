#!/usr/bin/env python3
"""Token Savings Dashboard — measure and display KB ROI.

Reads query_log.jsonl (MCP queries served) and cost_log.jsonl (indexing
costs) to show how much the KB is saving compared to raw file reads.

Usage:
    python dashboard.py                     # terminal summary
    python dashboard.py --period 7d         # last 7 days only
    python dashboard.py --json              # machine-readable output
    python dashboard.py --detail            # per-query breakdown
    python dashboard.py --by-module         # savings by module
    python dashboard.py --by-developer      # savings by developer (if tagged)
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codebase_shared.colors import bold, cyan, dim, green, ok, red, warn, yellow

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
QUERY_LOG = PROJECT_DIR / "mcp_server" / "query_log.jsonl"
COST_LOGS = [
    ("study_agent", PROJECT_DIR / "indexer" / "cost_log.jsonl"),
    ("doc_agent", PROJECT_DIR / "doc_agent" / "cost_log.jsonl"),
    ("ticket_agent", PROJECT_DIR / "ticket_agent" / "cost_log.jsonl"),
    ("auditor", PROJECT_DIR / "auditor" / "cost_log.jsonl"),
    ("reviewer", PROJECT_DIR / "reviewer" / "cost_log.jsonl"),
]

# ── Cost model ────────────────────────────────────────────────────────────────
# Estimated token costs per 1M tokens (USD) for common models.
# Used for dollar estimates. Override with --input-cost / --output-cost.

MODEL_COSTS = {
    # (input_per_1M, output_per_1M)
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
    "claude-haiku": (0.25, 1.25),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "default": (3.00, 15.00),  # Sonnet-class as default
}

# ── Estimation model ─────────────────────────────────────────────────────────
# How many tokens would the LLM have used WITHOUT the KB?
#
# Based on typical Claude Code behavior when answering codebase questions:
#   1. grep/search for relevant files (tool call + results)    ~500 tokens
#   2. Read file 1 (often wrong or partial match)             ~3000 tokens
#   3. Read file 2 (better match)                             ~3000 tokens
#   4. Sometimes read file 3                                  ~2000 tokens
#   5. LLM reasoning + response                              ~1500 tokens
#   Total: ~10,000-15,000 tokens per question
#
# Conservative estimate: 12,000 tokens per question without KB.
# With KB: actual tokens from query_log (typically 500-2000).

WITHOUT_KB_ESTIMATE = 12_000  # tokens per question without KB
FILE_READ_ESTIMATE = 3_000    # tokens per file read
SEARCH_ESTIMATE = 500         # tokens per grep/search operation


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp, handling various formats."""
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        # Fallback: try without timezone
        try:
            dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping bad lines."""
    entries = []
    if not path.exists():
        return entries
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def load_queries(since: datetime | None = None,
                 query_log: Path | None = None) -> list[dict]:
    """Load query log entries, optionally filtered by time."""
    entries = _load_jsonl(query_log or QUERY_LOG)
    if since:
        filtered = []
        for e in entries:
            ts = e.get("timestamp", "")
            if ts and _parse_timestamp(ts) >= since:
                filtered.append(e)
        return filtered
    return entries


def load_indexing_costs() -> dict[str, dict]:
    """Load indexing costs from all agents' cost logs."""
    costs = {}
    for agent_name, path in COST_LOGS:
        entries = _load_jsonl(path)
        total_tokens = 0
        total_prompt = 0
        total_completion = 0
        num_runs = 0
        for e in entries:
            total_tokens += e.get("total_tokens", 0)
            total_prompt += e.get("prompt_tokens", 0)
            total_completion += e.get("completion_tokens", 0)
            num_runs += 1
        if total_tokens > 0:
            costs[agent_name] = {
                "total_tokens": total_tokens,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "runs": num_runs,
            }
    return costs


# ── Estimation ────────────────────────────────────────────────────────────────

def estimate_without_kb(query: dict) -> int:
    """Estimate how many tokens this query would cost without the KB.

    Uses the number of result files as a proxy: if the KB returned results
    from 3 files, the LLM would likely have had to read those 3 files
    plus do some searching.
    """
    n_files = len(query.get("result_files", []))
    n_results = query.get("num_results", 0)

    if n_results == 0:
        # KB couldn't answer — LLM would still need to search
        return SEARCH_ESTIMATE + FILE_READ_ESTIMATE * 2

    # Conservative: LLM would read each file the KB referenced + search cost
    file_tokens = max(n_files, 2) * FILE_READ_ESTIMATE
    return SEARCH_ESTIMATE + file_tokens + 500  # 500 for reasoning overhead


def estimate_with_kb(query: dict) -> int:
    """Actual token cost of the KB answer."""
    return query.get("approx_tokens", 500) + 200  # 200 for MCP call overhead


# ── Dollar calculations ───────────────────────────────────────────────────────

def tokens_to_dollars(tokens: int, model: str = "default",
                      direction: str = "mixed") -> float:
    """Convert token count to estimated dollar cost.

    direction: "input", "output", or "mixed" (60/40 split)
    """
    costs = MODEL_COSTS.get(model, MODEL_COSTS["default"])
    input_rate, output_rate = costs

    if direction == "input":
        return tokens * input_rate / 1_000_000
    elif direction == "output":
        return tokens * output_rate / 1_000_000
    else:
        # Mixed: assume 60% input, 40% output (typical for Q&A)
        input_cost = tokens * 0.6 * input_rate / 1_000_000
        output_cost = tokens * 0.4 * output_rate / 1_000_000
        return input_cost + output_cost


# ── Reporting ─────────────────────────────────────────────────────────────────

def compute_stats(queries: list[dict], model: str = "default") -> dict:
    """Compute all dashboard statistics."""
    if not queries:
        return {
            "period_queries": 0,
            "total_with_kb": 0,
            "total_without_kb": 0,
            "tokens_saved": 0,
            "dollars_saved": 0.0,
            "dollars_with_kb": 0.0,
            "dollars_without_kb": 0.0,
            "avg_with_kb": 0,
            "avg_without_kb": 0,
            "reduction_pct": 0.0,
            "zero_result_queries": 0,
            "cache_hit_rate": 0.0,
            "by_module": {},
            "by_scope": {},
            "by_hour": defaultdict(int),
            "top_queries": [],
        }

    total_with = 0
    total_without = 0
    zero_results = 0
    by_module = defaultdict(lambda: {"with": 0, "without": 0, "count": 0})
    by_scope = defaultdict(lambda: {"with": 0, "without": 0, "count": 0})
    by_hour = defaultdict(int)
    query_counts = defaultdict(int)

    for q in queries:
        w = estimate_with_kb(q)
        wo = estimate_without_kb(q)
        total_with += w
        total_without += wo

        if q.get("num_results", 0) == 0:
            zero_results += 1

        # By module
        module = q.get("module", "") or "all"
        by_module[module]["with"] += w
        by_module[module]["without"] += wo
        by_module[module]["count"] += 1

        # By scope
        scope = q.get("scope", "") or "general"
        by_scope[scope]["with"] += w
        by_scope[scope]["without"] += wo
        by_scope[scope]["count"] += 1

        # By hour of day
        ts = q.get("timestamp", "")
        if ts:
            try:
                hour = _parse_timestamp(ts).hour
                by_hour[hour] += 1
            except Exception:
                pass

        # Query frequency
        query_text = q.get("query", "")[:80]
        if query_text:
            query_counts[query_text] += 1

    tokens_saved = total_without - total_with
    reduction_pct = (tokens_saved / total_without * 100) if total_without > 0 else 0
    answerable = len(queries) - zero_results
    cache_hit_rate = (answerable / len(queries) * 100) if queries else 0

    # Top repeated queries
    top_queries = sorted(query_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "period_queries": len(queries),
        "total_with_kb": total_with,
        "total_without_kb": total_without,
        "tokens_saved": tokens_saved,
        "dollars_saved": tokens_to_dollars(tokens_saved, model),
        "dollars_with_kb": tokens_to_dollars(total_with, model),
        "dollars_without_kb": tokens_to_dollars(total_without, model),
        "avg_with_kb": total_with // len(queries) if queries else 0,
        "avg_without_kb": total_without // len(queries) if queries else 0,
        "reduction_pct": reduction_pct,
        "zero_result_queries": zero_results,
        "cache_hit_rate": cache_hit_rate,
        "by_module": dict(by_module),
        "by_scope": dict(by_scope),
        "by_hour": dict(by_hour),
        "top_queries": top_queries,
    }


def compute_indexing_roi(indexing_costs: dict, stats: dict,
                         model: str = "default") -> dict:
    """Compute ROI: indexing cost vs cumulative savings."""
    total_index_tokens = sum(
        c["total_tokens"] for c in indexing_costs.values()
    )
    index_dollars = tokens_to_dollars(total_index_tokens, model)
    saved_dollars = stats["dollars_saved"]

    if index_dollars > 0:
        roi_ratio = saved_dollars / index_dollars
        queries_to_break_even = 0
        if stats["period_queries"] > 0:
            savings_per_query = saved_dollars / stats["period_queries"]
            if savings_per_query > 0:
                queries_to_break_even = int(index_dollars / savings_per_query)
    else:
        roi_ratio = 0
        queries_to_break_even = 0

    return {
        "index_tokens": total_index_tokens,
        "index_dollars": index_dollars,
        "saved_dollars": saved_dollars,
        "roi_ratio": roi_ratio,
        "queries_to_break_even": queries_to_break_even,
        "by_agent": {
            name: {
                "tokens": c["total_tokens"],
                "dollars": tokens_to_dollars(c["total_tokens"], model),
                "runs": c["runs"],
            }
            for name, c in indexing_costs.items()
        },
    }


# ── Terminal display ──────────────────────────────────────────────────────────

def _bar(value: float, max_value: float, width: int = 30) -> str:
    """Render a simple horizontal bar."""
    if max_value <= 0:
        return ""
    filled = int(value / max_value * width)
    filled = min(filled, width)
    return green("█" * filled) + dim("░" * (width - filled))


def _fmt_tokens(n: int) -> str:
    """Format token count human-readably."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_dollars(d: float) -> str:
    if d >= 100:
        return f"${d:,.0f}"
    if d >= 1:
        return f"${d:.2f}"
    return f"${d:.4f}"


def print_dashboard(stats: dict, roi: dict, period_label: str,
                    model: str, show_detail: bool = False,
                    show_by_module: bool = False):
    """Print the full dashboard to terminal."""

    w = 56  # box width

    # ── Header ──
    print()
    print(bold(f"┌{'─' * w}┐"))
    print(bold(f"│{'Token Savings Dashboard':^{w}}│"))
    print(bold(f"│{period_label:^{w}}│"))
    print(bold(f"├{'─' * w}┤"))

    # ── Savings summary ──
    if stats["period_queries"] == 0:
        print(f"│{'':^{w}}│")
        print(f"│{yellow('  No queries logged yet.'):^{w+9}}│")
        print(f"│{'  Use the MCP server, then check again.':^{w}}│")
        print(f"│{'':^{w}}│")
        print(bold(f"└{'─' * w}┘"))
        print()
        return

    print(f"│{'':^{w}}│")

    # Queries
    q_line = f"  Queries served:     {bold(str(stats['period_queries']))}"
    print(f"│{q_line:<{w + 4}}│")

    zero_line = f"  Zero-result queries: {stats['zero_result_queries']}"
    hit_rate = f"({stats['cache_hit_rate']:.0f}% answerable)"
    print(f"│  {zero_line}  {dim(hit_rate):<{w - len(zero_line) + 3}}│")

    print(f"│{'':^{w}}│")

    # Token comparison
    print(f"│  {bold('Token Usage Comparison'):<{w + 2}}│")
    print(f"│{'':^{w}}│")

    wo_line = f"  Without KB:  {_fmt_tokens(stats['total_without_kb']):>8}  " \
              f"({_fmt_dollars(stats['dollars_without_kb'])})"
    print(f"│{wo_line:<{w}}│")

    w_line = f"  With KB:     {_fmt_tokens(stats['total_with_kb']):>8}  " \
             f"({_fmt_dollars(stats['dollars_with_kb'])})"
    print(f"│{w_line:<{w}}│")

    print(f"│{'':^{w}}│")

    saved_line = f"  {ok('Saved')}:        " \
                 f"{green(_fmt_tokens(stats['tokens_saved']))}  " \
                 f"({green(_fmt_dollars(stats['dollars_saved']))})"
    # ANSI codes add length; pad manually
    print(f"│{saved_line}{'':>{w - 40}}│")

    # Reduction bar
    pct = stats["reduction_pct"]
    bar = _bar(pct, 100, 30)
    pct_str = f"{pct:.0f}% reduction"
    print(f"│  {bar} {pct_str:<{w - 35}}│")

    print(f"│{'':^{w}}│")

    # Per-query averages
    print(f"│  {bold('Per-Query Average'):<{w + 2}}│")
    avg_wo = f"  Without KB:  ~{_fmt_tokens(stats['avg_without_kb'])} tokens/query"
    avg_w = f"  With KB:     ~{_fmt_tokens(stats['avg_with_kb'])} tokens/query"
    print(f"│{avg_wo:<{w}}│")
    print(f"│{avg_w:<{w}}│")

    print(f"│{'':^{w}}│")
    print(bold(f"├{'─' * w}┤"))

    # ── ROI ──
    print(f"│  {bold('Indexing ROI'):<{w + 2}}│")
    print(f"│{'':^{w}}│")

    if roi["index_tokens"] > 0:
        idx_line = f"  Index cost:    {_fmt_tokens(roi['index_tokens'])} tokens  " \
                   f"({_fmt_dollars(roi['index_dollars'])})"
        print(f"│{idx_line:<{w}}│")

        for agent_name, ac in roi["by_agent"].items():
            a_line = f"    {agent_name:<14} {_fmt_tokens(ac['tokens']):>7}  " \
                     f"({_fmt_dollars(ac['dollars'])}, {ac['runs']} runs)"
            print(f"│{a_line:<{w}}│")

        print(f"│{'':^{w}}│")

        sav_line = f"  Saved so far:  {_fmt_dollars(roi['saved_dollars'])}"
        print(f"│{sav_line:<{w}}│")

        if roi["roi_ratio"] >= 1.0:
            roi_str = ok(f"  ROI: {roi['roi_ratio']:.1f}x — KB has paid for itself!")
            print(f"│{roi_str}{'':>{w - 47}}│")
        else:
            roi_str = f"  ROI: {roi['roi_ratio']:.2f}x"
            be_str = f"  Break-even at ~{roi['queries_to_break_even']} queries"
            print(f"│{yellow(roi_str):<{w + 9}}│")
            print(f"│{be_str:<{w}}│")
    else:
        print(f"│  {dim('No indexing cost data found.'):<{w + 4}}│")

    print(f"│{'':^{w}}│")
    print(bold(f"├{'─' * w}┤"))

    # ── Projections ──
    print(f"│  {bold('Projections'):<{w + 2}}│")
    print(f"│{'':^{w}}│")

    if stats["period_queries"] > 0:
        # Daily rate (assume the period covers actual days of usage)
        q_per_day = stats["period_queries"]  # Will be adjusted below
        savings_per_q = stats["tokens_saved"] / stats["period_queries"]
        dollar_per_q = stats["dollars_saved"] / stats["period_queries"]

        team_sizes = [5, 10, 20]
        queries_per_dev_per_day = 25

        for team in team_sizes:
            daily_q = team * queries_per_dev_per_day
            monthly_q = daily_q * 22  # work days
            monthly_savings = monthly_q * dollar_per_q
            m_line = f"  {team} devs × {queries_per_dev_per_day}q/day:" \
                     f"  {green(_fmt_dollars(monthly_savings))}/month saved"
            print(f"│{m_line}{'':>{w - len(m_line) + 13}}│")

    print(f"│{'':^{w}}│")

    # ── Top queries ──
    if stats["top_queries"]:
        print(bold(f"├{'─' * w}┤"))
        print(f"│  {bold('Most Frequent Queries'):<{w + 2}}│")
        print(f"│{'':^{w}}│")
        for query_text, count in stats["top_queries"][:7]:
            truncated = query_text[:42]
            q_line = f"  {count:>3}×  {truncated}"
            print(f"│{q_line:<{w}}│")

    print(f"│{'':^{w}}│")

    # ── By module ──
    if show_by_module and stats["by_module"]:
        print(bold(f"├{'─' * w}┤"))
        print(f"│  {bold('Savings by Module'):<{w + 2}}│")
        print(f"│{'':^{w}}│")

        # Sort by savings descending
        modules = sorted(
            stats["by_module"].items(),
            key=lambda x: x[1]["without"] - x[1]["with"],
            reverse=True,
        )
        for mod_name, mod_stats in modules[:10]:
            saved = mod_stats["without"] - mod_stats["with"]
            m_line = f"  {mod_name:<20} {mod_stats['count']:>4} queries  " \
                     f"saved {_fmt_tokens(saved)}"
            print(f"│{m_line:<{w}}│")

        print(f"│{'':^{w}}│")

    # ── By scope ──
    if stats["by_scope"] and len(stats["by_scope"]) > 1:
        print(bold(f"├{'─' * w}┤"))
        print(f"│  {bold('Savings by Query Scope'):<{w + 2}}│")
        print(f"│{'':^{w}}│")

        for scope_name, scope_stats in sorted(stats["by_scope"].items()):
            saved = scope_stats["without"] - scope_stats["with"]
            pct = (saved / scope_stats["without"] * 100) if scope_stats["without"] > 0 else 0
            s_line = f"  {scope_name:<18} {scope_stats['count']:>4} queries  " \
                     f"{pct:.0f}% reduction"
            print(f"│{s_line:<{w}}│")

        print(f"│{'':^{w}}│")

    # ── Footer ──
    model_note = f"Cost model: {model}"
    print(bold(f"├{'─' * w}┤"))
    print(f"│  {dim(model_note):<{w + 4}}│")
    print(f"│  {dim(f'Estimate: {WITHOUT_KB_ESTIMATE:,} tokens/query without KB'):<{w + 4}}│")
    print(bold(f"└{'─' * w}┘"))
    print()


def print_detail(queries: list[dict]):
    """Print per-query breakdown."""
    print(f"\n{'─' * 80}")
    print(bold("Per-Query Breakdown"))
    print(f"{'─' * 80}")
    print(f"{'#':>4}  {'Tokens':>10}  {'Saved':>10}  {'Results':>4}  Query")
    print(f"{'─' * 80}")

    for i, q in enumerate(queries):
        w = estimate_with_kb(q)
        wo = estimate_without_kb(q)
        saved = wo - w
        n = q.get("num_results", 0)
        query_text = q.get("query", "")[:50]
        saved_str = green(f"+{_fmt_tokens(saved)}") if saved > 0 else red(_fmt_tokens(saved))

        print(f"{i + 1:>4}  {_fmt_tokens(w):>10}  {_fmt_tokens(saved):>10}  "
              f"{n:>4}  {query_text}")

    print(f"{'─' * 80}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_period(period_str: str) -> tuple[datetime | None, str]:
    """Parse period string like '7d', '30d', '24h', 'all'."""
    if not period_str or period_str == "all":
        return None, "All time"

    unit = period_str[-1].lower()
    try:
        value = int(period_str[:-1])
    except ValueError:
        print(f"Invalid period: {period_str}. Use e.g. '7d', '30d', '24h'")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    if unit == "d":
        since = now - timedelta(days=value)
        label = f"Last {value} day{'s' if value != 1 else ''}"
    elif unit == "h":
        since = now - timedelta(hours=value)
        label = f"Last {value} hour{'s' if value != 1 else ''}"
    elif unit == "w":
        since = now - timedelta(weeks=value)
        label = f"Last {value} week{'s' if value != 1 else ''}"
    else:
        print(f"Invalid period unit: {unit}. Use d(ays), h(ours), w(eeks)")
        sys.exit(1)

    return since, label


def main():
    parser = argparse.ArgumentParser(
        description="Token Savings Dashboard — measure KB ROI"
    )
    parser.add_argument("--period", default="all",
                        help="Time period: 7d, 30d, 24h, 1w, all (default: all)")
    parser.add_argument("--model", default="default",
                        choices=list(MODEL_COSTS.keys()),
                        help="Cost model for dollar estimates")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of terminal display")
    parser.add_argument("--detail", action="store_true",
                        help="Show per-query breakdown")
    parser.add_argument("--by-module", action="store_true",
                        help="Show savings broken down by module")
    parser.add_argument("--query-log", default=None,
                        help=f"Path to query_log.jsonl (default: {QUERY_LOG})")

    args = parser.parse_args()

    query_log_path = Path(args.query_log) if args.query_log else QUERY_LOG

    # Parse period
    since, period_label = parse_period(args.period)

    # Load data
    queries = load_queries(since, query_log_path)
    indexing_costs = load_indexing_costs()

    # Compute stats
    stats = compute_stats(queries, args.model)
    roi = compute_indexing_roi(indexing_costs, stats, args.model)

    if args.json:
        output = {
            "period": period_label,
            "model": args.model,
            "stats": stats,
            "roi": roi,
        }
        # Clean up defaultdicts for JSON
        output["stats"]["by_module"] = dict(output["stats"]["by_module"])
        output["stats"]["by_scope"] = dict(output["stats"]["by_scope"])
        output["stats"]["by_hour"] = dict(output["stats"]["by_hour"])
        print(json.dumps(output, indent=2, default=str))
    else:
        print_dashboard(stats, roi, period_label, args.model,
                        show_by_module=args.by_module)
        if args.detail:
            print_detail(queries)


if __name__ == "__main__":
    main()
