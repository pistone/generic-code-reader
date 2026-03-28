#!/usr/bin/env python3
"""Show knowledge base status: what's indexed, cost so far, service health.

Usage:
    python status.py
"""

import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")


def _read_cost_log(path: Path) -> tuple[float, int, str]:
    """Read a cost_log.jsonl and return (total_tokens, num_runs, last_run_time)."""
    total_tokens = 0
    num_runs = 0
    last_time = ""
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                total_tokens += entry.get("total_tokens", 0)
                num_runs += 1
                last_time = entry.get("timestamp", last_time)
            except json.JSONDecodeError:
                continue
    return total_tokens, num_runs, last_time


def _count_summaries(path: Path) -> int:
    """Count entries in a summaries JSON file."""
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def _count_manifest(path: Path) -> int:
    """Count entries in a hash manifest."""
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        return len(data) if isinstance(data, dict) else 0
    except Exception:
        return 0


def _time_ago(iso_str: str) -> str:
    """Convert ISO timestamp to 'X ago' string."""
    if not iso_str:
        return "never"
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now(dt.tzinfo)
        delta = now - dt
        if delta.days > 0:
            return f"{delta.days}d ago"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = delta.seconds // 60
        return f"{minutes}m ago"
    except Exception:
        return iso_str


def _check_r2r() -> str:
    """Check R2R health."""
    try:
        req = urllib.request.Request(f"{R2R_URL}/v3/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return "healthy"
            return f"status {resp.status}"
    except Exception:
        return "unreachable"


def main():
    print(f"\n{'='*50}")
    print(f"  Knowledge Base Status")
    print(f"{'='*50}")

    # Code analysis
    indexer_dir = PROJECT_DIR / "indexer"
    code_chunks = _count_summaries(indexer_dir / "summaries.json")
    code_files = _count_manifest(indexer_dir / "file_hashes.json")
    code_tokens, code_runs, code_last = _read_cost_log(indexer_dir / "cost_log.jsonl")

    # Module map
    module_map_path = indexer_dir / "module_map.json"
    modules = 0
    if module_map_path.exists():
        try:
            mm = json.loads(module_map_path.read_text())
            modules = len(mm.get("modules", []))
        except Exception:
            pass

    print(f"\n  Code:     {code_chunks:,} chunks from {code_files:,} files "
          f"({modules} modules)")
    print(f"            Last run: {_time_ago(code_last)} "
          f"({code_runs} total runs)")

    # Docs
    doc_dir = PROJECT_DIR / "doc_agent"
    doc_files = _count_manifest(doc_dir / "doc_hashes.json")
    doc_tokens, doc_runs, doc_last = _read_cost_log(doc_dir / "cost_log.jsonl")
    print(f"  Docs:     {doc_files:,} files indexed"
          f"  (last: {_time_ago(doc_last)})")

    # Tickets
    ticket_dir = PROJECT_DIR / "ticket_agent"
    ticket_entries = _count_summaries(ticket_dir / "ticket_summaries.json")
    ticket_files = _count_manifest(ticket_dir / "ticket_hashes.json")
    ticket_tokens, ticket_runs, ticket_last = _read_cost_log(ticket_dir / "cost_log.jsonl")
    print(f"  Tickets:  {ticket_entries:,} entries from {ticket_files:,} tickets"
          f"  (last: {_time_ago(ticket_last)})")

    # Staging queue
    staging_path = PROJECT_DIR / "mcp_server" / "staging_queue.json"
    pending = 0
    if staging_path.exists():
        try:
            q = json.loads(staging_path.read_text())
            pending = sum(1 for e in q if e.get("status") == "pending")
        except Exception:
            pass
    if pending:
        print(f"  Pending:  {pending} suggestions awaiting review")

    # Cost
    total_tokens = code_tokens + doc_tokens + ticket_tokens
    # Add auditor + reviewer
    for agent_dir in ["auditor", "reviewer"]:
        t, _, _ = _read_cost_log(PROJECT_DIR / agent_dir / "cost_log.jsonl")
        total_tokens += t

    print(f"\n  Tokens:   {total_tokens:,} total across all agents")

    # Query log
    query_log = PROJECT_DIR / "mcp_server" / "query_log.jsonl"
    query_count = 0
    if query_log.exists():
        query_count = sum(1 for _ in query_log.read_text().splitlines() if _.strip())
    if query_count:
        print(f"  Queries:  {query_count:,} searches logged")

    # Services
    r2r_status = _check_r2r()
    mcp_configured = (PROJECT_DIR / ".mcp.json").exists()

    print(f"\n  R2R:      {r2r_status} ({R2R_URL})")
    print(f"  MCP:      {'configured' if mcp_configured else 'not configured'} "
          f"(.mcp.json)")

    # Conflicts
    conflict_path = PROJECT_DIR / "auditor" / "conflict_report.json"
    if conflict_path.exists():
        try:
            conflicts = json.loads(conflict_path.read_text())
            n = len(conflicts) if isinstance(conflicts, list) else 0
            if n:
                print(f"  Conflicts: {n} doc/code conflicts found (auditor/conflict_report.json)")
        except Exception:
            pass

    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    main()
