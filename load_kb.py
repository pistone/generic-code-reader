#!/usr/bin/env python3
"""
Load all KB data from JSON files into a local ChromaDB.

Usage:
    # Load from default locations (indexer/, doc_agent/, ticket_agent/)
    python load_kb.py

    # Load from a KB directory (e.g. a git-shared kb/ folder)
    python load_kb.py --kb-dir /path/to/kb

    # Specify output ChromaDB location
    python load_kb.py --db-dir ./my_chroma_db

    # Force full reload (clear existing data first)
    python load_kb.py --clean
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codebase_shared.local_kb import LocalKB  # noqa: E402


def _load_json(path: Path) -> list:
    """Load a JSON array from a file. Returns [] if missing or invalid."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, Exception):
        return []


def _load_jsonl(path: Path) -> list:
    """Load a JSONL file (one JSON object per line). Returns [] if missing."""
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Load KB data from JSON files into local ChromaDB")
    parser.add_argument("--kb-dir", default=None,
                        help="Directory containing JSON files to load. "
                             "If not set, looks in indexer/, doc_agent/, ticket_agent/")
    parser.add_argument("--db-dir", default="chroma_db",
                        help="ChromaDB storage directory (default: ./chroma_db)")
    parser.add_argument("--clean", action="store_true",
                        help="Clear existing data before loading")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent

    # Determine source files
    if args.kb_dir:
        kb = Path(args.kb_dir)
        sources = {
            "summaries":      kb / "summaries.json",
            "doc_summaries":  kb / "doc_summaries.json",
            "tickets":        kb / "ticket_summaries.json",
            "user":           kb / "user_contributed.jsonl",
        }
    else:
        sources = {
            "summaries":      base / "indexer" / "summaries.json",
            "doc_summaries":  base / "doc_agent" / "doc_summaries.json",
            "tickets":        base / "ticket_agent" / "ticket_summaries.json",
            "user":           base / "mcp_server" / "user_contributed.jsonl",
        }

    print(f"Loading KB into {args.db_dir}")
    kb = LocalKB(args.db_dir)

    if args.clean:
        print("[Clean] Clearing existing data...")
        kb.clear()

    total = 0

    # Code summaries
    summaries = _load_json(sources["summaries"])
    if summaries:
        print(f"\n[Code] Loading {len(summaries)} entries from {sources['summaries'].name}")
        n = kb.add_entries(summaries, source_type="code")
        print(f"[Code] {n} entries indexed")
        total += n
    else:
        print(f"[Code] No summaries found at {sources['summaries']}")

    # Doc summaries
    doc_chunks = _load_json(sources["doc_summaries"])
    if doc_chunks:
        print(f"\n[Docs] Loading {len(doc_chunks)} entries from {sources['doc_summaries'].name}")
        n = kb.add_doc_chunks(doc_chunks)
        print(f"[Docs] {n} entries indexed")
        total += n
    else:
        print(f"[Docs] No doc summaries found at {sources['doc_summaries']}")

    # Ticket summaries
    tickets = _load_json(sources["tickets"])
    if tickets:
        print(f"\n[Tickets] Loading {len(tickets)} entries from {sources['tickets'].name}")
        n = kb.add_ticket_entries(tickets)
        print(f"[Tickets] {n} entries indexed")
        total += n
    else:
        print(f"[Tickets] No ticket summaries found at {sources['tickets']}")

    # User contributed (JSONL) — canonical source for user additions.
    # staging_queue.json is a subset (log_user_entry writes to both),
    # so we only load the JSONL to avoid double-counting.
    user_entries = _load_jsonl(sources["user"])
    if user_entries:
        print(f"\n[User] Loading {len(user_entries)} entries from {sources['user'].name}")
        n = kb.add_user_entries(user_entries)
        print(f"[User] {n} entries indexed")
        total += n

    print(f"\n{'='*50}")
    print(f"  Done: {total} total entries in local KB")
    print(f"  DB location: {args.db_dir}")
    stats = kb.stats()
    print(f"  Code: {stats.get('code', 0)}  Docs: {stats.get('doc', 0)}  "
          f"Tickets: {stats.get('ticket', 0)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
