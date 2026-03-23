#!/usr/bin/env python3
"""Ticket agent: extract reusable knowledge from Jira/PR exports into R2R.

Expects pre-exported JSON ticket files on disk (one ticket per file, or a
JSON array of tickets).  Filters aggressively, then uses LLM to extract
only the knowledge nuggets worth indexing.

Usage:
    python -m ticket_agent.ticket_agent --tickets /path/to/exported/tickets
    python -m ticket_agent.ticket_agent --tickets /tmp/tickets --incremental
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from r2r import R2RClient

# Shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import TokenTracker, llm_call, load_manifest, save_manifest  # noqa: E402

R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")
DEDUP_THRESHOLD = 0.85

RESOLVED_STATUSES = {"done", "closed", "resolved", "fixed", "complete"}
REJECT_RESOLUTIONS = {"won't fix", "wontfix", "duplicate", "cannot reproduce",
                      "incomplete", "not a bug"}


# ---------------------------------------------------------------------------
# Step 1: Load tickets
# ---------------------------------------------------------------------------

def load_tickets(tickets_dir: Path) -> list[dict]:
    """Load ticket JSON files.  Each file is a single ticket or list."""
    tickets: list[dict] = []
    for p in sorted(tickets_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] {p.name}: {e}")
            continue
        if isinstance(data, list):
            tickets.extend(data)
        elif isinstance(data, dict):
            tickets.append(data)
    return tickets


# ---------------------------------------------------------------------------
# Step 2: Structural filter
# ---------------------------------------------------------------------------

def structural_filter(tickets: list[dict]) -> list[dict]:
    """Filter to tickets likely to contain reusable knowledge."""
    passed: list[dict] = []
    for t in tickets:
        status = (t.get("status") or "").lower().strip()
        resolution = (t.get("resolution") or "").lower().strip()
        comments = t.get("comments") or []

        if status and status not in RESOLVED_STATUSES:
            continue
        if resolution in REJECT_RESOLUTIONS:
            continue
        if len(comments) < 2:
            continue

        passed.append(t)
    return passed


# ---------------------------------------------------------------------------
# Step 3: Build LLM context
# ---------------------------------------------------------------------------

def build_ticket_context(ticket: dict, max_chars: int = 3000) -> str:
    """Compose a condensed view of a ticket for LLM extraction."""
    parts: list[str] = []

    key = ticket.get("key", "unknown")
    title = ticket.get("title", "")
    parts.append(f"Ticket: {key} — {title}")

    desc = (ticket.get("description") or "")[:500]
    if desc:
        parts.append(f"\nDescription:\n{desc}")

    resolution = ticket.get("resolution", "")
    if resolution:
        parts.append(f"\nResolution: {resolution}")

    linked = ticket.get("linked_prs") or []
    if linked:
        parts.append(f"Linked PRs: {', '.join(str(p) for p in linked)}")

    labels = ticket.get("labels") or []
    if labels:
        parts.append(f"Labels: {', '.join(labels)}")

    comments = ticket.get("comments") or []
    if comments:
        # Take last 5 comments, cap each at 300 chars
        recent = comments[-5:]
        parts.append("\nRecent comments:")
        for c in recent:
            author = c.get("author", "?")
            body = (c.get("body") or "")[:300]
            parts.append(f"  [{author}]: {body}")

    text = "\n".join(parts)
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Step 4: LLM extraction
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You extract reusable technical knowledge from resolved tickets. "
    "If the ticket contains a root cause, workaround, design decision, "
    "or recurring pattern worth remembering, extract it as a concise summary. "
    "If the ticket is just a routine bug fix with no reusable insight, "
    "mark it as not useful. "
    "Output ONLY valid JSON — no markdown fences, no commentary."
)

EXTRACT_PROMPT_TEMPLATE = """{context}

Extract the reusable technical knowledge from this ticket.

Output this JSON:
{{
  "useful": true or false,
  "summary": "1-3 sentence knowledge nugget (only if useful)",
  "category": "root_cause" or "workaround" or "design_decision" or "pattern" (only if useful)
}}

If there's no reusable insight (routine fix, config change, typo), set useful to false."""


def extract_knowledge(model: str, context: str,
                      tracker: Optional[TokenTracker] = None) -> dict:
    """LLM call to extract knowledge from a ticket."""
    prompt = EXTRACT_PROMPT_TEMPLATE.format(context=context)
    raw = llm_call(model, EXTRACT_SYSTEM, prompt,
                   max_tokens=256, json_mode=True,
                   tracker=tracker, phase="Extraction")
    try:
        result = json.loads(raw)
        return {
            "useful": bool(result.get("useful", False)),
            "summary": str(result.get("summary", "")),
            "category": str(result.get("category", "general")),
        }
    except (json.JSONDecodeError, TypeError):
        return {"useful": False, "summary": "", "category": ""}


# ---------------------------------------------------------------------------
# Step 5: Dedup against KB
# ---------------------------------------------------------------------------

def dedup_against_kb(client: R2RClient, summary: str,
                     threshold: float = DEDUP_THRESHOLD) -> bool:
    """Return True if a similar entry already exists in R2R."""
    if not summary:
        return False
    try:
        results = client.retrieval.search(
            query=summary[:200],
            search_settings={"limit": 1, "use_hybrid_search": True},
        )
        hits = results.results.chunk_search_results
        if hits and getattr(hits[0], "score", 0) > threshold:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Step 6: Index to R2R
# ---------------------------------------------------------------------------

def index_ticket_summaries(entries: list[dict]) -> dict[str, str]:
    """Index extracted ticket knowledge into R2R.

    Returns a dict mapping ticket key → doc_id for successfully indexed entries.
    Failed entries are omitted from the result (no misaligned zips).
    """
    client = R2RClient(R2R_URL)
    indexed: dict[str, str] = {}

    for i, entry in enumerate(entries):
        try:
            resp = client.documents.create(
                raw_text=entry["summary"],
                metadata={
                    "source_file":   entry["key"],
                    "module":        entry["category"],
                    "chunk_type":    "ticket_summary",
                    "source_type":   "ticket",
                    "doc_title":     entry["title"],
                    "last_modified": entry.get("resolved", ""),
                },
            )
            indexed[entry["key"]] = str(resp.results.document_id)
        except Exception as e:
            print(f"  [warn] {entry['key']}: index failed: {e}")

        if i > 0 and i % 20 == 0:
            time.sleep(0.5)

    return indexed


# ---------------------------------------------------------------------------
# Incremental mode
# ---------------------------------------------------------------------------

def ticket_hash(ticket: dict) -> str:
    """Hash based on content that would change the extraction."""
    parts = [
        ticket.get("title", ""),
        (ticket.get("description") or "")[:500],
        str(len(ticket.get("comments") or [])),
        ticket.get("resolution", ""),
        ticket.get("status", ""),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# load_manifest / save_manifest imported from shared.utils


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ticket agent: extract knowledge from ticket exports → R2R",
    )
    parser.add_argument("--tickets", required=True,
                        help="Directory of exported ticket JSON files")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="litellm model string (default: %(default)s)")
    parser.add_argument("--incremental", action="store_true",
                        help="Only process tickets changed since last run")
    args = parser.parse_args()

    tickets_dir = Path(args.tickets).resolve()
    if not tickets_dir.is_dir():
        print(f"Error: '{tickets_dir}' is not a directory")
        sys.exit(1)

    output_dir = Path(__file__).resolve().parent
    manifest_path = output_dir / "ticket_hashes.json"
    cost_log_path = output_dir / "cost_log.jsonl"

    tracker = TokenTracker()
    manifest = load_manifest(manifest_path)

    # Step 1: load
    all_tickets = load_tickets(tickets_dir)
    print(f"Loaded {len(all_tickets)} ticket(s) from {tickets_dir}")
    if not all_tickets:
        return

    # Step 2: structural filter
    candidates = structural_filter(all_tickets)
    print(f"Structural filter: {len(candidates)}/{len(all_tickets)} passed")
    if not candidates:
        print("No tickets passed the structural filter.")
        return

    # Incremental filter
    if args.incremental:
        changed: list[dict] = []
        for t in candidates:
            key = t.get("key", "")
            h = ticket_hash(t)
            if manifest.get(key, {}).get("hash") != h:
                changed.append(t)
        if not changed:
            print("[Incremental] No tickets changed since last run.")
            return
        print(f"[Incremental] {len(changed)}/{len(candidates)} tickets changed")
        candidates = changed

    # Steps 3-4: extract knowledge
    client = R2RClient(R2R_URL)
    useful_entries: list[dict] = []
    not_useful = 0

    print(f"\nExtracting knowledge from {len(candidates)} ticket(s)...")
    for t in candidates:
        key = t.get("key", "unknown")
        context = build_ticket_context(t)
        result = extract_knowledge(args.model, context, tracker=tracker)

        if not result["useful"]:
            not_useful += 1
            print(f"  {key}: not useful")
        else:
            # Step 5: dedup
            if dedup_against_kb(client, result["summary"]):
                print(f"  {key}: duplicate (skipped)")
            else:
                useful_entries.append({
                    "key": key,
                    "title": t.get("title", ""),
                    "summary": result["summary"],
                    "category": result["category"],
                    "resolved": t.get("resolved", ""),
                })
                print(f"  {key}: [{result['category']}] {result['summary'][:80]}...")

        time.sleep(0.3)  # rate limit

    print(f"\nExtraction: {len(useful_entries)} useful, "
          f"{not_useful} not useful, "
          f"{len(candidates) - len(useful_entries) - not_useful} duplicates")

    # Step 6: index
    if useful_entries:
        print(f"\nIndexing {len(useful_entries)} ticket summaries into R2R...")
        indexed = index_ticket_summaries(useful_entries)

        # Update manifest with indexed tickets
        for entry in useful_entries:
            key = entry["key"]
            doc_id = indexed.get(key)
            if doc_id is None:
                continue  # indexing failed for this entry
            # Find the original ticket to compute hash
            orig = next((t for t in candidates if t.get("key") == key), {})
            manifest[key] = {
                "hash": ticket_hash(orig),
                "doc_id": doc_id,
            }
    else:
        print("\nNo new knowledge to index.")

    # Mark non-useful tickets in manifest too (so we don't re-process them)
    for t in candidates:
        key = t.get("key", "")
        if key and key not in manifest:
            manifest[key] = {"hash": ticket_hash(t), "doc_id": None}

    # Save manifest
    save_manifest(manifest_path, manifest)

    # Token tracking
    if tracker.phases:
        print(f"\n{tracker.summary()}")
        entry = tracker.to_log_entry(model=args.model, agent="ticket_agent")
        with cost_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[Done] {len(useful_entries)} ticket knowledge entries indexed")


if __name__ == "__main__":
    main()
