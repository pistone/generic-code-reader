"""
Reviewer Agent — verifies staged suggestions and promotes approved ones to R2R.

The self-improving loop:
  1. Developer asks a question → KB miss
  2. Claude researches manually (reads files, greps)
  3. Claude calls suggest_index_item() via MCP → entry lands in staging_queue.json
  4. This agent picks it up, verifies accuracy, checks for duplicates
  5. On approve/edit  → indexes into R2R (available to whole team instantly)
  6. On reject        → logs to rejected_queue.json with reasoning

Usage:
  # Process all pending items once and exit
  python reviewer_agent.py --codebase /path/to/codebase

  # Watch mode: poll the staging queue every 30 s
  python reviewer_agent.py --codebase /path/to/codebase --watch

  # Different LLM provider (same as study_agent)
  python reviewer_agent.py --codebase /path/to/codebase --model ollama/llama3.1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from r2r import R2RClient

# Shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codebase_shared.utils import TokenTracker, llm_call, load_queue, save_queue  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = os.getenv("LLM_MODEL", "openai/gpt-4o")
R2R_URL         = os.getenv("R2R_URL", "http://localhost:7272")
WATCH_INTERVAL  = 30   # seconds between queue polls in --watch mode
MAX_FILE_CHARS  = 3000  # max chars to read per cited source file
KB_SEARCH_LIMIT = 5     # how many similar KB entries to fetch for dup check

BASE_DIR       = Path(__file__).parent
STAGING_FILE   = BASE_DIR.parent / "mcp_server" / "staging_queue.json"
REJECTED_FILE  = BASE_DIR / "rejected_queue.json"


# ── Pydantic model for LLM decision ───────────────────────────────────────────

class ReviewDecision(BaseModel):
    decision:      str   # "approve" | "edit" | "reject"
    final_summary: str   # corrected summary (same as original on approve/reject)
    reasoning:     str   # why this decision was made

# ── Helpers ───────────────────────────────────────────────────────────────────

# load_queue / save_queue imported from codebase_shared.utils


def llm_call_json(model: str, system: str, user: str, max_tokens: int = 1024,
                  tracker: Optional[TokenTracker] = None, phase: str = "") -> str:
    """Call LLM in JSON mode via shared llm_call, return raw text."""
    return llm_call(model, system, user, max_tokens=max_tokens,
                    json_mode=True, tracker=tracker, phase=phase)


def read_source_file(fpath: Path, max_chars: int = MAX_FILE_CHARS) -> str:
    """Read a source file, returning up to max_chars with a truncation note."""
    try:
        content = fpath.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            return content[:max_chars] + f"\n... [{len(content) - max_chars} more chars]"
        return content
    except Exception as e:
        return f"[could not read: {e}]"


def search_kb(query: str, limit: int = KB_SEARCH_LIMIT) -> list[dict]:
    """Search R2R for similar entries (for duplicate detection)."""
    try:
        client = R2RClient(R2R_URL)
        results = client.retrieval.search(
            query=query,
            search_settings={"limit": limit},
        )
        hits = results.results.chunk_search_results
        return [
            {
                "score":   round(getattr(hit, "score", 0), 3),
                "summary": (hit.text or "")[:300],
                "file":    (hit.metadata or {}).get("source_file", "unknown"),
            }
            for hit in hits
        ]
    except Exception as e:
        return [{"error": str(e)}]


def index_entry(entry: dict) -> tuple[str, str]:
    """Promote an approved entry into R2R. Returns (summary_doc_id, code_doc_id).
    Also indexes raw code as a separate searchable chunk."""
    client = R2RClient(R2R_URL)
    src_file = entry.get("source_file", "")
    module   = entry.get("module", "")

    # Index the summary
    response = client.documents.create(
        raw_text=entry["summary"],
        metadata={
            "source_file": src_file,
            "module":      module,
            "chunk_type":  entry.get("chunk_type", "function_summary"),
            "source_type": "code",
            "origin":      "suggestion",
        },
    )
    doc_id = str(response.results.document_id)

    # Index raw code as a separate searchable chunk
    code_doc_id = ""
    raw_code = entry.get("raw_code", "").strip()
    if raw_code and len(raw_code) > 30:
        try:
            resp = client.documents.create(
                raw_text=raw_code,
                metadata={
                    "source_file": src_file,
                    "module":      module,
                    "chunk_type":  "raw_code",
                    "source_type": "code",
                    "origin":      "suggestion",
                },
            )
            code_doc_id = str(resp.results.document_id)
        except Exception:
            pass  # non-fatal

    return doc_id, code_doc_id

# ── Reviewer prompt ───────────────────────────────────────────────────────────

REVIEWER_SYSTEM = (
    "You are a senior engineer reviewing a suggested knowledge-base entry. "
    "Your job: verify accuracy, check for duplicates, and approve/edit/reject. "
    "Output ONLY valid JSON — no markdown, no text outside the JSON object."
)

def build_review_prompt(entry: dict, file_contents: dict[str, str],
                        kb_hits: list[dict]) -> str:
    files_section = ""
    if file_contents:
        for fpath, content in file_contents.items():
            files_section += f"\n### {fpath}\n```\n{content}\n```\n"
    else:
        files_section = "(no source files provided)"

    kb_section = ""
    if kb_hits:
        for h in kb_hits:
            if "error" in h:
                kb_section += f"  [KB search failed: {h['error']}]\n"
            else:
                kb_section += f"  Score {h['score']} — {h['file']}: {h['summary']}\n"
    else:
        kb_section = "  (no similar entries found)"

    return f"""## Suggested KB Entry

Topic: {entry.get("topic", "(no topic)")}

Summary to verify:
{entry.get("summary", "")}

Source files cited:
{", ".join(entry.get("source_files", [])) or "(none)"}

Why suggested:
{entry.get("reasoning", "(no reasoning provided)")}

## Source File Contents
{files_section}

## Similar Entries Already in KB (duplicate check)
{kb_section}

## Your Task

1. Read the source file(s) above. Is the summary accurate? Does it use correct domain vocabulary?
2. Check the KB hits. Is this a meaningful duplicate (same concept already covered well)?
3. Decide:
   - "approve"  → summary is accurate and useful as-is
   - "edit"     → mostly right but needs wording fixes (provide corrected summary)
   - "reject"   → inaccurate, duplicate, trivial, or not worth indexing

Output ONLY this JSON:
{{
  "decision": "approve | edit | reject",
  "final_summary": "the summary to index (corrected if edit, empty string if reject)",
  "reasoning": "1-2 sentences explaining your decision"
}}"""

# ── Core review logic ─────────────────────────────────────────────────────────

def review_one(entry: dict, model: str, codebase: Optional[Path],
               tracker: Optional[TokenTracker] = None) -> ReviewDecision:
    """Run the LLM reviewer on a single staging entry. Returns a ReviewDecision."""

    # 1. Read cited source files
    file_contents: dict[str, str] = {}
    for rel_path in entry.get("source_files", [])[:3]:  # cap at 3 files
        if codebase:
            fpath = (codebase / rel_path).resolve()
            # Ensure resolved path stays within codebase (no path traversal)
            try:
                _in_codebase = fpath.is_relative_to(codebase.resolve())
            except AttributeError:
                # Python < 3.9 fallback
                _in_codebase = str(fpath).startswith(str(codebase.resolve()) + os.sep)
            if not _in_codebase:
                file_contents[rel_path] = f"[blocked: path escapes codebase]"
                continue
            if not fpath.exists():
                candidates = list(codebase.rglob(Path(rel_path).name))
                if candidates:
                    fpath = candidates[0]
        else:
            fpath = Path(rel_path)

        if fpath.exists():
            file_contents[rel_path] = read_source_file(fpath)
        else:
            file_contents[rel_path] = f"[file not found: {fpath}]"

    # 2. Search KB for duplicates
    kb_hits = search_kb(entry.get("summary", entry.get("topic", "")))

    # 3. Call LLM
    prompt = build_review_prompt(entry, file_contents, kb_hits)
    raw = llm_call_json(model, REVIEWER_SYSTEM, prompt,
                        tracker=tracker, phase="Review")

    # 4. Parse decision
    try:
        data = json.loads(raw)
        # Normalise decision string — match first valid keyword
        raw_decision = data.get("decision", "reject").strip().lower()
        matched = "reject"
        for valid in ("approve", "edit", "reject"):
            if valid in raw_decision:
                matched = valid
                break
        data["decision"] = matched
        return ReviewDecision(**data)
    except Exception as e:
        print(f"  [warn] could not parse LLM response: {e}\n  Raw: {raw[:300]}")
        return ReviewDecision(
            decision="reject",
            final_summary="",
            reasoning=f"Reviewer failed to parse LLM output: {e}",
        )

# ── Queue processor ───────────────────────────────────────────────────────────

def process_queue(model: str, codebase: Optional[Path],
                  staging_path: Optional[Path] = None,
                  tracker: Optional[TokenTracker] = None) -> int:
    """
    Process all 'pending' entries in staging_queue.json.
    Returns the number of entries processed.
    """
    if staging_path is None:
        staging_path = STAGING_FILE
    queue = load_queue(staging_path)
    pending = [e for e in queue if e.get("status") == "pending"]

    if not pending:
        return 0

    print(f"\n[Reviewer] {len(pending)} pending suggestion(s) to review")
    rejected = load_queue(REJECTED_FILE)
    processed = 0

    for idx, entry in enumerate(queue):
        if entry.get("status") != "pending":
            continue

        topic = entry.get("topic", "?")
        print(f"\n  [{idx+1}] Reviewing: {topic}")
        print(f"       Files: {entry.get('source_files', [])}")

        decision = review_one(entry, model, codebase, tracker=tracker)
        print(f"       Decision: {decision.decision.upper()} — {decision.reasoning}")

        now = datetime.now(timezone.utc).isoformat()

        if decision.decision in ("approve", "edit"):
            # Build the entry to index
            source_files = entry.get("source_files", [])
            to_index = {
                "summary":     decision.final_summary or entry["summary"],
                "raw_code":    entry.get("raw_code", ""),
                "source_file": ", ".join(source_files) if source_files else "",
                "module":      entry.get("module", "suggestion"),
                "chunk_type":  "function_summary",
            }
            try:
                doc_id, code_doc_id = index_entry(to_index)
                entry["status"]    = decision.decision  # "approve" or "edit"
                entry["doc_id"]    = doc_id
                if code_doc_id:
                    entry["code_doc_id"] = code_doc_id
                entry["reasoning"] = decision.reasoning
                entry["reviewed_at"] = now
                print(f"       ✓ Indexed → doc_id {doc_id}")
            except Exception as e:
                print(f"       ✗ Indexing failed: {e}")
                entry["status"]    = "index_error"
                entry["error"]     = str(e)
                entry["reviewed_at"] = now

        else:  # reject
            entry["status"]    = "rejected"
            entry["reasoning"] = decision.reasoning
            entry["reviewed_at"] = now
            rejected.append(entry)
            print(f"       ✗ Rejected")

        processed += 1

    # Prune completed/rejected entries — keep only pending and errored
    pruned_queue = [e for e in queue if e.get("status") in ("pending", "index_error")]
    save_queue(staging_path, pruned_queue)
    save_queue(REJECTED_FILE, rejected)

    approved = sum(1 for e in queue if e.get("status") in ("approve", "edit"))
    rejected_count = sum(1 for e in queue if e.get("status") == "rejected")
    print(f"\n[Reviewer] Done: {approved} approved, {rejected_count} rejected "
          f"({processed} processed this run)")
    return processed


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reviewer agent: verify staged KB suggestions and promote approved ones to R2R"
    )
    parser.add_argument("--codebase", default=None,
                        help="Root of the codebase (used to read cited source files). "
                             "Optional but recommended.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="litellm model string (default: %(default)s). "
                             "Examples: openai/gpt-4o, anthropic/claude-sonnet-4-20250514, ollama/llama3.1")
    parser.add_argument("--watch", action="store_true",
                        help=f"Keep running and poll staging queue every {WATCH_INTERVAL}s")
    parser.add_argument("--staging-file", default=None,
                        help=f"Path to staging_queue.json (default: {STAGING_FILE})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-item progress, show only summaries")
    args = parser.parse_args()

    staging_file = Path(args.staging_file) if args.staging_file else STAGING_FILE
    codebase = Path(args.codebase).resolve() if args.codebase else None

    print(f"[Reviewer] Model: {args.model}")
    print(f"[Reviewer] Staging queue: {staging_file}")
    if codebase:
        print(f"[Reviewer] Codebase: {codebase}")
    if args.watch:
        print(f"[Reviewer] Watch mode: polling every {WATCH_INTERVAL}s (Ctrl-C to stop)")

    tracker = TokenTracker()
    cost_log_path = BASE_DIR / "cost_log.jsonl"

    if args.watch:
        while True:
            try:
                process_queue(args.model, codebase, staging_file, tracker=tracker)
                time.sleep(WATCH_INTERVAL)
            except KeyboardInterrupt:
                print("\n[Reviewer] Stopped.")
                break
    else:
        processed = process_queue(args.model, codebase, staging_file, tracker=tracker)
        if processed == 0:
            print("[Reviewer] No pending items found.")

    if tracker.phases:
        print(f"\n{tracker.summary()}")
        entry = tracker.to_log_entry(model=args.model, agent="reviewer")
        with cost_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Tokens] Logged to {cost_log_path}")


if __name__ == "__main__":
    main()
