#!/usr/bin/env python3
"""Ticket agent: extract reusable knowledge from Jira/PR exports into R2R.

Expects pre-exported JSON ticket files on disk (one ticket per file, or a
JSON array of tickets).  Filters aggressively, then uses LLM to extract
only the knowledge nuggets worth indexing.

For tickets with linked MRs/PRs, fetches the MR diff and uses it to
summarize the actual solution and extract generalizable lessons.
Lessons are written as .md files to ticket_agent/lessons/ for later
indexing by doc_agent.

Usage:
    python -m ticket_agent.ticket_agent --tickets /path/to/exported/tickets
    python -m ticket_agent.ticket_agent --tickets /tmp/tickets --incremental
    python -m ticket_agent.ticket_agent --reindex   # re-index from saved summaries

Environment variables:
    GITHUB_TOKEN   GitHub personal access token (read:repo scope)
    GITLAB_TOKEN   GitLab personal access token
    GITLAB_URL     GitLab instance URL (default: https://gitlab.com)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from r2r import R2RClient

# Shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codebase_shared.utils import (  # noqa: E402
    TokenTracker, llm_call, load_manifest, save_manifest,
    _is_quota_error,
)

R2R_URL        = os.getenv("R2R_URL", "http://localhost:7272")
DEFAULT_MODEL  = os.getenv("LLM_MODEL", "openai/gpt-4o")
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "")
GITLAB_TOKEN   = os.getenv("GITLAB_TOKEN", "")
GITLAB_URL     = os.getenv("GITLAB_URL", "https://gitlab.com")

DEDUP_THRESHOLD   = 0.85
MAX_DIFF_CHARS    = 4000   # cap on diff text fed to LLM
MAX_LESSON_CHARS  = 1200   # cap on generated lesson text

RESOLVED_STATUSES  = {"done", "closed", "resolved", "fixed", "complete"}
REJECT_RESOLUTIONS = {"won't fix", "wontfix", "duplicate", "cannot reproduce",
                      "incomplete", "not a bug"}

_dedup_warned = False

OUTPUT_DIR   = Path(__file__).resolve().parent
LESSONS_DIR  = OUTPUT_DIR / "lessons"


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
        status     = (t.get("status") or "").lower().strip()
        resolution = (t.get("resolution") or "").lower().strip()
        comments   = t.get("comments") or []

        if status and status not in RESOLVED_STATUSES:
            continue
        if resolution in REJECT_RESOLUTIONS:
            continue
        if len(comments) < 2:
            continue

        passed.append(t)
    return passed


# ---------------------------------------------------------------------------
# Step 2b: Fetch MR/PR context
# ---------------------------------------------------------------------------

# Patterns to extract MR/PR references from text
_GITHUB_URL_RE = re.compile(
    r'https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)', re.IGNORECASE)
_GITLAB_URL_RE = re.compile(
    r'https?://[^/]*gitlab[^/]*/([^/]+)/([^/]+)/-/merge_requests/(\d+)',
    re.IGNORECASE)
_BARE_PR_RE    = re.compile(r'\bPR\s*#?(\d+)\b', re.IGNORECASE)
_BARE_MR_RE    = re.compile(r'\bMR\s*!?(\d+)\b', re.IGNORECASE)


def parse_mr_refs(ticket: dict) -> list[str]:
    """Extract GitHub PR URLs and GitLab MR URLs from a ticket.

    Looks in linked_prs (list of strings/URLs) and description.
    Returns a list of canonical API URLs ready for fetching.
    """
    text_sources = [ticket.get("description") or ""]
    for ref in (ticket.get("linked_prs") or []):
        text_sources.append(str(ref))

    urls: list[str] = []
    for text in text_sources:
        for m in _GITHUB_URL_RE.finditer(text):
            owner, repo, number = m.group(1), m.group(2), m.group(3)
            urls.append(f"github:{owner}/{repo}/{number}")
        for m in _GITLAB_URL_RE.finditer(text):
            owner, repo, number = m.group(1), m.group(2), m.group(3)
            base = GITLAB_URL.rstrip("/")
            urls.append(f"gitlab:{base}/{owner}/{repo}/{number}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]  # type: ignore[func-returns-value]


def _fetch_github_pr(owner: str, repo: str, number: str) -> Optional[str]:
    """Fetch GitHub PR title, description and diff summary via REST API."""
    if not GITHUB_TOKEN:
        return None
    try:
        import urllib.request
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        # Fetch PR metadata
        pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
        req = urllib.request.Request(pr_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            pr = json.loads(resp.read().decode())

        title = pr.get("title", "")
        body  = (pr.get("body") or "")[:300]

        # Fetch diff
        diff_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/files"
        req = urllib.request.Request(diff_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            files = json.loads(resp.read().decode())

        diff_lines: list[str] = []
        total_chars = 0
        for f in files[:20]:
            fname    = f.get("filename", "?")
            additions = f.get("additions", 0)
            deletions = f.get("deletions", 0)
            patch    = (f.get("patch") or "")[:800]
            line = f"  {fname} (+{additions}/-{deletions})\n{patch}"
            if total_chars + len(line) > MAX_DIFF_CHARS:
                diff_lines.append("  ... (diff truncated)")
                break
            diff_lines.append(line)
            total_chars += len(line)

        diff_str = "\n".join(diff_lines)
        return f"PR #{number}: {title}\n{body}\nChanged files:\n{diff_str}"

    except Exception as e:
        print(f"  [warn] GitHub PR fetch failed ({owner}/{repo}#{number}): {e}")
        return None


def _fetch_gitlab_mr(base: str, owner: str, repo: str, number: str) -> Optional[str]:
    """Fetch GitLab MR title, description and diff summary via REST API."""
    if not GITLAB_TOKEN:
        return None
    try:
        import urllib.request
        import urllib.parse
        headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}
        project_path = urllib.parse.quote(f"{owner}/{repo}", safe="")

        mr_url = f"{base}/api/v4/projects/{project_path}/merge_requests/{number}"
        req = urllib.request.Request(mr_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            mr = json.loads(resp.read().decode())

        title = mr.get("title", "")
        body  = (mr.get("description") or "")[:300]

        changes_url = f"{base}/api/v4/projects/{project_path}/merge_requests/{number}/changes"
        req = urllib.request.Request(changes_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        diff_lines: list[str] = []
        total_chars = 0
        for f in (data.get("changes") or [])[:20]:
            fname = f.get("new_path", f.get("old_path", "?"))
            diff  = (f.get("diff") or "")[:800]
            line  = f"  {fname}\n{diff}"
            if total_chars + len(line) > MAX_DIFF_CHARS:
                diff_lines.append("  ... (diff truncated)")
                break
            diff_lines.append(line)
            total_chars += len(line)

        diff_str = "\n".join(diff_lines)
        return f"MR !{number}: {title}\n{body}\nChanged files:\n{diff_str}"

    except Exception as e:
        print(f"  [warn] GitLab MR fetch failed ({owner}/{repo}!{number}): {e}")
        return None


def fetch_mr_context(ticket: dict) -> str:
    """Fetch MR/PR diffs for a ticket and return a combined context string.

    Returns empty string if no tokens configured or all fetches fail.
    """
    refs = parse_mr_refs(ticket)
    if not refs:
        return ""

    parts: list[str] = []
    for ref in refs[:3]:  # cap at 3 MRs per ticket
        if ref.startswith("github:"):
            _, rest = ref.split(":", 1)
            owner, repo, number = rest.rsplit("/", 2)
            result = _fetch_github_pr(owner, repo, number)
            if result:
                parts.append(result)
        elif ref.startswith("gitlab:"):
            _, rest = ref.split(":", 1)
            # rest = base/owner/repo/number
            tokens = rest.rsplit("/", 3)
            if len(tokens) == 4:
                base_url = tokens[0]
                owner, repo, number = tokens[1], tokens[2], tokens[3]
                result = _fetch_gitlab_mr(base_url, owner, repo, number)
                if result:
                    parts.append(result)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 3: Build LLM context
# ---------------------------------------------------------------------------

def build_ticket_context(ticket: dict, mr_context: str = "",
                         max_chars: int = 3000) -> str:
    """Compose a condensed view of a ticket (+ MR diffs) for LLM extraction."""
    parts: list[str] = []

    key   = ticket.get("key", "unknown")
    title = ticket.get("title", "")
    parts.append(f"Ticket: {key} — {title}")

    desc = (ticket.get("description") or "")[:500]
    if desc:
        parts.append(f"\nDescription:\n{desc}")

    resolution = ticket.get("resolution", "")
    if resolution:
        parts.append(f"\nResolution: {resolution}")

    labels = ticket.get("labels") or []
    if labels:
        parts.append(f"Labels: {', '.join(labels)}")

    comments = ticket.get("comments") or []
    if comments:
        recent = comments[-5:]
        parts.append("\nRecent comments:")
        for c in recent:
            author = c.get("author", "?")
            body   = (c.get("body") or "")[:300]
            parts.append(f"  [{author}]: {body}")

    if mr_context:
        parts.append(f"\nMerge request(s):\n{mr_context[:MAX_DIFF_CHARS]}")

    text = "\n".join(parts)
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Step 4: LLM extraction
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You extract reusable technical knowledge from resolved tickets. "
    "If the ticket contains a root cause, workaround, design decision, "
    "or recurring pattern worth remembering, extract it concisely. "
    "If MR/PR diffs are provided, summarize what the solution actually changed. "
    "If the ticket is just a routine bug fix with no reusable insight, "
    "mark it as not useful. "
    "Output ONLY valid JSON — no markdown fences, no commentary."
)

EXTRACT_PROMPT_TEMPLATE = """{context}

Extract the reusable technical knowledge from this ticket.

Output this JSON:
{{
  "useful": true or false,
  "summary": "1-3 sentence description of the problem and its context (only if useful)",
  "solution": "1-2 sentence description of what the fix/change actually did — specific classes, functions, patterns used (only if useful and MR diff is available, otherwise omit)",
  "category": "root_cause" or "workaround" or "design_decision" or "pattern" (only if useful)
}}

If there's no reusable insight (routine fix, config change, typo), set useful to false."""


def extract_knowledge(model: str, context: str,
                      tracker: Optional[TokenTracker] = None) -> dict:
    """LLM call to extract knowledge from a ticket."""
    prompt = EXTRACT_PROMPT_TEMPLATE.format(context=context)
    raw = llm_call(model, EXTRACT_SYSTEM, prompt,
                   max_tokens=400, json_mode=True,
                   tracker=tracker, phase="Extraction")
    try:
        result = json.loads(raw)
        return {
            "useful":   bool(result.get("useful", False)),
            "summary":  str(result.get("summary", "")),
            "solution": str(result.get("solution", "")),
            "category": str(result.get("category", "general")),
        }
    except (json.JSONDecodeError, TypeError):
        return {"useful": False, "summary": "", "solution": "", "category": ""}


# ---------------------------------------------------------------------------
# Step 4b: Generate lesson learned
# ---------------------------------------------------------------------------

LESSON_SYSTEM = (
    "You generate generalizable lessons from resolved engineering tickets. "
    "A lesson is a reusable how-to or pattern that helps a future developer "
    "working in this area of the codebase — not a summary of the bug itself. "
    "Be specific: name the classes, patterns, or conventions involved. "
    "Output ONLY the lesson text (plain markdown, no JSON, no preamble)."
)

LESSON_PROMPT_TEMPLATE = """Ticket: {key} — {title}

Problem summary: {summary}

Solution: {solution}

Write a lesson learned that a future developer can apply. Focus on:
- What pattern or convention this reveals (how to do X in this codebase)
- What to watch out for (pitfalls, assumptions the code makes)
- Any architectural constraint the fix exposed

Write 2-4 sentences of plain prose. Be specific — use actual class/function
names from the solution if available. Do NOT summarize the bug; teach the lesson."""


def generate_lesson(model: str, ticket: dict, extraction: dict,
                    tracker: Optional[TokenTracker] = None) -> Optional[str]:
    """Generate a generalizable lesson from an extracted ticket.

    Returns lesson text, or None if no meaningful lesson can be generated.
    Only called when extraction has a non-empty solution.
    """
    solution = extraction.get("solution", "").strip()
    summary  = extraction.get("summary", "").strip()
    if not solution or not summary:
        return None

    prompt = LESSON_PROMPT_TEMPLATE.format(
        key=ticket.get("key", ""),
        title=ticket.get("title", ""),
        summary=summary,
        solution=solution,
    )
    try:
        lesson = llm_call(model, LESSON_SYSTEM, prompt,
                          max_tokens=300, tracker=tracker, phase="Lessons")
        lesson = lesson.strip()
        if len(lesson) < 30:
            return None
        return lesson[:MAX_LESSON_CHARS]
    except Exception as e:
        print(f"  [warn] Lesson generation failed: {e}")
        return None


def save_lesson_file(ticket: dict, extraction: dict, lesson: str) -> Path:
    """Write a lesson to ticket_agent/lessons/<KEY>.md.

    The file includes YAML frontmatter so doc_agent can classify it as
    a tutorial/how-to when it indexes the lessons/ directory.
    Returns the path written.
    """
    LESSONS_DIR.mkdir(exist_ok=True)
    key      = ticket.get("key", "unknown")
    title    = ticket.get("title", "")
    category = extraction.get("category", "general")
    date     = (ticket.get("resolved") or ticket.get("created") or "")[:10]

    content = (
        f"---\n"
        f"ticket: {key}\n"
        f"title: \"{title}\"\n"
        f"category: {category}\n"
        f"date: {date}\n"
        f"---\n\n"
        f"# Lesson from {key}: {title}\n\n"
        f"{lesson}\n"
    )

    path = LESSONS_DIR / f"{key}.md"
    path.write_text(content, encoding="utf-8")
    return path


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
        global _dedup_warned
        if not _dedup_warned:
            print("  [warn] R2R unreachable for dedup — duplicates may be indexed")
            _dedup_warned = True
    return False


# ---------------------------------------------------------------------------
# Step 6: Index to R2R
# ---------------------------------------------------------------------------

_CATEGORY_TO_KIND = {
    "root_cause":       "rationale",
    "workaround":       "operational",
    "design_decision":  "rationale",
    "pattern":          "reference",
}


def index_ticket_summaries(entries: list[dict]) -> dict[str, str]:
    """Index extracted ticket knowledge into R2R.

    Each entry is indexed as two documents if a lesson exists:
      - ticket_summary: problem + solution (for "why did X fail?" queries)
      - lesson: generalizable guidance (for "how do I do X?" queries)

    Returns a dict mapping ticket key → doc_id of the summary document.
    """
    client   = R2RClient(R2R_URL)
    indexed: dict[str, str] = {}

    for i, entry in enumerate(entries):
        key        = entry["key"]
        source_kind = _CATEGORY_TO_KIND.get(entry["category"], "reference")

        # Compose the indexed text: summary + solution if present
        text_parts = [entry["summary"]]
        if entry.get("solution"):
            text_parts.append(f"Solution: {entry['solution']}")
        text = "\n".join(text_parts)

        try:
            resp = client.documents.create(
                raw_text=text,
                metadata={
                    "source_file":   key,
                    "module":        entry["category"],
                    "chunk_type":    "ticket_summary",
                    "source_type":   "ticket",
                    "source_kind":   source_kind,
                    "doc_title":     entry["title"],
                    "last_modified": entry.get("resolved", ""),
                },
            )
            indexed[key] = str(resp.results.document_id)
        except Exception as e:
            print(f"  [warn] {key}: index failed: {e}")
            continue

        # Index lesson separately if present
        if entry.get("lesson"):
            try:
                client.documents.create(
                    raw_text=entry["lesson"],
                    metadata={
                        "source_file":   key,
                        "module":        entry["category"],
                        "chunk_type":    "lesson",
                        "source_type":   "ticket",
                        "source_kind":   "tutorial",
                        "doc_title":     f"Lesson from {key}: {entry['title']}",
                        "last_modified": entry.get("resolved", ""),
                    },
                )
            except Exception as e:
                print(f"  [warn] {key}: lesson index failed: {e}")

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
        ",".join(str(p) for p in (ticket.get("linked_prs") or [])),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ticket agent: extract knowledge from ticket exports → R2R",
    )
    parser.add_argument("--tickets", default=None,
                        help="Directory of exported ticket JSON files")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="litellm model string (default: %(default)s)")
    parser.add_argument("--incremental", action="store_true",
                        help="Only process tickets changed since last run")
    parser.add_argument("--reindex", action="store_true",
                        help="Skip extraction; re-index from saved ticket_summaries.json")
    parser.add_argument("--no-mr", action="store_true",
                        help="Disable MR/PR fetching (skips GITHUB_TOKEN / GITLAB_TOKEN lookups)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-item progress, show only summaries")
    args = parser.parse_args()

    manifest_path  = OUTPUT_DIR / "ticket_hashes.json"
    cost_log_path  = OUTPUT_DIR / "cost_log.jsonl"
    summaries_path = OUTPUT_DIR / "ticket_summaries.json"

    # --reindex: replay from saved summaries, no LLM calls
    if args.reindex:
        if not summaries_path.exists():
            print(f"Error: {summaries_path} not found. Run extraction first.")
            sys.exit(1)
        try:
            entries = json.loads(summaries_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error: {summaries_path} is corrupted: {e}")
            sys.exit(1)
        print(f"Re-indexing {len(entries)} saved ticket summaries into R2R...")
        indexed = index_ticket_summaries(entries)
        print(f"[Done] {len(indexed)}/{len(entries)} entries indexed (no LLM calls)")
        return

    if not args.tickets:
        print("Error: --tickets is required unless using --reindex")
        sys.exit(1)

    tickets_dir = Path(args.tickets).resolve()
    if not tickets_dir.is_dir():
        print(f"Error: '{tickets_dir}' is not a directory")
        sys.exit(1)

    fetch_mrs = not args.no_mr
    if fetch_mrs and not GITHUB_TOKEN and not GITLAB_TOKEN:
        print("  [info] No GITHUB_TOKEN or GITLAB_TOKEN set — MR fetching disabled")
        fetch_mrs = False

    tracker  = TokenTracker()
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
            h   = ticket_hash(t)
            if manifest.get(key, {}).get("hash") != h:
                changed.append(t)
        if not changed:
            print("[Incremental] No tickets changed since last run.")
            return
        print(f"[Incremental] {len(changed)}/{len(candidates)} tickets changed")
        candidates = changed

    client = R2RClient(R2R_URL)

    # Load existing summaries for merging
    existing_summaries: list[dict] = []
    if summaries_path.exists():
        try:
            existing_summaries = json.loads(summaries_path.read_text())
        except Exception:
            existing_summaries = []
    existing_keys = {e["key"] for e in existing_summaries}

    # Skip tickets already processed in a prior run (resume support)
    remaining: list[dict] = []
    for t in candidates:
        key = t.get("key", "")
        if key and manifest.get(key, {}).get("hash") == ticket_hash(t):
            continue
        remaining.append(t)

    if not remaining:
        print("[Resume] All candidate tickets already processed.")
        return

    print(f"\nProcessing {len(remaining)} ticket(s)"
          f" ({len(candidates) - len(remaining)} already done)...")

    useful_entries: list[dict] = []
    not_useful = 0
    duplicates = 0
    lessons_written = 0
    quota_hit = False

    for i, t in enumerate(remaining, 1):
        key = t.get("key", "unknown")

        # Fetch MR context (best-effort)
        mr_context = ""
        if fetch_mrs:
            mr_context = fetch_mr_context(t)

        context = build_ticket_context(t, mr_context=mr_context)

        try:
            result = extract_knowledge(args.model, context, tracker=tracker)
        except Exception as e:
            if _is_quota_error(str(e).lower()):
                print(f"\n  Quota exhausted at ticket {key} ({i}/{len(remaining)})")
                print(f"  Progress saved. Re-run to resume.")
                quota_hit = True
                break
            raise

        if not result["useful"]:
            not_useful += 1
            if not args.quiet:
                print(f"  [{i}/{len(remaining)}] {key}: not useful")
            manifest[key] = {"hash": ticket_hash(t), "doc_id": None}
        else:
            # Dedup against existing KB
            if dedup_against_kb(client, result["summary"]):
                duplicates += 1
                if not args.quiet:
                    print(f"  [{i}/{len(remaining)}] {key}: duplicate (skipped)")
                manifest[key] = {"hash": ticket_hash(t), "doc_id": None}
            else:
                # Generate lesson if we have a solution from MR diff
                lesson: Optional[str] = None
                if result.get("solution"):
                    try:
                        lesson = generate_lesson(args.model, t, result,
                                                 tracker=tracker)
                    except Exception as e:
                        if _is_quota_error(str(e).lower()):
                            quota_hit = True
                            break
                        print(f"  [warn] Lesson generation error for {key}: {e}")

                # Write lesson .md file
                lesson_path: Optional[Path] = None
                if lesson:
                    lesson_path = save_lesson_file(t, result, lesson)
                    lessons_written += 1

                entry = {
                    "key":      key,
                    "title":    t.get("title", ""),
                    "summary":  result["summary"],
                    "solution": result.get("solution", ""),
                    "lesson":   lesson or "",
                    "category": result["category"],
                    "resolved": t.get("resolved", ""),
                }
                useful_entries.append(entry)

                # Persist to summaries file after each ticket
                if key not in existing_keys:
                    existing_summaries.append(entry)
                    existing_keys.add(key)
                else:
                    existing_summaries = [
                        entry if e["key"] == key else e
                        for e in existing_summaries
                    ]
                summaries_path.write_text(
                    json.dumps(existing_summaries, indent=2))

                if not args.quiet:
                    lesson_tag = f" + lesson → {lesson_path.name}" if lesson_path else ""
                    print(f"  [{i}/{len(remaining)}] {key}: "
                          f"[{result['category']}] "
                          f"{result['summary'][:80]}...{lesson_tag}")

                manifest[key] = {
                    "hash": ticket_hash(t),
                    "doc_id": None,
                    "extracted": True,
                }

        save_manifest(manifest_path, manifest)
        time.sleep(0.3)

    print(f"\nExtraction: {len(useful_entries)} useful, "
          f"{not_useful} not useful, {duplicates} duplicates, "
          f"{lessons_written} lessons written")

    if lessons_written:
        print(f"  Lessons saved to {LESSONS_DIR}/")
        print(f"  Run: python -m doc_agent.doc_agent --docs {LESSONS_DIR} "
              f"to index them into R2R")

    # Index to R2R
    to_index = [
        e for e in useful_entries
        if not manifest.get(e["key"], {}).get("doc_id")
    ]
    if to_index:
        print(f"\nIndexing {len(to_index)} ticket entries into R2R"
              f" ({len(useful_entries) - len(to_index)} already indexed)...")
        indexed = index_ticket_summaries(to_index)

        for entry in to_index:
            key    = entry["key"]
            doc_id = indexed.get(key)
            if doc_id is None:
                continue
            orig = next((t for t in remaining if t.get("key") == key), {})
            manifest[key] = {"hash": ticket_hash(orig), "doc_id": doc_id}
        save_manifest(manifest_path, manifest)
    else:
        print("\nNo new knowledge to index.")

    if tracker.phases:
        print(f"\n{tracker.summary()}")
        log_entry = tracker.to_log_entry(model=args.model, agent="ticket_agent")
        with cost_log_path.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")

    if quota_hit:
        print(f"\n[Paused] {len(useful_entries)} entries extracted so far. "
              f"Re-run to continue from ticket {i}/{len(remaining)}.")
    else:
        print(f"\n[Done] {len(useful_entries)} ticket entries indexed")


if __name__ == "__main__":
    main()
