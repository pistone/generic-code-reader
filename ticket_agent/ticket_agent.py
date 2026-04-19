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

# Optional R2R import - gracefully handle if not installed
try:
    from r2r import R2RClient
    HAS_R2R = True
except ImportError:
    R2RClient = None  # type: ignore
    HAS_R2R = False

# Shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codebase_shared.utils import (  # noqa: E402
    TokenTracker, llm_call, load_manifest, save_manifest,
    _is_quota_error,
)
from codebase_shared.work_dir import get_ticket_agent_dir, ensure_work_dirs  # noqa: E402

R2R_URL        = os.getenv("R2R_URL", "http://localhost:7272")
DEFAULT_MODEL  = os.getenv("LLM_MODEL", "openai/gpt-4o")
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "")
GITLAB_TOKEN   = os.getenv("GITLAB_TOKEN", "")
GITLAB_URL     = os.getenv("GITLAB_URL", "https://gitlab.com")

DEDUP_THRESHOLD   = 0.85
MAX_DIFF_CHARS    = 4000   # cap on diff text fed to LLM
MAX_LESSON_CHARS  = 5000   # soft cap on generated lesson text (how-to recipes can be long)

RESOLVED_STATUSES  = {"done", "closed", "resolved", "fixed", "complete", "verified"}
REJECT_RESOLUTIONS = {"won't fix", "wontfix", "duplicate", "cannot reproduce",
                      "incomplete", "not a bug"}

_dedup_warned = False

# Output directory - defaults to work/ticket_agent/, configurable via WORK_DIR env var
OUTPUT_DIR   = get_ticket_agent_dir()
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

def structural_filter(tickets: list[dict],
                      key_pattern: Optional[re.Pattern] = None) -> list[dict]:
    """Filter to tickets likely to contain reusable knowledge.

    All tickets are processed regardless of status (open or resolved).
    Resolved tickets with poor resolutions (won't fix, duplicate, etc.) are filtered out.

    key_pattern: if set, only tickets whose key matches are kept.
    """
    passed: list[dict] = []
    for t in tickets:
        key        = t.get("key") or ""
        status     = (t.get("status") or "").lower().strip()
        resolution = (t.get("resolution") or "").lower().strip()
        comments   = t.get("comments") or []

        if key_pattern and not key_pattern.search(key):
            continue

        # Filter out resolved tickets with poor resolutions
        is_resolved = status and status in RESOLVED_STATUSES
        if is_resolved:
            if resolution in REJECT_RESOLUTIONS:
                continue
            # Require at least 2 comments for resolved tickets to ensure quality
            if len(comments) < 2:
                continue

        # Open tickets are included regardless of comment count
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
    "You extract reusable technical knowledge from engineering tickets. "
    "For open/in-progress tickets: extract the problem description and symptoms "
    "to help others find similar issues. "
    "For resolved tickets: if the ticket contains a root cause, workaround, "
    "design decision, or recurring pattern worth remembering, extract it concisely. "
    "If MR/PR diffs are provided, summarize what the solution actually changed. "
    "If the ticket is just a routine bug fix with no reusable insight, "
    "mark it as not useful. "
    "IMPORTANT: Write in declarative, factual statements only. "
    "NEVER use investigative or procedural language — do not write "
    "'check if', 'open file X', 'see whether', 'look for', 'verify that', "
    "'to reproduce', or any instruction that tells the reader to do something. "
    "State facts: what the root cause WAS (or IS for open tickets), what the fix DID, what the pattern IS. "
    "Output ONLY valid JSON — no markdown fences, no commentary."
)

EXTRACT_PROMPT_TEMPLATE = """{context}

Extract the reusable technical knowledge from this ticket.

Output this JSON:
{{
  "useful": true or false,
  "summary": "2-3 sentences stating the problem and its context as facts — what the symptoms are, what condition triggers it, what component is involved, and what the root cause was (if known). Write in present or past tense ('X fails when Y', 'The bug was caused by Z'). For open tickets, describe what's known about the problem. Include relevant technical details like checker names, error messages, language features, or API names. No instructions, no steps to reproduce.",
  "solution": "2-3 sentences stating what the fix did — specific classes, functions, files, or patterns changed. ('The fix added X to Y', 'Class Z now calls W before V'). IMPORTANT: Only mention file paths that actually appear in the MR diff provided above — do not invent or guess file paths. Only include this field if the ticket is resolved AND MR diff is available, otherwise omit this field entirely.",
  "category": "root_cause" or "workaround" or "design_decision" or "pattern" or "bug_report" (only if useful)
}}

For open tickets without a solution, still mark as useful if the problem description would help others find similar issues.
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
    "You write detailed how-to lessons from resolved engineering tickets. "
    "A lesson reads like a recipe: it tells a future developer exactly what "
    "steps, files, classes, and conventions are involved in making a particular "
    "type of change in this codebase. "
    "Be concrete and specific — name actual files, classes, traits, modules, "
    "and patterns from the solution. Avoid vague advice. "
    "Output ONLY the lesson in plain markdown (no JSON, no preamble, no ticket summary)."
)

LESSON_PROMPT_TEMPLATE = """Ticket: {key} — {title}

Problem summary: {summary}

Solution: {solution}

{mr_diff_section}

Write a how-to lesson for a future developer who needs to make a similar change.

Structure your lesson as follows:

**To <do the thing described by this ticket> in this codebase:**

1. Start with a one-sentence description of what kind of change this is
   (e.g. "To add a new Rust checker", "To register a new device driver",
   "To extend the policy engine with a new rule").

2. List the concrete steps the change involves — each step should name the
   specific file, class, trait, function, or config that needs to change and
   what needs to happen there. Aim for 4-8 steps.

3. Call out any non-obvious pitfalls, ordering constraints, or conventions
   the solution revealed (e.g. "registration must happen before X", "the
   base class requires overriding Y or it silently no-ops").

IMPORTANT: Only reference file paths that appear in the MR diff above. Do NOT
invent or guess file paths. Use the actual names from the diff.
Write for someone who already understands the codebase but has never made this
specific type of change before."""


SOURCE_CODE_EXTENSIONS = {
    '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.c++', '.h++',
    '.java', '.scala', '.kt', '.kts',
    '.py', '.pyx', '.pxd',
    '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
    '.go', '.rs', '.rb', '.php', '.swift', '.m', '.mm',
    '.cs', '.vb', '.fs', '.fsx',
    '.sh', '.bash', '.zsh', '.fish',
    '.sql', '.pl', '.pm', '.r', '.R',
    '.ml', '.mli', '.hs', '.lhs', '.elm', '.ex', '.exs',
    '.mk', 'Makefile', '.cmake', 'CMakeLists.txt',
}


def _filter_diff_to_source_code(mr_diff: str, max_chars: int = 6000) -> str:
    """Filter MR diff to only include source code files, excluding configs/docs."""
    if not mr_diff:
        return ""

    lines = mr_diff.split('\n')
    filtered_lines = []
    current_file = ""
    include_current = False

    for line in lines:
        # Detect file headers in diff output
        if line.startswith('  ') and not line.startswith('   '):
            # This is a filename line like "  path/to/file.cpp"
            current_file = line.strip()
            # Check if it's a source code file
            include_current = any(
                current_file.endswith(ext) or current_file == ext.lstrip('.')
                for ext in SOURCE_CODE_EXTENSIONS
            )
            if include_current:
                filtered_lines.append(line)
        elif include_current:
            filtered_lines.append(line)

    result = '\n'.join(filtered_lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n  ... (diff truncated)"
    return result


def generate_lesson(model: str, ticket: dict, extraction: dict,
                    mr_diff: str = "",
                    tracker: Optional[TokenTracker] = None) -> Optional[str]:
    """Generate a generalizable lesson from an extracted ticket.

    Returns lesson text, or None if no meaningful lesson can be generated.
    Only called when extraction has a non-empty solution.
    """
    solution = extraction.get("solution", "").strip()
    summary  = extraction.get("summary", "").strip()
    if not solution or not summary:
        return None

    # Filter MR diff to source code only
    filtered_diff = _filter_diff_to_source_code(mr_diff)
    mr_diff_section = ""
    if filtered_diff:
        mr_diff_section = f"MR/PR diff (source code files only):\n{filtered_diff}"

    prompt = LESSON_PROMPT_TEMPLATE.format(
        key=ticket.get("key", ""),
        title=ticket.get("title", ""),
        summary=summary,
        solution=solution,
        mr_diff_section=mr_diff_section,
    )
    try:
        lesson = llm_call(model, LESSON_SYSTEM, prompt,
                          max_tokens=2000, tracker=tracker, phase="Lessons")
        lesson = lesson.strip()
        if len(lesson) < 30:
            return None
        # No truncation - let lessons be complete
        return lesson
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

def dedup_against_kb(client, summary: str,
                     threshold: float = DEDUP_THRESHOLD) -> bool:
    """Return True if a similar entry already exists in R2R."""
    if not summary or not HAS_R2R or client is None:
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
    "bug_report":       "reference",  # For open tickets without solutions
}


def index_ticket_summaries(entries: list[dict]) -> dict[str, str]:
    """Index extracted ticket knowledge into R2R.

    Each entry is indexed as two documents if a lesson exists:
      - ticket_summary: problem + solution (for "why did X fail?" queries)
      - lesson: generalizable guidance (for "how do I do X?" queries)

    Returns a dict mapping ticket key → doc_id of the summary document.
    """
    if not HAS_R2R:
        print("  [warn] R2R not installed — skipping indexing")
        return {}

    client   = R2RClient(R2R_URL)
    indexed: dict[str, str] = {}

    for i, entry in enumerate(entries):
        key        = entry["key"]
        source_kind = _CATEGORY_TO_KIND.get(entry["category"], "reference")

        # Skip entries with empty summary
        if not entry.get("summary") or not entry["summary"].strip():
            continue

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
            if "already exists" in str(e):
                # Skip existing documents silently
                continue
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
                if "already exists" not in str(e):
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
    parser.add_argument("--key-pattern", default=None, metavar="REGEX",
                        help="Only process tickets whose key matches this regex "
                             "(e.g. '^ABC-' to restrict to the ABC project, "
                             "'^(ABC|DEF)-' for two teams). Applied before other filters.")
    parser.add_argument("--no-mr", action="store_true",
                        help="Disable MR/PR fetching (not recommended — MR diffs are the "
                             "primary source of solution context; only use if tokens are "
                             "unavailable or tickets have no linked MRs)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-item progress, show only summaries")
    args = parser.parse_args()

    # Ensure work directories exist
    ensure_work_dirs()

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

    # Compile key pattern once (fail early on bad regex)
    key_pattern: Optional[re.Pattern] = None
    if args.key_pattern:
        try:
            key_pattern = re.compile(args.key_pattern, re.IGNORECASE)
        except re.error as e:
            print(f"Error: invalid --key-pattern regex '{args.key_pattern}': {e}")
            sys.exit(1)

    fetch_mrs = not args.no_mr
    if fetch_mrs and not GITHUB_TOKEN and not GITLAB_TOKEN:
        print("Error: MR/PR fetching is enabled but neither GITHUB_TOKEN nor GITLAB_TOKEN is set.")
        print("  Set the appropriate token in your .env file, or pass --no-mr to skip MR fetching")
        print("  (not recommended — MR diffs are the primary source of solution context).")
        sys.exit(1)

    # R2R health check — warn early before spending LLM budget.
    # Extraction results are always saved to ticket_summaries.json so
    # --reindex can recover if R2R comes up later, but it's better to know now.
    if HAS_R2R:
        try:
            _r2r_probe = R2RClient(R2R_URL)
            _r2r_probe.retrieval.search(query="ping", search_settings={"limit": 1})
            print(f"R2R: connected at {R2R_URL}")
        except Exception as _e:
            print(f"Warning: R2R is not reachable at {R2R_URL} ({_e})")
            print("  Extraction will proceed and results saved to ticket_summaries.json,")
            print("  but nothing will be indexed. Start R2R then run --reindex afterwards.")
            print("  To abort: Ctrl-C now. To continue anyway: press Enter.")
            try:
                input()
            except KeyboardInterrupt:
                sys.exit(0)
    else:
        print("Warning: R2R package not installed")
        print("  Extraction will proceed and results saved to ticket_summaries.json,")
        print("  but nothing will be indexed. Install r2r then run --reindex afterwards.")

    tracker  = TokenTracker()
    manifest = load_manifest(manifest_path)

    # Step 1: load
    all_tickets = load_tickets(tickets_dir)
    print(f"Loaded {len(all_tickets)} ticket(s) from {tickets_dir}")
    if not all_tickets:
        return

    # Step 2: structural filter (includes optional key-pattern check)
    candidates = structural_filter(all_tickets, key_pattern=key_pattern)
    filter_desc = f"key=/{args.key_pattern}/, " if key_pattern else ""
    print(f"Structural filter ({filter_desc}exclude duplicates/won't-fix): "
          f"{len(candidates)}/{len(all_tickets)} passed")
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

    client = R2RClient(R2R_URL) if HAS_R2R else None

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
        # Skip MR fetching for open tickets (no solution to document yet)
        status = (t.get("status") or "").lower().strip()
        is_open = status and status not in RESOLVED_STATUSES

        mr_context = ""
        if fetch_mrs and not is_open:
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

        # For open tickets, always index for similarity search (override LLM usefulness check)
        # For resolved tickets, respect the LLM's usefulness judgment
        status = (t.get("status") or "").lower().strip()
        is_open = status and status not in RESOLVED_STATUSES
        is_useful = result["useful"] or is_open

        if not is_useful:
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
                                                 mr_diff=mr_context,
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
