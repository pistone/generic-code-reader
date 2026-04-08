"""
tools/tickets.py — Jira ticket fetching and knowledge extraction tools.

fetch_tickets:   Downloads Jira tickets matching a project/filter to disk.
process_tickets: Extracts generalizable lessons from tickets + MR diffs, indexes to R2R.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from codebase_shared.utils import TokenTracker, load_manifest, save_manifest


def fetch_tickets(
    project: str,
    since: str = "-365d",
    key_pattern: Optional[str] = None,
    jql: Optional[str] = None,
    output_dir: Optional[str] = None,
    debug: bool = False,
) -> dict:
    """Fetch resolved Jira tickets for a project and save them to disk.

    Args:
        project:     Jira project key (e.g. "PROJ") or space-separated list ("PROJ ABC").
        since:       How far back to fetch (e.g. "-365d", "-90d", "2024-01-01").
        key_pattern: Optional regex to filter ticket keys (e.g. r"ABC-\d+").
        jql:         Override JQL query (ignores project/since if set).
        output_dir:  Where to write ticket JSON files (default: ticket_agent/tickets/).
        debug:       Print extra diagnostic info.

    Returns:
        dict with fetched_count, updated_count, output_dir.
    """
    from ticket_agent.fetch_tickets import (
        JiraClient, fetch_tickets as _fetch, _build_default_jql,
    )

    jira_url   = os.environ.get("JIRA_URL", "")
    jira_email = os.environ.get("JIRA_EMAIL", "")
    jira_token = os.environ.get("JIRA_TOKEN", "")

    if not all([jira_url, jira_email, jira_token]):
        missing = [k for k, v in {
            "JIRA_URL": jira_url, "JIRA_EMAIL": jira_email, "JIRA_TOKEN": jira_token,
        }.items() if not v]
        raise EnvironmentError(f"Missing Jira credentials: {', '.join(missing)}")

    client = JiraClient(jira_url, jira_email, jira_token)
    projects = project.split() if isinstance(project, str) else list(project)
    query = jql or _build_default_jql(projects, since)
    print(f"[fetch_tickets] JQL: {query}")

    tickets_dir = Path(output_dir) if output_dir else (_ROOT / "ticket_agent" / "tickets")
    tickets_dir.mkdir(parents=True, exist_ok=True)

    fetched, updated, newest = _fetch(client, query, debug=debug)
    print(f"[fetch_tickets] Fetched {fetched}, updated {updated}")

    return {
        "fetched_count": fetched,
        "updated_count": updated,
        "newest_updated": newest,
        "output_dir": str(tickets_dir),
        "jql": query,
    }


def process_tickets(
    model: str = None,
    tickets_dir: Optional[str] = None,
    key_pattern: Optional[str] = None,
    no_mr: bool = False,
    tracker: Optional[TokenTracker] = None,
) -> dict:
    """Extract lessons from tickets and their linked MR/PR diffs, index into R2R.

    Mirrors the logic of ticket_agent/ticket_agent.py main() but callable as a function.

    Args:
        model:        LLM model for summarization.
        tickets_dir:  Directory containing ticket JSON files (default: ticket_agent/tickets/).
        key_pattern:  Regex to filter ticket keys (e.g. r"ABC-\d+").
        no_mr:        Skip MR/PR diff fetching (not recommended).
        tracker:      Optional token tracker.

    Returns:
        dict with processed_count, lessons_count, indexed_count, not_useful, duplicates.
    """
    import re as _re
    from r2r import R2RClient

    from ticket_agent.ticket_agent import (
        load_tickets, structural_filter, fetch_mr_context,
        build_ticket_context, extract_knowledge, generate_lesson,
        save_lesson_file, dedup_against_kb, index_ticket_summaries,
        ticket_hash, GITHUB_TOKEN, GITLAB_TOKEN,
    )
    from ticket_agent.ticket_agent import _is_quota_error, R2R_URL, OUTPUT_DIR

    _model = model or os.environ.get("LLM_MODEL", "openai/gpt-4o")
    t_dir  = Path(tickets_dir) if tickets_dir else (_ROOT / "ticket_agent" / "tickets")

    if not t_dir.exists():
        raise ValueError(f"Tickets directory not found: {t_dir}. Run fetch_tickets first.")

    fetch_mrs = not no_mr
    if fetch_mrs and not GITHUB_TOKEN and not GITLAB_TOKEN:
        raise EnvironmentError(
            "MR/PR fetching requires GITHUB_TOKEN or GITLAB_TOKEN. "
            "Set one, or pass no_mr=True to skip (not recommended)."
        )

    manifest_path  = OUTPUT_DIR / "ticket_hashes.json"
    summaries_path = OUTPUT_DIR / "ticket_summaries.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)

    kp = _re.compile(key_pattern, _re.IGNORECASE) if key_pattern else None
    all_tickets = load_tickets(t_dir)
    print(f"[process_tickets] Loaded {len(all_tickets)} tickets")

    candidates = structural_filter(all_tickets, key_pattern=kp)
    print(f"[process_tickets] {len(candidates)} passed structural filter")
    if not candidates:
        return {"processed_count": 0, "lessons_count": 0, "indexed_count": 0,
                "not_useful": 0, "duplicates": 0}

    # Resume: skip already-processed tickets
    remaining = [t for t in candidates
                 if manifest.get(t.get("key", ""), {}).get("hash") != ticket_hash(t)]
    print(f"[process_tickets] {len(remaining)} to process "
          f"({len(candidates) - len(remaining)} already done)")

    existing_summaries: list[dict] = []
    if summaries_path.exists():
        try:
            existing_summaries = json.loads(summaries_path.read_text())
        except Exception:
            pass
    existing_keys = {e["key"] for e in existing_summaries}

    client = R2RClient(R2R_URL)
    if tracker is None:
        tracker = TokenTracker()

    useful_entries: list[dict] = []
    not_useful = 0
    duplicates = 0
    lessons_written = 0

    for i, t in enumerate(remaining, 1):
        key = t.get("key", "unknown")
        mr_context = fetch_mr_context(t) if fetch_mrs else ""
        context = build_ticket_context(t, mr_context=mr_context)

        try:
            result = extract_knowledge(_model, context, tracker=tracker)
        except Exception as e:
            if _is_quota_error(str(e).lower()):
                print(f"  Quota exhausted at {key} ({i}/{len(remaining)}) — saving progress")
                break
            raise

        if not result["useful"]:
            not_useful += 1
            manifest[key] = {"hash": ticket_hash(t), "doc_id": None}
            continue

        if dedup_against_kb(client, result["summary"]):
            duplicates += 1
            manifest[key] = {"hash": ticket_hash(t), "doc_id": None}
            continue

        lesson: Optional[str] = None
        if result.get("solution"):
            try:
                lesson = generate_lesson(_model, t, result, tracker=tracker)
            except Exception as e:
                if _is_quota_error(str(e).lower()):
                    break
                print(f"  [warn] Lesson generation error for {key}: {e}")

        lesson_path = None
        if lesson:
            lesson_path = save_lesson_file(t, result, lesson)
            lessons_written += 1

        entry = {
            "key": key,
            "summary": result["summary"],
            "solution": result.get("solution", ""),
            "lesson_file": str(lesson_path) if lesson_path else None,
            "doc_id": None,
        }
        useful_entries.append(entry)
        print(f"  [{i}/{len(remaining)}] {key}: extracted")

    # Merge with existing and index
    all_entries = [e for e in existing_summaries if e["key"] not in {u["key"] for u in useful_entries}]
    all_entries.extend(useful_entries)
    summaries_path.write_text(json.dumps(all_entries, indent=2))

    to_index = [e for e in useful_entries if not e.get("doc_id")]
    indexed_map = index_ticket_summaries(to_index) if to_index else {}

    for entry in all_entries:
        if entry["key"] in indexed_map:
            entry["doc_id"] = indexed_map[entry["key"]]
            manifest[entry["key"]] = {"hash": ticket_hash(
                next((t for t in remaining if t.get("key") == entry["key"]), {})),
                "doc_id": entry["doc_id"]}

    summaries_path.write_text(json.dumps(all_entries, indent=2))
    save_manifest(manifest_path, manifest)

    print(f"[process_tickets] Done: {len(useful_entries)} useful, "
          f"{lessons_written} lessons, {len(indexed_map)} indexed, "
          f"{not_useful} not useful, {duplicates} duplicates")

    return {
        "processed_count": len(remaining),
        "lessons_count": lessons_written,
        "indexed_count": len(indexed_map),
        "not_useful": not_useful,
        "duplicates": duplicates,
    }
