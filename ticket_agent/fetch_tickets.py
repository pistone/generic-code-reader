#!/usr/bin/env python3
"""Fetch Jira tickets and persist them as individual JSON files.

Each ticket is saved as ticket_agent/tickets/<KEY>.json — one file per ticket,
human-readable, browsable, and editable. Re-runs only fetch tickets updated
since the last fetch (incremental via last_fetch.json watermark).

By default fetches resolved tickets updated in the past year. Override with
--project, --since, or a full --jql expression.

The persisted tickets are the input to ticket_agent.py:
    python -m ticket_agent.fetch_tickets --project PROJ
    python -m ticket_agent.fetch_tickets --project PROJ --since -90d
    python -m ticket_agent.fetch_tickets --jql "project = PROJ AND status = Done"
    python -m ticket_agent.ticket_agent --tickets ticket_agent/tickets/

Environment variables:
    JIRA_URL      Base URL, e.g. https://yourcompany.atlassian.net
    JIRA_EMAIL    Atlassian account email
    JIRA_TOKEN    API token from id.atlassian.com/manage-profile/security/api-tokens
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from base64 import b64encode
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

JIRA_URL   = os.getenv("JIRA_URL", "").rstrip("/")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_TOKEN = os.getenv("JIRA_TOKEN", "")

OUTPUT_DIR    = Path(__file__).resolve().parent
TICKETS_DIR   = OUTPUT_DIR / "tickets"
WATERMARK_PATH = OUTPUT_DIR / "last_fetch.json"

PAGE_SIZE = 100   # Jira max results per page


# ---------------------------------------------------------------------------
# Jira REST API client (no external dependencies)
# ---------------------------------------------------------------------------

class JiraClient:
    def __init__(self, base_url: str, email: str, token: str):
        self.base_url = base_url.rstrip("/")
        credentials  = b64encode(f"{email}:{token}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {credentials}",
            "Accept":        "application/json",
            "Content-Type":  "application/json",
        }

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}/rest/api/3/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Jira API {e.code} for {url}: {body}") from e

    def search(self, jql: str, start_at: int = 0,
               max_results: int = PAGE_SIZE) -> dict:
        return self._get("search/jql", {
            "jql":        jql,
            "startAt":    start_at,
            "maxResults": max_results,
            "fields":     ",".join([
                "summary", "description", "status", "resolution",
                "assignee", "reporter", "labels", "priority",
                "created", "updated", "resolutiondate",
                "comment", "issuelinks", "fixVersions",
                "components", "issuetype", "parent",
                "customfield_10014",   # Epic Link (common)
                "customfield_10016",   # Story Points (common)
            ]),
        })

    def get_issue(self, key: str) -> dict:
        return self._get(f"issue/{key}")


# ---------------------------------------------------------------------------
# Normalise a raw Jira issue into our storage schema
# ---------------------------------------------------------------------------

def _text_from_adf(node: object) -> str:
    """Extract plain text from Atlassian Document Format (ADF) node."""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("text", "")
    parts = []
    for child in node.get("content", []):
        chunk = _text_from_adf(child)
        if chunk:
            parts.append(chunk)
    sep = "\n" if node.get("type") in ("paragraph", "heading", "listItem",
                                        "bulletList", "orderedList") else " "
    return sep.join(parts).strip()


def _extract_description(fields: dict) -> str:
    desc = fields.get("description")
    if not desc:
        return ""
    if isinstance(desc, str):
        return desc
    # ADF format
    if isinstance(desc, dict):
        return _text_from_adf(desc)
    return ""


def _extract_comments(fields: dict) -> list[dict]:
    comments = []
    comment_data = fields.get("comment", {})
    for c in (comment_data.get("comments") or []):
        body = c.get("body", "")
        if isinstance(body, dict):
            body = _text_from_adf(body)
        author = (c.get("author") or {})
        comments.append({
            "author":  author.get("displayName") or author.get("emailAddress", "?"),
            "body":    body,
            "created": c.get("created", ""),
        })
    return comments


def _extract_linked_prs(fields: dict) -> list[str]:
    """Extract PR/MR URLs from issue links and remote links."""
    urls: list[str] = []
    for link in (fields.get("issuelinks") or []):
        for direction in ("outwardIssue", "inwardIssue"):
            linked = link.get(direction, {})
            if not linked:
                continue
            # Some teams link PRs as issue links
            url = linked.get("url", "")
            if url and ("pull" in url or "merge_request" in url):
                urls.append(url)
    return urls


def normalise_issue(raw: dict) -> dict:
    """Convert a raw Jira API issue dict to our storage schema."""
    key    = raw.get("key", "")
    fields = raw.get("fields", {})

    status     = (fields.get("status") or {}).get("name", "")
    resolution = (fields.get("resolution") or {}).get("name", "")
    assignee   = (fields.get("assignee") or {}).get("displayName", "")
    reporter   = (fields.get("reporter") or {}).get("displayName", "")
    issue_type = (fields.get("issuetype") or {}).get("name", "")
    priority   = (fields.get("priority") or {}).get("name", "")
    parent_key = (fields.get("parent") or {}).get("key", "")
    fix_versions = [v.get("name", "") for v in (fields.get("fixVersions") or [])]
    components   = [c.get("name", "") for c in (fields.get("components") or [])]
    labels       = fields.get("labels") or []

    return {
        "key":         key,
        "title":       fields.get("summary", ""),
        "description": _extract_description(fields),
        "status":      status,
        "resolution":  resolution,
        "issue_type":  issue_type,
        "priority":    priority,
        "assignee":    assignee,
        "reporter":    reporter,
        "labels":      labels,
        "components":  components,
        "fix_versions": fix_versions,
        "parent":      parent_key,
        "created":     fields.get("created", ""),
        "updated":     fields.get("updated", ""),
        "resolved":    fields.get("resolutiondate") or "",
        "comments":    _extract_comments(fields),
        "linked_prs":  _extract_linked_prs(fields),
        "_jira_url":   f"{JIRA_URL}/browse/{key}",
        "_fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Watermark (incremental fetch)
# ---------------------------------------------------------------------------

def load_watermark() -> Optional[str]:
    """Return the ISO timestamp of the last successful fetch, or None."""
    if WATERMARK_PATH.exists():
        try:
            data = json.loads(WATERMARK_PATH.read_text())
            return data.get("last_updated")
        except Exception:
            pass
    return None


def save_watermark(timestamp: str) -> None:
    WATERMARK_PATH.write_text(json.dumps({
        "last_updated": timestamp,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


# ---------------------------------------------------------------------------
# Fetch loop
# ---------------------------------------------------------------------------

def fetch_tickets(client: JiraClient, jql: str,
                  quiet: bool = False) -> tuple[int, int]:
    """Fetch all tickets matching jql, write each to TICKETS_DIR/<KEY>.json.

    Returns (fetched_count, updated_count).
    updated_count counts files that were written (new or changed).
    """
    TICKETS_DIR.mkdir(exist_ok=True)

    total    = None
    start_at = 0
    fetched  = 0
    updated  = 0
    newest_updated: Optional[str] = None

    while total is None or start_at < total:
        try:
            page = client.search(jql, start_at=start_at)
        except RuntimeError as e:
            print(f"  [error] {e}")
            break

        if total is None:
            total = page.get("total", 0)
            print(f"  {total} ticket(s) match the query")
            if total == 0:
                break

        issues = page.get("issues", [])
        if not issues:
            break

        for raw in issues:
            ticket = normalise_issue(raw)
            key    = ticket["key"]
            path   = TICKETS_DIR / f"{key}.json"

            # Track newest updated timestamp for watermark
            upd = ticket.get("updated", "")
            if upd and (newest_updated is None or upd > newest_updated):
                newest_updated = upd

            # Check if file content would change
            new_text = json.dumps(ticket, indent=2, ensure_ascii=False)
            if path.exists():
                try:
                    old_text = path.read_text(encoding="utf-8")
                    # Compare without _fetched_at (changes every run)
                    old_data = json.loads(old_text)
                    new_data = json.loads(new_text)
                    old_data.pop("_fetched_at", None)
                    new_data.pop("_fetched_at", None)
                    if old_data == new_data:
                        fetched += 1
                        if not quiet:
                            print(f"  {key}: unchanged")
                        continue
                except Exception:
                    pass

            path.write_text(new_text, encoding="utf-8")
            fetched += 1
            updated += 1
            n_comments = len(ticket.get("comments", []))
            if not quiet:
                print(f"  {key}: saved  ({n_comments} comments, "
                      f"status={ticket['status']}, "
                      f"resolution={ticket['resolution'] or '—'})")

        start_at += len(issues)
        if start_at < total:
            time.sleep(0.3)   # be polite to the Jira API

    return fetched, updated, newest_updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_SINCE = "-365d"   # fetch tickets updated within the past year by default
DEFAULT_STATUS_FILTER = 'status in (Done, Closed, Resolved, Fixed)'


def _build_default_jql(projects: list[str], since: str) -> str:
    """Build a sensible default JQL when the user hasn't supplied one."""
    parts = [DEFAULT_STATUS_FILTER, f'updated >= "{since}"']
    if len(projects) == 1:
        parts.insert(0, f"project = {projects[0]}")
    elif len(projects) > 1:
        keys = ", ".join(projects)
        parts.insert(0, f"project in ({keys})")
    return " AND ".join(parts)


def main():
    global TICKETS_DIR, WATERMARK_PATH  # may be overridden by --output

    parser = argparse.ArgumentParser(
        description=(
            "Fetch Jira tickets and save as individual JSON files in "
            "ticket_agent/tickets/. Each file is one ticket, human-readable "
            "and editable. Feed the directory to ticket_agent.py."
        ),
    )
    jql_group = parser.add_mutually_exclusive_group()
    jql_group.add_argument("--jql",
                           help="Full JQL query. If omitted, fetches resolved tickets "
                                f"updated in the past year (use --since to adjust).")
    jql_group.add_argument("--project", nargs="+", metavar="KEY",
                           help="One or more Jira project keys (e.g. --project ABC DEF). "
                                "Shorthand for a default JQL scoped to those projects.")
    parser.add_argument("--since", default=DEFAULT_SINCE, metavar="PERIOD",
                        help=f"How far back to look when using the default JQL "
                             f"(Jira relative date, e.g. -365d, -90d, -1y). "
                             f"Ignored if --jql is supplied. Default: {DEFAULT_SINCE}")
    parser.add_argument("--output", default=None,
                        help=f"Directory to write tickets (default: {TICKETS_DIR})")
    parser.add_argument("--incremental", action="store_true",
                        help="Only fetch tickets updated since the last run "
                             "(appends 'AND updated >= <watermark>' to JQL)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print the summary, not per-ticket status")
    args = parser.parse_args()

    # Validate env vars
    missing = [v for v, val in [("JIRA_URL", JIRA_URL),
                                 ("JIRA_EMAIL", JIRA_EMAIL),
                                 ("JIRA_TOKEN", JIRA_TOKEN)] if not val]
    if missing:
        print(f"Error: missing environment variable(s): {', '.join(missing)}")
        print("  Set them in your shell or .env file.")
        sys.exit(1)

    # Override output dir if specified
    if args.output:
        TICKETS_DIR    = Path(args.output).resolve()
        WATERMARK_PATH = TICKETS_DIR / "last_fetch.json"

    client = JiraClient(JIRA_URL, JIRA_EMAIL, JIRA_TOKEN)

    # Build JQL: explicit --jql wins; --project uses default template; bare default
    if args.jql:
        jql = args.jql
    else:
        jql = _build_default_jql(projects=args.project or [], since=args.since)
        print(f"[Default JQL] {jql}")

    # Incremental: inject updated filter
    if args.incremental:
        watermark = load_watermark()
        if watermark:
            # Jira uses date strings like "2024-01-15"
            date_str = watermark[:10]
            # Append only if not already in the JQL
            if "updated" not in jql.lower():
                jql = f"({jql}) AND updated >= \"{date_str}\""
                print(f"[Incremental] Fetching tickets updated since {date_str}")
            else:
                print(f"[Incremental] JQL already contains 'updated', using as-is")
        else:
            print("[Incremental] No watermark found — fetching all matching tickets")

    print(f"Fetching from {JIRA_URL}")
    print(f"JQL: {jql}")
    print(f"Output: {TICKETS_DIR}\n")

    fetched, updated, newest_updated = fetch_tickets(client, jql, quiet=args.quiet)

    print(f"\n[Done] {fetched} ticket(s) fetched, {updated} written/updated")
    print(f"  Tickets saved to: {TICKETS_DIR}/")
    print(f"  Next step: python -m ticket_agent.ticket_agent --tickets {TICKETS_DIR}")

    # Save watermark for next incremental run
    if newest_updated:
        save_watermark(newest_updated)
        if args.incremental:
            print(f"  Watermark updated: {newest_updated[:10]}")


if __name__ == "__main__":
    main()
