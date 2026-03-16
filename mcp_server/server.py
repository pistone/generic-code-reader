"""
Domain KB — MCP Server

Exposes two tools to Claude Code:
  search_codebase(query)          — semantic search over the knowledge base
  suggest_index_item(...)         — queue a new KB entry for review

Logs every search query to mcp_server/query_log.jsonl for experiment
measurement (query, tokens returned, timestamp).

Start with:
  /path/to/.venv/bin/python mcp_server/server.py
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from r2r import R2RClient

# ── Config ────────────────────────────────────────────────────────────────────

R2R_URL       = os.getenv("R2R_URL", "http://localhost:7272")
SEARCH_LIMIT  = int(os.getenv("KB_SEARCH_LIMIT", "5"))

BASE_DIR      = Path(__file__).parent
STAGING_FILE  = BASE_DIR / "staging_queue.json"
LOG_FILE      = BASE_DIR / "query_log.jsonl"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client() -> R2RClient:
    return R2RClient(R2R_URL)


def _log_query(query: str, num_results: int, approx_tokens: int) -> None:
    """Append one line to the query log for experiment measurement."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "num_results": num_results,
        "approx_tokens": approx_tokens,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _format_results(hits: list) -> str:
    """Format R2R search hits into a readable string for Claude."""
    if not hits:
        return "No results found in the knowledge base."

    parts = []
    for i, hit in enumerate(hits):
        meta     = hit.metadata or {}
        summary  = hit.text or ""
        raw_code = meta.get("raw_code", "")
        source   = meta.get("source_file", "unknown")
        module   = meta.get("module", "unknown")
        score    = getattr(hit, "score", 0)

        section = [
            f"## Result {i+1}  (score: {score:.3f})",
            f"**File**: `{source}`  |  **Module**: `{module}`",
            f"",
            f"**Summary**:",
            summary,
        ]
        if raw_code:
            section += ["", "**Source code**:", f"```\n{raw_code}\n```"]

        parts.append("\n".join(section))

    return "\n\n---\n\n".join(parts)


def _load_staging() -> list:
    if STAGING_FILE.exists():
        return json.loads(STAGING_FILE.read_text())
    return []


def _save_staging(queue: list) -> None:
    STAGING_FILE.write_text(json.dumps(queue, indent=2))


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP("domain-kb")


@mcp.tool()
def search_codebase(query: str) -> str:
    """
    Search the domain knowledge base semantically.

    Returns the most relevant code summaries and source snippets for the
    given query. Use this before reading files directly — it may already
    contain the answer with much less token cost.

    Args:
        query: Natural language description of what you're looking for.
               E.g. "how does the null dereference checker handle pointer arithmetic"
    """
    client = get_client()
    results = client.retrieval.search(
        query=query,
        search_settings={"limit": SEARCH_LIMIT},
    )
    hits = results.results.chunk_search_results

    # Approximate token count for logging (rough: 1 token ≈ 4 chars)
    formatted = _format_results(hits)
    approx_tokens = len(formatted) // 4
    _log_query(query, len(hits), approx_tokens)

    return formatted


@mcp.tool()
def suggest_index_item(
    topic: str,
    summary: str,
    source_files: list[str],
    reasoning: str,
) -> str:
    """
    Suggest a new entry for the domain knowledge base.

    Call this when you've manually researched something that wasn't in the
    KB and found the answer. Your suggestion will be reviewed and, if
    approved, will be available to the whole team instantly.

    Args:
        topic:        Short label for the entry (e.g. "null deref checker — pointer arithmetic")
        summary:      Domain-aware description of what you found. Should answer
                      the question clearly using domain vocabulary.
        source_files: List of source file paths you read to find the answer.
        reasoning:    Why this is worth adding — what question does it answer?
    """
    queue = _load_staging()
    entry = {
        "topic":        topic,
        "summary":      summary,
        "source_files": source_files,
        "reasoning":    reasoning,
        "status":       "pending",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }
    queue.append(entry)
    _save_staging(queue)

    return (
        f"Suggestion queued (#{len(queue)} in staging). "
        "A reviewer agent will verify and promote it to the knowledge base."
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
