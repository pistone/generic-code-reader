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
try:
    SEARCH_LIMIT = int(os.getenv("KB_SEARCH_LIMIT", "5"))
except (ValueError, TypeError):
    SEARCH_LIMIT = 5

BASE_DIR      = Path(__file__).parent
STAGING_FILE  = BASE_DIR / "staging_queue.json"
LOG_FILE      = BASE_DIR / "query_log.jsonl"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client() -> R2RClient:
    return R2RClient(R2R_URL)


def _log_query(query: str, num_results: int, approx_tokens: int) -> None:
    """Append one line to the query log for experiment measurement."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "num_results": num_results,
            "approx_tokens": approx_tokens,
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging is non-essential, don't crash the search


def _format_results(hits: list) -> str:
    """Format R2R search hits into a readable string for Claude."""
    if not hits:
        return "No results found in the knowledge base."

    parts = []
    for i, hit in enumerate(hits):
        meta       = hit.metadata or {}
        text       = hit.text or ""
        source     = meta.get("source_file", "unknown")
        module     = meta.get("module", "unknown")
        chunk_type = meta.get("chunk_type", "")
        score      = getattr(hit, "score", 0)

        header = f"## Result {i+1}  (score: {score:.3f})"
        file_line = f"**File**: `{source}`  |  **Module**: `{module}`"

        if chunk_type == "raw_code":
            section = [header, file_line, "", "**Source code**:", f"```\n{text}\n```"]
        else:
            section = [header, file_line, "", "**Summary**:", text]

        parts.append("\n".join(section))

    return "\n\n---\n\n".join(parts)


def _load_staging() -> list:
    if STAGING_FILE.exists():
        try:
            data = json.loads(STAGING_FILE.read_text())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, Exception):
            pass
    return []


def _save_staging(queue: list) -> None:
    STAGING_FILE.write_text(json.dumps(queue, indent=2))


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP("domain-kb")


@mcp.tool()
def search_codebase(query: str, module: str = "",
                    source_type: str = "") -> str:
    """
    Search the domain knowledge base semantically.

    Returns the most relevant code summaries and source snippets for the
    given query. Use this before reading files directly — it may already
    contain the answer with much less token cost.

    Args:
        query:  Natural language description of what you're looking for.
                E.g. "how does the null dereference checker handle pointer arithmetic"
        module: Optional module name to restrict the search scope.
                E.g. "checkers", "dataflow". Leave empty to search everything.
        source_type: Optional filter by knowledge source type.
                     "code" = code summaries, "doc" = design docs/wiki,
                     "ticket" = Jira/PR knowledge. Leave empty for all.
    """
    client = get_client()
    search_settings: dict = {
        "limit": SEARCH_LIMIT,
        "use_hybrid_search": True,
    }
    filters: dict = {}
    if module:
        filters["module"] = {"$eq": module}
    if source_type:
        filters["source_type"] = {"$eq": source_type}
    if filters:
        search_settings["filters"] = filters

    try:
        results = client.retrieval.search(
            query=query,
            search_settings=search_settings,
        )
        hits = results.results.chunk_search_results
    except Exception as e:
        return f"⚠ Knowledge base search failed: {e}\nIs R2R running? Check: curl {R2R_URL}/v3/health"

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
    raw_code: str = "",
    module: str = "",
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
        raw_code:     The key source code snippet that answers the question.
                      This gets stored alongside the summary so future searches
                      return ground truth, not just the summary.
        module:       Module this entry belongs to (e.g. "checkers", "dataflow").
                      Used for module-scoped search filtering.
    """
    queue = _load_staging()
    entry = {
        "topic":        topic,
        "summary":      summary,
        "source_files": source_files,
        "raw_code":     raw_code,
        "module":       module,
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
