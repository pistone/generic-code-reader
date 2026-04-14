"""
Domain KB — MCP Server

Exposes tools to Claude Code:
  search_codebase(query)   — semantic search over the knowledge base
  add_to_kb(...)           — index a new entry immediately
  kb_status()              — health check and stats
  list_modules()           — show indexed modules

Backend is selected via KB_BACKEND env var:
  r2r   — R2R vector DB (default, needs Docker)
  local — ChromaDB local file DB (no server needed)

For a local-only version with no R2R dependency, use server_local.py instead.

Start with:
  /path/to/.venv/bin/python mcp_server/server.py
"""

import os
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from mcp_server.common import (
    SEARCH_LIMIT, format_results, log_query, log_user_entry,
    get_module_map_text, user_contribution_count,
)

# ── Backend selection ────────────────────────────────────────────────────────

KB_BACKEND = os.getenv("KB_BACKEND", "r2r")

if KB_BACKEND == "local":
    from mcp_server import backend_local as backend
else:
    from mcp_server import backend_r2r as backend

# ── MCP Server ───────────────────────────────────────────────────────────────

mcp = FastMCP("domain-kb")


@mcp.tool()
def search_codebase(query: str) -> str:
    """
    Search the domain knowledge base semantically.

    Returns the most relevant code summaries, documentation, and ticket
    knowledge for the given query. Treat results as a starting point,
    not a final answer — investigate further if needed.

    Args:
        query:  Natural language description of what you're looking for.
                E.g. "how does the null dereference checker handle pointer arithmetic"
    """
    try:
        hits = backend.search(query, SEARCH_LIMIT)
    except Exception as e:
        return f"⚠ Knowledge base search failed: {e}\n{backend.error_hint()}"

    formatted = format_results(hits)

    # Cross-source linking
    if hits:
        source_types_seen = {
            (getattr(h, "metadata", None) or {}).get("source_type", "")
            for h in hits
        }
        other_types = {"code", "doc", "ticket"} - source_types_seen
        if other_types and len(source_types_seen) == 1:
            cross_note = _cross_source_check(query, other_types)
            if cross_note:
                formatted += "\n\n" + cross_note

    approx_tokens = len(formatted) // 4
    result_files = []
    for h in hits:
        meta = getattr(h, "metadata", None) or {}
        src = meta.get("source_file", "")
        if src and src not in result_files:
            result_files.append(src)

    log_query(query, len(hits), approx_tokens,
              answer=formatted, result_files=result_files)
    return formatted


def _cross_source_check(query: str, source_types: set[str]) -> str:
    """Check if other source types have related knowledge."""
    notes = []
    for st in source_types:
        try:
            hits = backend.search_by_type(query, st, limit=1)
            if hits and getattr(hits[0], "score", 0) > 0.3:
                meta = getattr(hits[0], "metadata", None) or {}
                source = meta.get("source_file", "?")
                title = meta.get("doc_title", "")
                label = {"code": "Code", "doc": "Documentation",
                         "ticket": "Ticket knowledge"}.get(st, st)
                ref = f"`{source}`"
                if title:
                    ref += f" ({title})"
                notes.append(f"- **{label}**: {ref}")
        except Exception:
            pass
    if notes:
        return "**Related knowledge in other sources:**\n" + "\n".join(notes)
    return ""


@mcp.tool()
def add_to_kb(
    topic: str,
    summary: str,
    source_files: list[str],
    reasoning: str,
    raw_code: str = "",
) -> str:
    """
    Add a new entry to the domain knowledge base.

    Call this when you've researched something that wasn't in the KB and
    found the answer. The entry is indexed immediately and available to
    the whole team right away.

    Args:
        topic:        Short label for the entry (e.g. "null deref checker — pointer arithmetic")
        summary:      Domain-aware description of what you found. Should answer
                      the question clearly using domain vocabulary.
        source_files: List of source file paths you read to find the answer.
        reasoning:    Why this is worth adding — what question does it answer?
        raw_code:     The key source code snippet that answers the question.
    """
    src_file = source_files[0] if source_files else "unknown"
    ts = datetime.now(timezone.utc).isoformat()
    full_text = f"[{topic}] {summary}"

    doc_id = ""
    code_doc_id = ""

    try:
        doc_id = backend.add_entry(full_text, {
            "source_file": src_file,
            "module":      "user_contributed",
            "chunk_type":  "function_summary",
            "source_type": "code",
            "source_kind": "reference",
            "origin":      "suggestion",
        })
        if raw_code and len(raw_code.strip()) > 30:
            try:
                code_doc_id = backend.add_entry(raw_code, {
                    "source_file": src_file,
                    "module":      "user_contributed",
                    "chunk_type":  "raw_code",
                    "source_type": "code",
                    "origin":      "suggestion",
                })
            except Exception:
                pass
    except Exception as e:
        return f"⚠ Failed to index: {e}"

    log_user_entry({
        "topic": topic, "summary": summary, "source_files": source_files,
        "raw_code": raw_code, "reasoning": reasoning,
        "doc_id": doc_id, "code_doc_id": code_doc_id,
        "status": "indexed", "timestamp": ts,
    })

    return "✓ Indexed into KB. Available to the team now."


@mcp.tool()
def kb_status() -> str:
    """Check knowledge base health and statistics.

    Returns indexed content counts, service health, and pending suggestions.
    Use this to verify the KB is working before searching.
    """
    lines = [f"Backend: {KB_BACKEND}"]
    lines.extend(backend.status())

    n = user_contribution_count()
    if n:
        lines.append(f"\nUser contributions: {n}")

    return "\n".join(lines)


@mcp.tool()
def list_modules() -> str:
    """List all indexed modules with their descriptions.

    Returns the module map from the last study_agent run, showing
    what areas of the codebase have been analyzed.
    """
    return get_module_map_text()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
