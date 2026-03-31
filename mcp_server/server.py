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

# Scope → source_kind mapping.  Each scope boosts certain source_kinds
# and optionally filters source_type.
SCOPE_PROFILES = {
    "implementation": {
        "description": "How is X implemented? Find code details.",
        "boost_source_types": ["code"],
        "boost_source_kinds": ["reference"],
    },
    "rationale": {
        "description": "Why was X designed this way? Find design decisions.",
        "boost_source_types": ["doc", "ticket"],
        "boost_source_kinds": ["rationale"],
    },
    "howto": {
        "description": "How do I accomplish X? Find tutorials and examples.",
        "boost_source_kinds": ["tutorial", "operational"],
    },
    "troubleshooting": {
        "description": "How to fix/debug X? Find workarounds and root causes.",
        "boost_source_types": ["ticket"],
        "boost_source_kinds": ["operational", "rationale"],
    },
    "architecture": {
        "description": "What is the high-level design of X?",
        "boost_source_kinds": ["overview", "specification"],
        "boost_chunk_types": ["file_overview", "class_overview"],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client() -> R2RClient:
    return R2RClient(R2R_URL)


def _log_query(query: str, num_results: int, approx_tokens: int,
               answer: str = "", scope: str = "", module: str = "",
               source_type: str = "",
               result_files: list[str] | None = None) -> None:
    """Append one line to the query log for experiment measurement.

    Logs both the question and the full answer so that queries can be
    replayed against a different KB (or no KB) for A/B comparison.
    """
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "scope": scope,
            "module": module,
            "source_type": source_type,
            "num_results": num_results,
            "approx_tokens": approx_tokens,
            "result_files": result_files or [],
            "answer": answer,
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
        meta        = hit.metadata or {}
        text        = hit.text or ""
        source      = meta.get("source_file", "unknown")
        module      = meta.get("module", "unknown")
        chunk_type  = meta.get("chunk_type", "")
        source_type = meta.get("source_type", "")
        source_kind = meta.get("source_kind", "")
        score       = getattr(hit, "score", 0)

        # Build a type tag like [code/method] or [doc/rationale] or [ticket/root_cause]
        type_parts = []
        if source_type:
            type_parts.append(source_type)
        if source_kind:
            type_parts.append(source_kind)
        elif chunk_type:
            type_parts.append(chunk_type)
        type_tag = "/".join(type_parts) if type_parts else "unknown"

        header = f"## Result {i+1}  [{type_tag}]  (score: {score:.3f})"
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
    """Atomic write to prevent corruption from concurrent calls."""
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=STAGING_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(queue, f, indent=2)
        os.replace(tmp_path, STAGING_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP("domain-kb")


@mcp.tool()
def search_codebase(query: str, module: str = "",
                    source_type: str = "",
                    scope: str = "") -> str:
    """
    Search the domain knowledge base semantically.

    Returns the most relevant code summaries, documentation, and ticket
    knowledge for the given query. Use this before reading files directly
    — it may already contain the answer with much less token cost.

    MULTI-HOP SEARCH: A single search may return results that reference
    names, IDs, or identifiers you don't yet understand. When this
    happens, do a FOLLOW-UP search for that unresolved name to find what
    it maps to. Two searches are almost always enough to resolve indirect
    references. Combine the results in your final answer.

    Args:
        query:  Natural language description of what you're looking for.
                E.g. "how does the null dereference checker handle pointer arithmetic"
        module: Optional module name to restrict the search scope.
                E.g. "checkers", "dataflow". Leave empty to search everything.
        source_type: Optional filter by knowledge source type.
                     "code" = code summaries, "doc" = design docs/wiki,
                     "ticket" = Jira/PR knowledge. Leave empty for all.
        scope:  Optional intent-based scope that adjusts which kinds of
                results are prioritized. Options:
                  "implementation" — how is X implemented? (boosts code)
                  "rationale" — why was X designed this way? (boosts docs/tickets)
                  "howto" — how do I accomplish X? (boosts tutorials)
                  "troubleshooting" — how to fix/debug X? (boosts tickets)
                  "architecture" — what is the high-level design? (boosts overviews)
                Leave empty to search everything equally.
    """
    client = get_client()
    profile = SCOPE_PROFILES.get(scope, {}) if scope else {}

    # --- Primary search ---
    search_settings: dict = {
        "limit": SEARCH_LIMIT,
        "use_hybrid_search": True,
    }
    filters: dict = {}
    if module:
        filters["module"] = {"$eq": module}

    # source_type explicitly set by user takes precedence over scope
    if source_type:
        filters["source_type"] = {"$eq": source_type}
    elif profile.get("boost_source_types"):
        # Scope suggests source types — filter to them
        types = profile["boost_source_types"]
        if len(types) == 1:
            filters["source_type"] = {"$eq": types[0]}
        else:
            filters["source_type"] = {"$in": types}

    if profile.get("boost_source_kinds"):
        kinds = profile["boost_source_kinds"]
        if len(kinds) == 1:
            filters["source_kind"] = {"$eq": kinds[0]}
        else:
            filters["source_kind"] = {"$in": kinds}

    if profile.get("boost_chunk_types"):
        ctypes = profile["boost_chunk_types"]
        if len(ctypes) == 1:
            filters["chunk_type"] = {"$eq": ctypes[0]}
        else:
            filters["chunk_type"] = {"$in": ctypes}

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

    formatted = _format_results(hits)

    # --- Cross-source linking ---
    # If we got results from only one source type, check if other sources
    # have related knowledge and add a brief note
    cross_note = ""
    if hits and scope != "implementation":
        source_types_seen = {
            (getattr(h, "metadata", None) or {}).get("source_type", "")
            for h in hits
        }
        # Look for related results in other source types
        other_types = {"code", "doc", "ticket"} - source_types_seen
        if other_types and len(source_types_seen) == 1:
            cross_note = _cross_source_check(client, query, other_types)

    # Approximate token count for logging (rough: 1 token ≈ 4 chars)
    full_output = formatted
    if cross_note:
        full_output += "\n\n" + cross_note
    approx_tokens = len(full_output) // 4

    # Extract result file list for evaluation
    result_files = []
    for h in hits:
        meta = h.metadata or {}
        src = meta.get("source_file", "")
        if src and src not in result_files:
            result_files.append(src)

    _log_query(query, len(hits), approx_tokens,
               answer=full_output, scope=scope, module=module,
               source_type=source_type, result_files=result_files)

    return full_output


def _cross_source_check(client: R2RClient, query: str,
                        source_types: set[str]) -> str:
    """Check if other source types have related knowledge. Returns a note."""
    notes = []
    for st in source_types:
        try:
            results = client.retrieval.search(
                query=query,
                search_settings={
                    "limit": 1,
                    "use_hybrid_search": True,
                    "filters": {"source_type": {"$eq": st}},
                },
            )
            hits = results.results.chunk_search_results
            if hits and getattr(hits[0], "score", 0) > 0.3:
                meta = hits[0].metadata or {}
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
        "Run `python reviewer/reviewer_agent.py --codebase /path/to/src` to review, "
        "or use `--watch` for continuous review."
    )


@mcp.tool()
def kb_status() -> str:
    """Check knowledge base health and statistics.

    Returns indexed content counts, service health, and pending suggestions.
    Use this to verify the KB is working before searching.
    """
    lines: list[str] = []

    # R2R health
    try:
        client = get_client()
        health = "healthy"
    except Exception as e:
        health = f"error: {e}"
    lines.append(f"R2R: {health} ({R2R_URL})")

    # Count indexed documents by source type
    try:
        client = get_client()
        # Use a broad search to estimate counts
        for stype in ["code", "doc", "ticket"]:
            try:
                results = client.retrieval.search(
                    query="*",
                    search_settings={
                        "limit": 1,
                        "filters": {"source_type": {"$eq": stype}},
                    },
                )
                hits = getattr(results, "results", results) if results else []
                chunk_results = getattr(hits, "chunk_search_results", hits) if hits else []
                count = "1+" if chunk_results else "0"
                lines.append(f"  {stype}: {count} entries")
            except Exception:
                lines.append(f"  {stype}: unknown")
    except Exception:
        lines.append("  Could not query R2R for counts")

    # Pending suggestions
    queue = _load_staging()
    pending = sum(1 for e in queue if e.get("status") == "pending")
    if pending:
        lines.append(f"\nPending suggestions: {pending}")
        lines.append("Run reviewer agent to process them.")

    return "\n".join(lines)


@mcp.tool()
def list_modules() -> str:
    """List all indexed modules with their descriptions.

    Returns the module map from the last study_agent run, showing
    what areas of the codebase have been analyzed.
    """
    module_map_path = Path(__file__).resolve().parent.parent / "indexer" / "module_map.json"
    if not module_map_path.exists():
        return "No module map found. Run study_agent first."

    try:
        data = json.loads(module_map_path.read_text())
    except Exception as e:
        return f"Could not read module map: {e}"

    project = data.get("project", "Unknown")
    desc = data.get("description", "")
    modules = data.get("modules", [])

    lines = [f"**{project}**: {desc}", f"{len(modules)} modules:\n"]
    for mod in modules:
        name = mod.get("name", "?")
        mod_desc = mod.get("description", "")
        files = mod.get("files", [])
        questions = mod.get("questions", [])
        lines.append(f"- **{name}** ({len(files)} files): {mod_desc}")
        if questions:
            for q in questions[:2]:
                lines.append(f"  - {q}")
            if len(questions) > 2:
                lines.append(f"  - ...and {len(questions) - 2} more questions")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
