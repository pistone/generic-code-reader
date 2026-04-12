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
LOG_FILE      = BASE_DIR / "query_log.jsonl"

# STAGING_FILE: override with STAGING_FILE env var for team deployments.
# Point all teammates at the same shared path (e.g. a network drive or
# the server's local path when running the reviewer centrally).
STAGING_FILE  = Path(os.getenv("STAGING_FILE", str(BASE_DIR / "staging_queue.json")))

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client() -> R2RClient:
    return R2RClient(R2R_URL)


def _log_query(query: str, num_results: int, approx_tokens: int,
               answer: str = "",
               result_files: list[str] | None = None) -> None:
    """Append one line to the query log for experiment measurement.

    Logs both the question and the full answer so that queries can be
    replayed against a different KB (or no KB) for A/B comparison.
    """
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
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
    client = get_client()

    search_settings: dict = {
        "limit": SEARCH_LIMIT,
        "use_hybrid_search": True,
    }

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
    if hits:
        source_types_seen = {
            (getattr(h, "metadata", None) or {}).get("source_type", "")
            for h in hits
        }
        other_types = {"code", "doc", "ticket"} - source_types_seen
        if other_types and len(source_types_seen) == 1:
            cross_note = _cross_source_check(client, query, other_types)

    full_output = formatted
    if cross_note:
        full_output += "\n\n" + cross_note
    approx_tokens = len(full_output) // 4

    result_files = []
    for h in hits:
        meta = h.metadata or {}
        src = meta.get("source_file", "")
        if src and src not in result_files:
            result_files.append(src)

    _log_query(query, len(hits), approx_tokens,
               answer=full_output, result_files=result_files)

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
                      This gets stored alongside the summary so future searches
                      return ground truth, not just the summary.
    """
    client = get_client()

    # Primary source file for metadata (first in list, or "unknown")
    src_file = source_files[0] if source_files else "unknown"
    ts = datetime.now(timezone.utc).isoformat()

    # Index the summary immediately into R2R
    doc_id = ""
    code_doc_id = ""
    try:
        resp = client.documents.create(
            raw_text=f"[{topic}] {summary}",
            metadata={
                "source_file": src_file,
                "module":      "user_contributed",
                "chunk_type":  "function_summary",
                "source_type": "code",
                "source_kind": "reference",
                "origin":      "suggestion",
            },
        )
        doc_id = str(resp.results.document_id)
    except Exception as e:
        return f"⚠ Failed to index: {e}\nIs R2R running? Check: curl {R2R_URL}/v3/health"

    # Index raw code as a separate searchable chunk
    if raw_code and len(raw_code.strip()) > 30:
        try:
            resp = client.documents.create(
                raw_text=raw_code,
                metadata={
                    "source_file": src_file,
                    "module":      "user_contributed",
                    "chunk_type":  "raw_code",
                    "source_type": "code",
                    "origin":      "suggestion",
                },
            )
            code_doc_id = str(resp.results.document_id)
        except Exception:
            pass  # non-fatal

    # Log to staging for audit trail (not a review queue)
    entry = {
        "topic":        topic,
        "summary":      summary,
        "source_files": source_files,
        "raw_code":     raw_code,
        "reasoning":    reasoning,
        "doc_id":       doc_id,
        "code_doc_id":  code_doc_id,
        "status":       "indexed",
        "timestamp":    ts,
    }
    queue = _load_staging()
    queue.append(entry)
    _save_staging(queue)

    return f"✓ Indexed into KB (doc_id: {doc_id}). Available to the team now."


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
