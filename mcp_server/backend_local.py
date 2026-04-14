"""Local ChromaDB backend for the MCP server. No Docker or server needed."""

import os
import sys
from pathlib import Path

LOCAL_KB_DIR = os.getenv("LOCAL_KB_DIR", str(Path(__file__).parent.parent / "chroma_db"))

_kb = None


def _get_kb():
    global _kb
    if _kb is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from codebase_shared.local_kb import LocalKB
        _kb = LocalKB(LOCAL_KB_DIR)
    return _kb


def search(query: str, limit: int) -> list:
    return _get_kb().search(query, limit=limit)


def search_by_type(query: str, source_type: str, limit: int = 1) -> list:
    return _get_kb().search(query, limit=limit, source_type=source_type)


def add_entry(text: str, metadata: dict) -> str:
    """Index a document into local ChromaDB. Returns a synthetic doc_id."""
    n = _get_kb().add_entries([{
        "summary":     text,
        "source_file": metadata.get("source_file", ""),
        "module":      metadata.get("module", ""),
        "chunk_type":  metadata.get("chunk_type", ""),
        "source_kind": metadata.get("source_kind", ""),
        "doc_title":   metadata.get("doc_title", ""),
    }], source_type=metadata.get("source_type", "code"))
    return f"local:{n}"


def status() -> list[str]:
    lines = []
    try:
        kb = _get_kb()
        stats = kb.stats()
        lines.append(f"Local DB: {LOCAL_KB_DIR}")
        lines.append(f"  Total: {stats.get('total', 0)} entries")
        for st in ("code", "doc", "ticket"):
            lines.append(f"  {st}: {stats.get(st, 0)}")
    except Exception as e:
        lines.append(f"Local DB error: {e}")
    return lines


def error_hint() -> str:
    return f"Check: ls {LOCAL_KB_DIR}"
