"""
Local KB backend using ChromaDB.

Drop-in replacement for R2R semantic search. No Docker, no server —
just a local directory with embeddings.

Usage:
    from codebase_shared.local_kb import LocalKB

    kb = LocalKB("/path/to/chroma_db")
    kb.add_entries(summaries, source_type="code")
    results = kb.search("how does auth work", limit=5)
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb

# Default embedding function uses chromadb's built-in (all-MiniLM-L6-v2),
# no API key needed.  Override with CHROMA_EMBEDDING env var if desired.
DEFAULT_DB_DIR = "chroma_db"


@dataclass
class SearchHit:
    """Mirrors the R2R hit interface so the MCP server can use either backend."""
    text: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class LocalKB:
    """ChromaDB-backed local knowledge base."""

    COLLECTION = "domain_kb"

    def __init__(self, db_dir: Optional[str] = None):
        db_path = db_dir or os.getenv("LOCAL_KB_DIR", DEFAULT_DB_DIR)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    # ── Loading ──────────────────────────────────────────────────────────

    def add_entries(self, entries: list[dict], source_type: str = "code",
                    show_progress: bool = True) -> int:
        """Add entries to the collection. Returns number added.

        Each entry needs at least a 'summary' field. Entries with
        skip_index=True are skipped.
        """
        docs: list[str] = []
        metas: list[dict] = []
        ids: list[str] = []

        for entry in entries:
            if entry.get("skip_index"):
                continue
            text = entry.get("summary", "").strip()
            if not text:
                continue

            meta = {
                "source_file":  entry.get("source_file", ""),
                "module":       entry.get("module", ""),
                "chunk_type":   entry.get("chunk_type", ""),
                "source_type":  source_type,
                "source_kind":  entry.get("source_kind", ""),
                "search_value": entry.get("search_value", ""),
                "doc_title":    entry.get("doc_title", ""),
            }
            # ChromaDB metadata values must be str, int, float, or bool
            meta = {k: v for k, v in meta.items() if v}

            # Stable ID based on content to allow idempotent re-loads
            doc_id = _stable_id(text, meta)

            docs.append(text)
            metas.append(meta)
            ids.append(doc_id)

        if not docs:
            return 0

        # ChromaDB upsert handles duplicates gracefully
        batch_size = 500
        added = 0
        for i in range(0, len(docs), batch_size):
            batch_end = min(i + batch_size, len(docs))
            self._collection.upsert(
                documents=docs[i:batch_end],
                metadatas=metas[i:batch_end],
                ids=ids[i:batch_end],
            )
            added += batch_end - i
            if show_progress and added % 1000 == 0:
                print(f"  [{added}/{len(docs)}] loaded...", flush=True)

        return added

    def add_doc_chunks(self, chunks: list[dict]) -> int:
        """Add doc agent chunks. Adapts the doc format to the standard entry format."""
        entries = []
        for chunk in chunks:
            text = chunk.get("summary") or chunk.get("text", "")
            if not text:
                continue
            entries.append({
                "summary":     text,
                "source_file": chunk.get("source_file", ""),
                "module":      "documentation",
                "chunk_type":  "doc_summary",
                "source_kind": chunk.get("source_kind", ""),
                "doc_title":   chunk.get("doc_title", ""),
            })
        return self.add_entries(entries, source_type="doc")

    def add_ticket_entries(self, tickets: list[dict]) -> int:
        """Add ticket agent entries. Combines summary + solution into one searchable text."""
        entries = []
        for t in tickets:
            parts = []
            if t.get("summary"):
                parts.append(t["summary"])
            if t.get("solution"):
                parts.append(f"Solution: {t['solution']}")
            if t.get("lesson"):
                parts.append(t["lesson"])
            if not parts:
                continue

            category = t.get("category", "general")
            source_kind = {
                "root_cause": "rationale",
                "design_decision": "rationale",
                "workaround": "operational",
                "pattern": "reference",
            }.get(category, "reference")

            entries.append({
                "summary":     "\n".join(parts),
                "source_file": t.get("key", ""),
                "module":      category,
                "chunk_type":  "ticket_summary",
                "source_kind": source_kind,
            })
        return self.add_entries(entries, source_type="ticket")

    def add_user_entries(self, entries: list[dict]) -> int:
        """Add user-contributed entries from staging_queue / user_contributed.jsonl."""
        adapted = []
        for e in entries:
            if e.get("status") not in ("indexed", "approved", None):
                continue
            text = e.get("summary", "").strip()
            if not text:
                continue
            topic = e.get("topic", "")
            if topic:
                text = f"[{topic}] {text}"
            src = e.get("source_files", ["unknown"])
            adapted.append({
                "summary":     text,
                "source_file": src[0] if src else "unknown",
                "module":      "user_contributed",
                "chunk_type":  "function_summary",
                "source_kind": "reference",
            })
        return self.add_entries(adapted, source_type="code")

    def clear(self):
        """Delete all entries."""
        self._client.delete_collection(self.COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 5,
               source_type: Optional[str] = None) -> list[SearchHit]:
        """Semantic search. Returns list of SearchHit (same interface as R2R hits)."""
        where = None
        if source_type:
            where = {"source_type": {"$eq": source_type}}

        results = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if not results["documents"] or not results["documents"][0]:
            return hits

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns cosine distance; convert to similarity score
            score = max(0.0, 1.0 - dist)
            hits.append(SearchHit(text=doc, score=score, metadata=meta))

        return hits

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return counts by source_type."""
        total = self._collection.count()
        counts = {}
        for st in ("code", "doc", "ticket"):
            try:
                r = self._collection.get(
                    where={"source_type": {"$eq": st}},
                    include=[],
                )
                counts[st] = len(r["ids"]) if r["ids"] else 0
            except Exception:
                counts[st] = 0
        counts["total"] = total
        return counts


def _stable_id(text: str, meta: dict) -> str:
    """Generate a deterministic ID from content + metadata."""
    key = json.dumps({"t": text[:2000], "m": meta}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:24]
