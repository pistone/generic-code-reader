"""
Standalone concurrent R2R indexer.

Indexes chunks (code summaries, doc summaries, ticket knowledge) into R2R
using a ThreadPoolExecutor for parallelism. Handles upsert semantics
(delete-and-replace if document already exists).

Usage as a library:
    from codebase_shared.r2r_indexer import index_entries, index_doc_chunks

Usage from CLI (for --index-only workflows):
    python -m codebase_shared.r2r_indexer --input summaries.json --source-type code
    python -m codebase_shared.r2r_indexer --input doc_summaries.json --source-type doc
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from r2r import R2RClient

R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")


def get_client() -> R2RClient:
    return R2RClient(R2R_URL)


# ── Low-level helpers ────────────────────────────────────────────────────────

def _create_or_replace(client: R2RClient, raw_text: str,
                       metadata: dict) -> str:
    """Create a document in R2R; if it already exists, delete and retry."""
    try:
        resp = client.documents.create(raw_text=raw_text, metadata=metadata)
        return str(resp.results.document_id)
    except Exception as e:
        err = str(e)
        if "already exists" not in err.lower():
            raise
        match = re.search(r"document\s+([0-9a-f-]{36})", err, re.IGNORECASE)
        if match:
            try:
                client.documents.delete(match.group(1))
            except Exception:
                pass
            resp = client.documents.create(raw_text=raw_text,
                                           metadata=metadata)
            return str(resp.results.document_id)
        raise


def purge_docs(doc_ids: list[str]) -> None:
    """Delete a list of document IDs from R2R. Ignores errors."""
    if not doc_ids:
        return
    client = get_client()
    for doc_id in doc_ids:
        try:
            client.documents.delete(doc_id)
        except Exception:
            pass


# ── Code summary indexing ────────────────────────────────────────────────────

def _index_one_code_entry(client: R2RClient,
                          entry: dict) -> tuple[str, Optional[str],
                                                Optional[str]]:
    """Index one code summary + raw_code into R2R.

    Returns (source_file, doc_id, code_doc_id).
    Entries with skip_index=True are skipped.
    """
    src_file = entry.get("source_file", "")
    module = entry.get("module", "")
    doc_id = None
    code_doc_id = None

    if entry.get("skip_index"):
        return (src_file, None, None)

    chunk_type = entry.get("chunk_type", "function_summary")
    source_kind = "reference"
    if chunk_type == "function_card":
        source_kind = "specification"
    elif chunk_type in ("file_overview", "class_overview"):
        source_kind = "overview"

    category = entry.get("chunk_category", "")
    if category == "contract":
        source_kind = "specification"
    elif category == "error_handling":
        source_kind = "operational"

    try:
        metadata = {
            "source_file": src_file,
            "module": module,
            "chunk_type": chunk_type,
            "source_type": "code",
            "source_kind": source_kind,
        }
        if category and category != "unknown":
            metadata["chunk_category"] = category
        search_val = entry.get("search_value", "")
        if search_val:
            metadata["search_value"] = search_val

        doc_id = _create_or_replace(client, entry["summary"], metadata)
    except Exception as e:
        print(f"  [warn] {src_file}: summary index failed: {e}")

    raw_code = entry.get("raw_code", "").strip()
    if raw_code and len(raw_code) > 30:
        try:
            code_doc_id = _create_or_replace(client, raw_code, {
                "source_file": src_file,
                "module": module,
                "chunk_type": "raw_code",
                "source_type": "code",
            })
        except Exception as e:
            print(f"  [warn] {src_file}: code index failed: {e}")

    return (src_file, doc_id, code_doc_id)


def index_entries(summaries: list[dict], index_workers: int = 8) -> None:
    """Index code summaries into R2R with concurrent workers.

    Mutates each entry dict to add 'doc_id' and 'code_doc_id' fields.
    Entries with skip_index=True are skipped.
    """
    client = get_client()
    skip_count = sum(1 for e in summaries if e.get("skip_index"))
    n = len(summaries)
    if skip_count:
        print(f"\n[Index] Indexing {n - skip_count} summaries into R2R "
              f"({skip_count} low-value skipped, {index_workers} workers)...")
    else:
        print(f"\n[Index] Indexing {n} summaries + raw code into R2R "
              f"({index_workers} workers)...")

    with ThreadPoolExecutor(max_workers=index_workers) as pool:
        future_to_idx = {
            pool.submit(_index_one_code_entry, client, entry): i
            for i, entry in enumerate(summaries)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                src_file, doc_id, code_doc_id = future.result()
                if doc_id:
                    summaries[idx]["doc_id"] = doc_id
                if code_doc_id:
                    summaries[idx]["code_doc_id"] = code_doc_id
            except Exception as e:
                print(f"  [warn] index failed for entry {idx}: {e}")
            done += 1
            if done % 100 == 0:
                print(f"  [{done}/{n}] indexed...")

    succeeded = sum(1 for e in summaries if e.get("doc_id"))
    failed = n - succeeded
    if failed > 0:
        print(f"[Index] Done: {succeeded}/{n} indexed ({failed} failed)")
    else:
        print(f"[Index] Done: {succeeded}/{n} indexed")


# ── Doc chunk indexing ───────────────────────────────────────────────────────

def _index_one_doc_chunk(client: R2RClient, chunk: dict,
                         last_modified: str) -> Optional[str]:
    """Index a single doc chunk into R2R. Returns doc_id or None."""
    index_text = chunk.get("summary") or chunk["text"]
    source_kind = chunk.get("source_kind", "")

    metadata = {
        "source_file":   chunk["source_file"],
        "module":        "documentation",
        "chunk_type":    "doc_summary",
        "doc_title":     chunk.get("doc_title", ""),
        "source_type":   "doc",
        "source_kind":   source_kind,
        "last_modified": last_modified,
    }
    return _create_or_replace(client, index_text, metadata)


def index_doc_chunks(chunks: list[dict], last_modified: str = "",
                     index_workers: int = 8) -> list[str]:
    """Index doc chunks into R2R with concurrent workers.

    Returns list of successfully created document IDs.
    """
    client = get_client()
    n = len(chunks)
    print(f"[Index] Indexing {n} doc chunks into R2R ({index_workers} workers)...")

    doc_ids: list[Optional[str]] = [None] * n

    with ThreadPoolExecutor(max_workers=index_workers) as pool:
        future_to_idx = {
            pool.submit(_index_one_doc_chunk, client, chunk, last_modified): i
            for i, chunk in enumerate(chunks)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                doc_ids[idx] = future.result()
            except Exception as e:
                print(f"  [warn] chunk {idx}: index failed: {e}")
            done += 1
            if done % 100 == 0:
                print(f"  [{done}/{n}] indexed...")

    succeeded = sum(1 for d in doc_ids if d)
    print(f"[Index] Done: {succeeded}/{n} doc chunks indexed")
    return [d for d in doc_ids if d is not None]


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Index summaries or doc chunks into R2R",
    )
    parser.add_argument("--input", required=True,
                        help="Path to JSON file containing entries to index")
    parser.add_argument("--source-type", choices=["code", "doc"],
                        default="code",
                        help="Type of entries: 'code' for summaries.json, "
                             "'doc' for doc_summaries.json (default: code)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent indexing workers (default: 8)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    data = json.loads(input_path.read_text())
    if not isinstance(data, list):
        print(f"Error: expected a JSON array, got {type(data).__name__}")
        sys.exit(1)

    print(f"Loaded {len(data)} entries from {input_path}")

    if args.source_type == "code":
        index_entries(data, index_workers=args.workers)
        # Write back doc_ids
        input_path.write_text(json.dumps(data, indent=2))
        print(f"Updated {input_path} with doc_ids")
    else:
        doc_ids = index_doc_chunks(data, index_workers=args.workers)
        print(f"Indexed {len(doc_ids)} doc chunks")


if __name__ == "__main__":
    main()
