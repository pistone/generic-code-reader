"""
Indexer — feeds LLM-generated summaries into R2R.

Each entry has:
  - summary:     LLM-generated text (what gets embedded for search)
  - raw_code:    original source code (returned alongside results so Claude
                 reasons on ground truth, not the summary)
  - source_file: path to the source file
  - module:      module name (e.g. "checkers", "dataflow")
  - chunk_type:  "function_summary" | "file_summary" | "doc_summary"

Usage:
  # Index a single entry
  python indexer.py --index --file path/to/summaries.json

  # Search (smoke test)
  python indexer.py --search "null pointer handling across function calls"

  # List all indexed documents
  python indexer.py --list
"""

import argparse
import json
import sys
import time
from pathlib import Path

import os

from r2r import R2RClient

R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")


def get_client() -> R2RClient:
    return R2RClient(R2R_URL)


def index_entry(client: R2RClient, entry: dict) -> str:
    """
    Index one summary entry into R2R.

    entry keys:
      summary (str)      — text to embed
      raw_code (str)     — original source, stored in metadata
      source_file (str)  — e.g. "src/checkers/null_deref.cpp"
      module (str)       — e.g. "checkers"
      chunk_type (str)   — "function_summary" | "file_summary" | "doc_summary"
    """
    metadata = {
        "source_file": entry.get("source_file", ""),
        "module":      entry.get("module", ""),
        "chunk_type":  entry.get("chunk_type", "function_summary"),
        "raw_code":    entry.get("raw_code", ""),
    }

    response = client.documents.create(
        raw_text=entry["summary"],
        metadata=metadata,
    )
    return response.results.document_id


def index_file(path: str) -> None:
    """Index all entries from a JSON file (list of entry dicts)."""
    client = get_client()
    entries = json.loads(Path(path).read_text())

    if isinstance(entries, dict):
        entries = [entries]

    print(f"Indexing {len(entries)} entries from {path}...")
    for i, entry in enumerate(entries):
        try:
            doc_id = index_entry(client, entry)
            print(f"  [{i+1}/{len(entries)}] indexed {entry.get('source_file', '?')} → {doc_id}")
        except Exception as e:
            print(f"  [{i+1}/{len(entries)}] ERROR {entry.get('source_file', '?')}: {e}")
        # small delay to avoid overwhelming the embedding API
        time.sleep(0.1)

    print("Done.")


def search(query: str, limit: int = 5) -> None:
    """Run a search query and print results."""
    client = get_client()
    results = client.retrieval.search(query=query, search_settings={"limit": limit})

    hits = results.results.chunk_search_results
    if not hits:
        print("No results found.")
        return

    print(f"\nTop {len(hits)} results for: \"{query}\"\n")
    for i, hit in enumerate(hits):
        meta = hit.metadata or {}
        print(f"--- Result {i+1} (score: {hit.score:.3f}) ---")
        print(f"  File:   {meta.get('source_file', 'unknown')}")
        print(f"  Module: {meta.get('module', 'unknown')}")
        print(f"  Type:   {meta.get('chunk_type', 'unknown')}")
        print(f"  Summary snippet: {hit.text[:200]}...")
        if meta.get("raw_code"):
            print(f"  Raw code snippet: {meta['raw_code'][:200]}...")
        print()


def list_documents() -> None:
    """List all documents currently in R2R."""
    client = get_client()
    response = client.documents.list()
    docs = response.results
    if not docs:
        print("No documents indexed yet.")
        return
    print(f"{len(docs)} document(s) in the knowledge base:\n")
    for doc in docs:
        print(f"  {doc.id}  {doc.metadata.get('source_file', '(no file)')}  [{doc.metadata.get('chunk_type', '')}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain KB Indexer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", metavar="FILE",
                       help="JSON file of summary entries to index")
    group.add_argument("--search", metavar="QUERY",
                       help="Search the knowledge base")
    group.add_argument("--list", action="store_true",
                       help="List all indexed documents")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of search results (default: 5)")

    args = parser.parse_args()

    if args.index:
        index_file(args.index)
    elif args.search:
        search(args.search, limit=args.limit)
    elif args.list:
        list_documents()
