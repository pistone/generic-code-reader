#!/usr/bin/env python3
"""Doc agent: ingest documents into R2R for knowledge-base search.

Parses .md/.html/.txt/.rst files, chunks by sections, and indexes
raw text directly into R2R (no LLM summarization — docs are already
human-readable prose).  Runs BEFORE study_agent so code summarization
can pull doc knowledge via RAG.

Usage:
    python -m doc_agent.doc_agent --docs /path/to/docs
    python -m doc_agent.doc_agent --docs /path/to/docs --incremental
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from r2r import R2RClient

from doc_agent.parsers import ParsedDocument, get_parser
from doc_agent.sources import LocalFileSource, RawDocument

R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")
MAX_SECTION_CHARS = 4000


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_document(doc: ParsedDocument,
                   max_chars: int = MAX_SECTION_CHARS) -> list[dict]:
    """Split a ParsedDocument into chunks suitable for R2R indexing.

    Strategy:
      - If the document has headings: each section → one chunk.
        Sections longer than *max_chars* are sub-chunked by paragraph.
      - If there are no headings: paragraph-based chunking.
    """
    has_headings = any(s.level > 0 for s in doc.sections)
    chunks: list[dict] = []

    if has_headings:
        for section in doc.sections:
            text = section.content.strip()
            if not text:
                continue
            heading = section.heading
            if len(text) <= max_chars:
                chunks.append(_chunk_dict(text, heading, doc))
            else:
                for sub in _split_by_paragraphs(text, max_chars):
                    chunks.append(_chunk_dict(sub, heading, doc))
    else:
        full = doc.text.strip()
        if full:
            for sub in _split_by_paragraphs(full, max_chars):
                chunks.append(_chunk_dict(sub, "", doc))

    # fallback: always produce at least one chunk
    if not chunks:
        chunks.append(_chunk_dict(doc.text[:max_chars], "", doc))

    return chunks


def _chunk_dict(text: str, heading: str, doc: ParsedDocument) -> dict:
    return {
        "text": text,
        "heading": heading,
        "doc_title": doc.title,
        "source_file": doc.source_file,
    }


def _split_by_paragraphs(text: str, max_chars: int) -> list[str]:
    """Accumulate paragraphs (split by blank line) up to *max_chars*."""
    paragraphs = text.split("\n\n")
    result: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            result.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        result.append(current.strip())
    return result


# ---------------------------------------------------------------------------
# R2R indexing
# ---------------------------------------------------------------------------

def index_chunks(chunks: list[dict], last_modified: str) -> list[str]:
    """Index all chunks into R2R.  Returns list of document IDs."""
    client = R2RClient(R2R_URL)
    doc_ids: list[str] = []

    for i, chunk in enumerate(chunks):
        try:
            resp = client.documents.create(
                raw_text=chunk["text"],
                metadata={
                    "source_file":   chunk["source_file"],
                    "module":        "documentation",
                    "chunk_type":    "doc_summary",
                    "doc_title":     chunk["doc_title"],
                    "source_type":   "doc",
                    "last_modified": last_modified,
                },
            )
            doc_ids.append(str(resp.results.document_id))
        except Exception as e:
            print(f"  [warn] chunk {i}: index failed: {e}")

        if i > 0 and i % 20 == 0:
            time.sleep(0.5)

    return doc_ids


def purge_old_docs(source_file: str, manifest: dict) -> None:
    """Delete previously indexed docs for a source file."""
    old_ids = manifest.get(source_file, {}).get("doc_ids", [])
    if not old_ids:
        return
    client = R2RClient(R2R_URL)
    for doc_id in old_ids:
        try:
            client.documents.delete(doc_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Incremental mode
# ---------------------------------------------------------------------------

def file_hash(content_bytes: bytes) -> str:
    """SHA256 hash truncated to 16 hex chars."""
    return hashlib.sha256(content_bytes).hexdigest()[:16]


def load_manifest(path: Path) -> dict:
    """Load the hash + doc_id manifest from disk."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Doc agent: parse documents → chunk by section → index to R2R",
    )
    parser.add_argument("--docs", required=True,
                        help="Root directory of documents to ingest")
    parser.add_argument("--incremental", action="store_true",
                        help="Only process docs changed since last run")
    args = parser.parse_args()

    docs_path = Path(args.docs).resolve()
    if not docs_path.is_dir():
        print(f"Error: '{docs_path}' is not a directory")
        sys.exit(1)

    output_dir = Path(__file__).resolve().parent
    manifest_path = output_dir / "doc_hashes.json"
    manifest = load_manifest(manifest_path)

    # 1. Collect raw documents
    source = LocalFileSource(docs_path)
    raw_docs = list(source.list_documents())
    print(f"Found {len(raw_docs)} document file(s) under {docs_path}")
    if not raw_docs:
        print("No document files found.")
        return

    # 2. Filter to changed docs (incremental mode)
    if args.incremental:
        changed: list[RawDocument] = []
        for doc in raw_docs:
            h = file_hash(doc.content_bytes)
            old_hash = manifest.get(doc.path, {}).get("hash", "")
            if h != old_hash:
                changed.append(doc)
        if not changed:
            print("[Incremental] No documents changed since last run.")
            return
        print(f"[Incremental] {len(changed)}/{len(raw_docs)} docs changed")
        raw_docs = changed

    # 3. Parse → chunk → index
    total_chunks = 0
    for raw in raw_docs:
        suffix = Path(raw.path).suffix.lower()
        parser_inst = get_parser(suffix)
        try:
            parsed = parser_inst.parse(raw.path, raw.content_bytes,
                                       raw.last_modified)
        except Exception as e:
            print(f"  [warn] {raw.path}: parse failed: {e}")
            continue

        chunks = chunk_document(parsed)
        last_mod = (raw.last_modified.isoformat()
                    if raw.last_modified else "")

        # purge old indexed docs before re-indexing
        purge_old_docs(raw.path, manifest)

        print(f"  {raw.path}: {len(chunks)} chunk(s)")
        doc_ids = index_chunks(chunks, last_mod)
        total_chunks += len(chunks)

        # update manifest
        manifest[raw.path] = {
            "hash": file_hash(raw.content_bytes),
            "doc_ids": doc_ids,
        }

    # 4. Save manifest
    save_manifest(manifest_path, manifest)
    print(f"\n[Done] {total_chunks} chunks from {len(raw_docs)} doc(s) "
          f"indexed into R2R")


if __name__ == "__main__":
    main()
