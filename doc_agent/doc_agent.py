#!/usr/bin/env python3
"""Doc agent: ingest documents into R2R for knowledge-base search.

Parses .md/.html/.txt/.rst files, chunks by sections, optionally
summarizes each chunk via LLM (classifying its source_kind), and
indexes into R2R.  Runs BEFORE study_agent so code summarization
can pull doc knowledge via RAG.

Usage:
    python -m doc_agent.doc_agent --docs /path/to/docs
    python -m doc_agent.doc_agent --docs /path/to/docs --incremental
    python -m doc_agent.doc_agent --docs /path/to/docs --model openai/gpt-4o-mini
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from r2r import R2RClient

from doc_agent.parsers import ParsedDocument, get_parser
from doc_agent.sources import LocalFileSource, RawDocument

# Shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codebase_shared.utils import (  # noqa: E402
    TokenTracker, llm_call, load_manifest, save_manifest,
    _is_quota_error,
)

R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
MAX_SECTION_CHARS = 4000


# ---------------------------------------------------------------------------
# LLM summarization
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM = (
    "You summarize documentation chunks for a knowledge base. "
    "Produce a concise, search-friendly summary that preserves domain terms, "
    "key decisions, and actionable details. Also classify the chunk's kind. "
    "Output ONLY valid JSON — no markdown fences, no commentary."
)

SUMMARIZE_PROMPT = """\
Document: {doc_title}
Section: {heading}
Source: {source_file}

--- content ---
{text}
--- end ---

Summarize this documentation chunk in 2-4 sentences. Preserve domain terms
and key details so the summary is useful for semantic search.

Also classify the chunk's source_kind as exactly one of:
- "specification" — defines behavior, API contracts, formats, protocols
- "rationale" — explains why a design decision was made
- "tutorial" — step-by-step instructions, how-to guides
- "operational" — runbooks, deployment, configuration, troubleshooting
- "reference" — API reference, parameter lists, tables of values
- "overview" — high-level architecture, module descriptions

Output this JSON:
{{
  "summary": "your 2-4 sentence summary",
  "source_kind": "one of the categories above"
}}"""


def summarize_chunk(model: str, chunk: dict,
                    tracker: Optional[TokenTracker] = None) -> dict:
    """Summarize a single doc chunk via LLM. Returns updated chunk dict."""
    prompt = SUMMARIZE_PROMPT.format(
        doc_title=chunk["doc_title"],
        heading=chunk.get("heading", ""),
        source_file=chunk["source_file"],
        text=chunk["text"][:3000],  # cap to control cost
    )
    try:
        raw = llm_call(model, SUMMARIZE_SYSTEM, prompt,
                       max_tokens=256, json_mode=True,
                       tracker=tracker, phase="DocSummary")
        result = json.loads(raw)
        chunk["summary"] = result.get("summary", "")
        chunk["source_kind"] = result.get("source_kind", "overview")
    except (json.JSONDecodeError, TypeError) as e:
        # LLM returned malformed JSON — use first 200 chars of text as summary
        print(f"  [warn] summarize failed for {chunk.get('source_file', '?')}: {e}")
        chunk["summary"] = chunk["text"][:200].strip()
        chunk["source_kind"] = "overview"
    return chunk


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
    """Accumulate paragraphs (split by blank line) up to *max_chars*.

    Single paragraphs exceeding *max_chars* are hard-split at the limit
    so no chunk is ever larger than *max_chars*.
    """
    paragraphs = text.split("\n\n")
    result: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            result.append(current.strip())
            current = ""
        # Hard-split a single paragraph that exceeds max_chars
        while len(para) > max_chars:
            result.append(para[:max_chars])
            para = para[max_chars:]
        current = f"{current}\n\n{para}" if current else para
    if current.strip():
        result.append(current.strip())
    return result


# ---------------------------------------------------------------------------
# R2R indexing
# ---------------------------------------------------------------------------

def _index_one_chunk(client, chunk: dict, last_modified: str) -> Optional[str]:
    """Index a single chunk into R2R. Returns doc_id or None."""
    # If LLM summary available, index that; otherwise fall back to raw text
    index_text = chunk.get("summary") or chunk["text"]
    source_kind = chunk.get("source_kind", "")

    resp = client.documents.create(
        raw_text=index_text,
        metadata={
            "source_file":   chunk["source_file"],
            "module":        "documentation",
            "chunk_type":    "doc_summary",
            "doc_title":     chunk["doc_title"],
            "source_type":   "doc",
            "source_kind":   source_kind,
            "last_modified": last_modified,
        },
    )
    return str(resp.results.document_id)


def index_chunks(chunks: list[dict], last_modified: str,
                 index_workers: int = 8) -> list[str]:
    """Index all chunks into R2R using parallel workers.  Returns list of document IDs."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = R2RClient(R2R_URL)
    doc_ids: list[Optional[str]] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=index_workers) as pool:
        future_to_idx = {
            pool.submit(_index_one_chunk, client, chunk, last_modified): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                doc_ids[idx] = future.result()
            except Exception as e:
                print(f"  [warn] chunk {idx}: index failed: {e}")

    return [d for d in doc_ids if d is not None]


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


# load_manifest / save_manifest imported from codebase_shared.utils


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Doc agent: parse documents → chunk → summarize → index to R2R",
    )
    parser.add_argument("--docs", required=True, nargs="+",
                        help="Root directory/directories of documents to ingest. "
                             "Can specify multiple: --docs /path/a /path/b")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="litellm model string for summarization "
                             "(default: %(default)s). "
                             "Use --no-summarize to skip LLM calls entirely.")
    parser.add_argument("--no-summarize", action="store_true",
                        help="Index raw text without LLM summarization "
                             "(faster, cheaper, but lower search quality)")
    parser.add_argument("--incremental", action="store_true",
                        help="Only process docs changed since last run")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress (each chunk, LLM responses)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-chunk progress")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent
    manifest_path = output_dir / "doc_hashes.json"
    cost_log_path = output_dir / "cost_log.jsonl"
    manifest = load_manifest(manifest_path)
    tracker = TokenTracker()

    # 1. Collect raw documents from all doc paths
    raw_docs = []
    for doc_arg in args.docs:
        docs_path = Path(doc_arg).resolve()
        if not docs_path.is_dir():
            print(f"Warning: '{docs_path}' is not a directory, skipping")
            continue
        source = LocalFileSource(docs_path)
        found = list(source.list_documents())
        print(f"Found {len(found)} document file(s) under {docs_path}")
        raw_docs.extend(found)

    if not raw_docs:
        print("No document files found in any of the provided paths.")
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

    use_llm = not args.no_summarize
    if use_llm:
        print(f"LLM summarization enabled (model: {args.model})")
    else:
        print("LLM summarization disabled (--no-summarize)")

    # 3. Parse → chunk → (summarize) → index
    #    Manifest is saved after each file so we can resume on quota exhaustion.
    total_chunks = 0
    total_files = len(raw_docs)
    quota_hit = False
    for file_idx, raw in enumerate(raw_docs, 1):
        # Skip files already completed (resume support)
        current_hash = file_hash(raw.content_bytes)
        existing = manifest.get(raw.path, {})
        if (existing.get("hash") == current_hash
                and existing.get("doc_ids")
                and not args.incremental):
            # Already processed with same content — skip
            if not args.quiet:
                print(f"  [{file_idx}/{total_files}] {raw.path}: "
                      f"already done, skipping")
            continue

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

        # LLM summarization pass
        quota_hit = False
        if use_llm:
            for i, chunk in enumerate(chunks):
                try:
                    summarize_chunk(args.model, chunk, tracker=tracker)
                except Exception as e:
                    err_str = str(e).lower()
                    if _is_quota_error(err_str):
                        print(f"\n⚠ Quota exhausted at {raw.path} "
                              f"chunk {i+1}/{len(chunks)}")
                        print(f"  Progress saved. Re-run to resume.")
                        quota_hit = True
                        break
                    # Transient errors (timeout, network) — skip chunk, continue
                    print(f"  [warn] {raw.path} chunk {i+1}: LLM error: {e}")
                    continue
                if not args.quiet:
                    kind = chunk.get("source_kind", "?")
                    print(f"  [{file_idx}/{total_files}] "
                          f"{raw.path} chunk {i+1}/{len(chunks)}: {kind}")
                time.sleep(0.2)  # basic rate limiting

        if quota_hit:
            # Don't index partial file — leave it for next run
            break

        # purge old indexed docs before re-indexing
        purge_old_docs(raw.path, manifest)

        print(f"  [{file_idx}/{total_files}] {raw.path}: "
              f"{len(chunks)} chunk(s)")
        doc_ids = index_chunks(chunks, last_mod)
        total_chunks += len(chunks)

        # Update manifest and save immediately (resume support)
        manifest[raw.path] = {
            "hash": current_hash,
            "doc_ids": doc_ids,
        }
        save_manifest(manifest_path, manifest)

    # 4. Final manifest save (redundant but explicit)
    save_manifest(manifest_path, manifest)

    # Token tracking
    if tracker.phases:
        print(f"\n{tracker.summary()}")
        entry = tracker.to_log_entry(model=args.model, agent="doc_agent")
        with cost_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    if quota_hit:
        print(f"\n[Paused] {total_chunks} chunks from {file_idx - 1} doc(s) "
              f"indexed. Re-run to continue from file {file_idx}/{total_files}.")
    else:
        print(f"\n[Done] {total_chunks} chunks from {len(raw_docs)} doc(s) "
              f"indexed into R2R")


if __name__ == "__main__":
    main()
