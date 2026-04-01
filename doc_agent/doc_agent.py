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
import asyncio
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
    TokenTracker, llm_call, allm_call, load_manifest, save_manifest,
    _is_quota_error, AsyncRateLimiter,
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
    """Summarize a single doc chunk via LLM (sync). Returns updated chunk dict."""
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


async def _async_summarize_chunk(model: str, chunk: dict,
                                  tracker: Optional[TokenTracker] = None) -> dict:
    """Summarize a single doc chunk via async LLM call. Returns updated chunk dict."""
    prompt = SUMMARIZE_PROMPT.format(
        doc_title=chunk["doc_title"],
        heading=chunk.get("heading", ""),
        source_file=chunk["source_file"],
        text=chunk["text"][:3000],
    )
    try:
        raw = await allm_call(model, SUMMARIZE_SYSTEM, prompt,
                              max_tokens=256, json_mode=True,
                              tracker=tracker, phase="DocSummary")
        result = json.loads(raw)
        chunk["summary"] = result.get("summary", "")
        chunk["source_kind"] = result.get("source_kind", "overview")
    except (json.JSONDecodeError, TypeError) as e:
        print(f"  [warn] summarize failed for {chunk.get('source_file', '?')}: {e}")
        chunk["summary"] = chunk["text"][:200].strip()
        chunk["source_kind"] = "overview"
    return chunk


async def _summarize_chunks_async(
    model: str,
    all_file_chunks: list[tuple[str, str, list[dict]]],
    tracker: Optional[TokenTracker] = None,
    max_concurrent: int = 20,
    rpm: int = 500,
    quiet: bool = False,
) -> tuple[list[tuple[str, str, list[dict]]], bool]:
    """Summarize all chunks across all files concurrently.

    Args:
        all_file_chunks: list of (source_path, content_hash, chunks) tuples
        Returns: (completed file chunks, quota_hit flag)
    """
    limiter = AsyncRateLimiter(max_concurrent=max_concurrent,
                               calls_per_minute=rpm)

    total_chunks = sum(len(fc[2]) for fc in all_file_chunks)
    done = 0

    # Print every N chunks, scaling with total (at least every 5)
    print_interval = max(1, min(10, total_chunks // 10))

    async def _do_one(chunk: dict) -> dict:
        nonlocal done
        result = await _async_summarize_chunk(model, chunk, tracker=tracker)
        done += 1
        if not quiet and (done % print_interval == 0 or done == total_chunks):
            print(f"  [summarize] {done}/{total_chunks} chunks done")
        return result

    # Build tasks: (coroutine_factory, callback) pairs for run_many
    tasks = []
    chunk_to_file: dict[int, int] = {}  # chunk index → file index
    flat_chunks = []
    for file_idx, (path, chash, chunks) in enumerate(all_file_chunks):
        for chunk in chunks:
            chunk_idx = len(flat_chunks)
            flat_chunks.append(chunk)
            chunk_to_file[chunk_idx] = file_idx
            tasks.append(chunk)

    # Use gather with rate limiting
    async def _run_one(chunk: dict):
        if limiter._halted:
            return
        try:
            await limiter._wait_for_slot()
            async with limiter._semaphore:
                await _do_one(chunk)
        except Exception as e:
            err_str = str(e).lower()
            if _is_quota_error(err_str):
                limiter.halt("Quota exhausted")
                limiter.quota_exhausted = True
            else:
                print(f"  [warn] LLM error: {e}")

    await asyncio.gather(*[_run_one(c) for c in flat_chunks],
                         return_exceptions=True)

    if not quiet:
        print(f"  [summarize] {done}/{total_chunks} chunks done")

    return all_file_chunks, limiter.quota_exhausted


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
    parser.add_argument("--no-index", action="store_true",
                        help="Parse and summarize only — write doc_summaries.json "
                             "but do NOT index into R2R. Use the indexer separately.")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip parsing/summarizing — read doc_summaries.json "
                             "and index into R2R.")
    parser.add_argument("--incremental", action="store_true",
                        help="Only process docs changed since last run")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress (each chunk, LLM responses)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-chunk progress")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Max concurrent LLM calls (default: 20)")
    parser.add_argument("--rpm", type=int, default=500,
                        help="Rate limit: requests per minute (default: 500)")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent
    manifest_path = output_dir / "doc_hashes.json"
    cost_log_path = output_dir / "cost_log.jsonl"
    summaries_path = output_dir / "doc_summaries.json"
    manifest = load_manifest(manifest_path)
    tracker = TokenTracker()

    # ── Index-only mode: read doc_summaries.json and push to R2R ────────
    if args.index_only:
        if not summaries_path.exists():
            print(f"Error: {summaries_path} not found. Run without --index-only first.")
            sys.exit(1)
        all_chunks = json.loads(summaries_path.read_text())
        print(f"[Index-only] Indexing {len(all_chunks)} doc chunks from {summaries_path}")
        # Group by source_file for purge + manifest update
        by_file: dict[str, list[dict]] = {}
        for chunk in all_chunks:
            by_file.setdefault(chunk["source_file"], []).append(chunk)
        for source_file, chunks in by_file.items():
            purge_old_docs(source_file, manifest)
            last_mod = chunks[0].get("last_modified", "")
            doc_ids = index_chunks(chunks, last_mod)
            print(f"  {source_file}: {len(chunks)} chunk(s) indexed")
            manifest[source_file] = {
                "hash": manifest.get(source_file, {}).get("hash", ""),
                "doc_ids": doc_ids,
            }
            save_manifest(manifest_path, manifest)
        print(f"[Index-only] Done — {len(all_chunks)} chunks indexed into R2R")
        return

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

    # 3. Parse → chunk all files first
    total_chunks = 0
    total_files = len(raw_docs)
    file_work: list[tuple[RawDocument, str, list[dict], str]] = []  # (raw, hash, chunks, last_mod)

    for file_idx, raw in enumerate(raw_docs, 1):
        current_hash = file_hash(raw.content_bytes)
        existing = manifest.get(raw.path, {})
        if (existing.get("hash") == current_hash
                and existing.get("doc_ids")
                and not args.incremental):
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
        file_work.append((raw, current_hash, chunks, last_mod))

    skipped = total_files - len(file_work)
    if skipped > 0:
        print(f"  ({skipped} file(s) already done, skipped)")

    quota_hit = False
    if not file_work:
        print("All documents already processed.")
    else:
        all_chunks_count = sum(len(fw[2]) for fw in file_work)
        print(f"\n  {len(file_work)} file(s), {all_chunks_count} chunk(s) to process")

        # 4. Async LLM summarization (all chunks concurrently)
        quota_hit = False
        if use_llm:
            all_file_chunks = [(fw[0].path, fw[1], fw[2]) for fw in file_work]
            _, quota_hit = asyncio.run(
                _summarize_chunks_async(
                    args.model, all_file_chunks,
                    tracker=tracker,
                    max_concurrent=args.max_concurrent,
                    rpm=args.rpm,
                    quiet=args.quiet,
                )
            )

        # 5. Fill in missing summaries with raw text fallback
        for raw, current_hash, chunks, last_mod in file_work:
            missing = sum(1 for c in chunks if not c.get("summary"))
            if missing:
                if quota_hit:
                    print(f"  [skip] {raw.path}: {missing}/{len(chunks)} chunks "
                          f"unsummarized (quota), will retry next run")
                    continue
                for c in chunks:
                    if not c.get("summary"):
                        c["summary"] = c["text"][:300].strip()
                        c["source_kind"] = c.get("source_kind", "overview")
                print(f"  [warn] {raw.path}: {missing}/{len(chunks)} chunks "
                      f"used raw text fallback")
            # Stamp last_modified into each chunk for later indexing
            for c in chunks:
                c["last_modified"] = last_mod

        # 5b. Save doc_summaries.json (always — this is the intermediate artifact)
        all_output_chunks = []
        for raw, current_hash, chunks, last_mod in file_work:
            for c in chunks:
                if c.get("summary"):
                    all_output_chunks.append(c)
        if all_output_chunks:
            summaries_path.write_text(json.dumps(all_output_chunks, indent=2))
            print(f"\n  Wrote {len(all_output_chunks)} chunks to {summaries_path}")

        # 6. Index into R2R (unless --no-index)
        if args.no_index:
            print(f"\n[No-index] Skipping R2R indexing. "
                  f"Run with --index-only to index later.")
        else:
            for raw, current_hash, chunks, last_mod in file_work:
                if not any(c.get("summary") for c in chunks):
                    continue  # skip files with no summaries (quota hit)

                purge_old_docs(raw.path, manifest)
                doc_ids = index_chunks(chunks, last_mod)
                total_chunks += len(chunks)
                print(f"  {raw.path}: {len(chunks)} chunk(s) indexed")

                manifest[raw.path] = {
                    "hash": current_hash,
                    "doc_ids": doc_ids,
                }
                save_manifest(manifest_path, manifest)

    # Final manifest save
    save_manifest(manifest_path, manifest)

    # Token tracking
    if tracker.phases:
        print(f"\n{tracker.summary()}")
        entry = tracker.to_log_entry(model=args.model, agent="doc_agent")
        with cost_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    if args.no_index:
        print(f"\n[Done] {len(all_output_chunks) if file_work else 0} chunks "
              f"from {len(raw_docs)} doc(s) written to {summaries_path}")
    elif file_work and quota_hit:
        indexed = sum(1 for r, h, c, l in file_work
                      if manifest.get(r.path, {}).get("doc_ids"))
        print(f"\n[Paused] {total_chunks} chunks from {indexed} doc(s) "
              f"indexed. Re-run to continue ({len(file_work) - indexed} "
              f"remaining).")
    else:
        print(f"\n[Done] {total_chunks} chunks from {len(raw_docs)} doc(s) "
              f"indexed into R2R")


if __name__ == "__main__":
    main()
