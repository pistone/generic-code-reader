"""
tools/docs.py — Documentation ingestion tool.

index_docs: Parses, summarizes, and indexes documentation files into R2R.
            Supports .md, .html, .rst, .txt files.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from codebase_shared.utils import TokenTracker, load_manifest, save_manifest


def index_docs(
    paths: list[str],
    model: str = None,
    reindex: bool = False,
    tracker: Optional[TokenTracker] = None,
) -> dict:
    """Parse, summarize, and index documentation into R2R.

    Calls the underlying doc_agent functions directly (not via argparse).

    Args:
        paths:   List of file or directory paths. Directories are recursively
                 scanned for .md/.html/.rst/.txt files. Files are indexed directly.
        model:   LLM model for summarization (default: LLM_MODEL env var).
        reindex: Force re-indexing even if already indexed.
        tracker: Optional token tracker.

    Returns:
        dict with doc_count, chunk_count, indexed_count.
    """
    from doc_agent.doc_agent import (
        chunk_document, summarize_chunk, index_chunks, purge_old_docs,
        file_hash, _summarize_chunks_async, DEFAULT_MODEL,
    )
    from doc_agent.parsers import get_parser
    from doc_agent.sources import LocalFileSource, RawDocument

    _model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    output_dir     = _ROOT / "doc_agent"
    manifest_path  = output_dir / "doc_hashes.json"
    summaries_path = output_dir / "doc_summaries.json"
    manifest = load_manifest(manifest_path)

    # Collect raw documents
    raw_docs: list[RawDocument] = []
    for p in paths:
        resolved = Path(p).resolve()
        if resolved.is_file():
            # Single file: create a RawDocument directly
            content = resolved.read_bytes()
            from datetime import datetime
            import stat
            mtime = datetime.fromtimestamp(resolved.stat().st_mtime)
            raw_docs.append(RawDocument(
                path=str(resolved),
                content_bytes=content,
                last_modified=mtime,
            ))
        elif resolved.is_dir():
            source = LocalFileSource(resolved)
            found = list(source.list_documents())
            print(f"[index_docs] {len(found)} file(s) under {resolved}")
            raw_docs.extend(found)
        else:
            print(f"[index_docs] Warning: {resolved} not found, skipping")

    if not raw_docs:
        print("[index_docs] No document files found.")
        return {"doc_count": 0, "chunk_count": 0, "indexed_count": 0}

    # Parse and chunk
    file_work = []
    skipped = 0
    for raw in raw_docs:
        current_hash = file_hash(raw.content_bytes)
        existing = manifest.get(raw.path, {})
        if not reindex and existing.get("hash") == current_hash and existing.get("doc_ids"):
            skipped += 1
            continue

        suffix = Path(raw.path).suffix.lower()
        parser_inst = get_parser(suffix)
        try:
            parsed = parser_inst.parse(raw.path, raw.content_bytes, raw.last_modified)
        except Exception as e:
            print(f"  [warn] {raw.path}: parse failed: {e}")
            continue

        chunks = chunk_document(parsed)
        last_mod = raw.last_modified.isoformat() if raw.last_modified else ""
        file_work.append((raw, current_hash, chunks, last_mod))

    if skipped:
        print(f"[index_docs] {skipped} file(s) already indexed (use reindex=True to force)")

    if not file_work:
        print("[index_docs] Nothing new to index.")
        return {"doc_count": 0, "chunk_count": 0, "indexed_count": skipped}

    total_chunks = sum(len(fw[2]) for fw in file_work)
    print(f"[index_docs] {len(file_work)} file(s), {total_chunks} chunk(s) to summarize")

    # Async LLM summarization
    all_file_chunks = [(fw[0].path, fw[1], fw[2]) for fw in file_work]
    quota_hit = asyncio.run(
        _summarize_chunks_async(
            _model, all_file_chunks,
            tracker=tracker or TokenTracker(),
            max_concurrent=20,
            rpm=500,
            quiet=False,
        )
    )

    # Index into R2R and update manifest
    # Re-read summarized chunks from the summaries file written by _summarize_chunks_async
    indexed_total = 0
    if summaries_path.exists():
        import json
        all_summaries = json.loads(summaries_path.read_text())
        # Group by source_file for the newly processed files
        new_paths = {fw[0].path for fw in file_work}
        for source_file in new_paths:
            chunks = [c for c in all_summaries if c.get("source_file") == source_file]
            if not chunks:
                continue
            purge_old_docs(source_file, manifest)
            last_mod = next((fw[3] for fw in file_work if fw[0].path == source_file), "")
            doc_ids = index_chunks(chunks, last_mod)
            manifest[source_file] = {
                "hash": next(fw[1] for fw in file_work if fw[0].path == source_file),
                "doc_ids": doc_ids,
            }
            save_manifest(manifest_path, manifest)
            indexed_total += len(chunks)

    if quota_hit:
        print("[index_docs] Warning: quota exhausted before all docs were processed")

    print(f"[index_docs] Done: {len(file_work)} files, {total_chunks} chunks, "
          f"{indexed_total} indexed")
    return {
        "doc_count": len(file_work),
        "chunk_count": total_chunks,
        "indexed_count": indexed_total,
        "quota_hit": quota_hit,
    }
