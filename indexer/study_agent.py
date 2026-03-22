"""
Study Agent — Multi-pass codebase analysis for the domain knowledge base.

Uses litellm so it works with ANY provider:
  OpenAI:    --model openai/gpt-4o        (needs OPENAI_API_KEY)
  Ollama:    --model ollama/llama3.1      (no key, runs locally)
  Anthropic: --model anthropic/claude-opus-4-6  (needs ANTHROPIC_API_KEY)
  Groq:      --model groq/llama3-70b-8192 (needs GROQ_API_KEY)

Pass 1 — Module Discovery
  - Walks the directory tree and reads file samples
  - Calls the LLM to understand overall architecture
  - Generates per-module domain-specific question lists
  - Output: module_map.json

Pass 2 — Summarization
  - Loads module_map.json
  - Chunks each file with CodeSplitter (tree-sitter AST boundaries)
  - Calls the LLM with the module's questions to generate domain-aware summaries
  - With --rag: queries the vector DB before each chunk for richer context
  - Output: summaries.json  →  fed to indexer.py (or auto-indexed with --passes)

Pass 3 — Review  (runs automatically when --passes > 1)
  - LLM reviews each summary for accuracy and domain vocabulary
  - Rewrites weak summaries; improves the vector DB in-place
  - Stops early when edit rate drops below 5% (convergence)

Usage:
  # Standard two-pass run (same as before)
  OPENAI_API_KEY=sk-... python study_agent.py --codebase /path/to/src

  # Bootstrap design docs into the vector DB, then run with RAG augmentation
  python study_agent.py --codebase /path/to/src \\
      --docs /path/to/docs --bootstrap-docs --rag

  # Three iterative passes (summarize → review → improve) until convergence
  python study_agent.py --codebase /path/to/src \\
      --docs /path/to/docs --bootstrap-docs --rag --passes 3

  # Skip Pass 1 if module_map.json already exists
  python study_agent.py --codebase /path/to/src --pass2-only
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL        = os.getenv("LLM_MODEL", "openai/gpt-4o")
R2R_URL              = os.getenv("R2R_URL", "http://localhost:7272")
CONVERGENCE_THRESHOLD = 0.05   # stop iterating when < 5% of summaries are edited

# Max lines to sample from each file for Pass 1
SAMPLE_LINES = 80

# CodeSplitter settings
CHUNK_LINES         = 60   # target lines per chunk
CHUNK_LINES_OVERLAP = 10   # overlap to preserve context across boundaries
MAX_CHARS           = 4000  # hard cap per chunk

# Files / dirs to skip
SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll",
                 ".egg-info", ".dist-info", ".lock"}
SKIP_DIRS    = {"__pycache__", ".git", ".hg", ".svn", "node_modules",
                ".venv", "venv", "env", ".env", "build", "dist",
                ".mypy_cache", ".pytest_cache", ".tox"}

# ── Pydantic models (used to validate Pass 1 JSON output) ─────────────────────

class ModuleDefinition(BaseModel):
    name:        str
    description: str
    files:       list[str]
    questions:   list[str]

class ModuleMap(BaseModel):
    project:     str
    description: str
    modules:     list[ModuleDefinition]

# ── LLM helper ────────────────────────────────────────────────────────────────

def llm_call(model: str, system: str, user: str,
             max_tokens: int = 4096,
             json_mode: bool = False,
             stream: bool = False):
    """
    Unified LLM call via litellm.
    Returns the full response text (or a generator of text chunks if stream=True).
    """
    from litellm import completion

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, stream=stream)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = completion(**kwargs)

    if stream:
        # Return a generator that yields text chunks
        def _gen():
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        return _gen()
    else:
        return response.choices[0].message.content or ""

# ── File utilities ─────────────────────────────────────────────────────────────

def collect_source_files(codebase: Path, language: str = "python") -> list[Path]:
    """Return all source files under codebase, filtering out build artifacts."""
    ext_map = {
        "python": {".py"},
        "javascript": {".js", ".mjs"},
        "typescript": {".ts", ".tsx"},
        "cpp": {".cpp", ".cc", ".cxx", ".h", ".hpp"},
        "java": {".java"},
        "go": {".go"},
        "rust": {".rs"},
    }
    exts = ext_map.get(language, {".py"})

    results = []
    for p in sorted(codebase.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in SKIP_SUFFIXES:
            continue
        if p.suffix in exts:
            results.append(p)
    return results


def file_hash(path: Path) -> str:
    """Fast content hash of a source file (sha256, hex-truncated to 16 chars)."""
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except Exception:
        return ""
    return h.hexdigest()[:16]


def load_hash_manifest(path: Path) -> dict[str, str]:
    """Load {relative_path: hash} from a previous run."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_hash_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def filter_changed_files(codebase: Path, files: list[Path],
                         manifest_path: Path) -> tuple[list[Path], dict[str, str]]:
    """
    Compare current file hashes against a saved manifest.
    Returns (changed_files, new_manifest).
    Only returns files whose content has changed since the last run.
    """
    old_manifest = load_hash_manifest(manifest_path)
    new_manifest: dict[str, str] = {}
    changed: list[Path] = []

    for f in files:
        rel = str(f.relative_to(codebase))
        h = file_hash(f)
        new_manifest[rel] = h
        if h != old_manifest.get(rel, ""):
            changed.append(f)

    return changed, new_manifest


def build_directory_tree(codebase: Path, files: list[Path]) -> str:
    """Build a compact directory tree string from the list of files."""
    lines = [str(codebase.name) + "/"]
    for rp in sorted(f.relative_to(codebase) for f in files):
        indent = "  " * (len(rp.parts) - 1)
        lines.append(f"{indent}  {rp.name}")
    return "\n".join(lines)


def read_file_sample(path: Path, max_lines: int = SAMPLE_LINES) -> str:
    """Read up to max_lines from a file, handling encoding errors."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        sampled = lines[:max_lines]
        if len(lines) > max_lines:
            sampled.append(f"... ({len(lines) - max_lines} more lines)")
        return "\n".join(sampled)
    except Exception as e:
        return f"[could not read: {e}]"


def chunk_file(path: Path, language: str = "python") -> list[str]:
    """
    Split a source file at AST boundaries using CodeSplitter.
    Falls back to line-based chunking if tree-sitter fails.
    """
    try:
        from llama_index.core import Document
        from llama_index.core.node_parser import CodeSplitter

        content = path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return []

        doc = Document(text=content)
        splitter = CodeSplitter(
            language=language,
            chunk_lines=CHUNK_LINES,
            chunk_lines_overlap=CHUNK_LINES_OVERLAP,
            max_chars=MAX_CHARS,
        )
        nodes = splitter.get_nodes_from_documents([doc])
        chunks = [n.get_content() for n in nodes if n.get_content().strip()]
        return chunks if chunks else [content[:MAX_CHARS]]

    except Exception as e:
        # Fallback: simple line-based chunking
        print(f"  [warn] CodeSplitter failed for {path.name}: {e} — using line split")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        result = []
        for i in range(0, len(lines), CHUNK_LINES):
            chunk = "\n".join(lines[i : i + CHUNK_LINES + CHUNK_LINES_OVERLAP])
            if chunk.strip():
                result.append(chunk[:MAX_CHARS])
        return result or []


# ── R2R helpers (bootstrap, RAG search, auto-index) ───────────────────────────

def _r2r_client():
    from r2r import R2RClient
    return R2RClient(R2R_URL)


def _chunk_doc(content: str, max_chars: int = 1500) -> list[str]:
    """Split a doc at markdown headings; fall back to fixed-size chunks."""
    import re
    sections = re.split(r'\n(?=#{1,3} )', content)
    chunks = []
    for section in sections:
        if len(section) <= max_chars:
            if section.strip():
                chunks.append(section.strip())
        else:
            for i in range(0, len(section), max_chars):
                part = section[i : i + max_chars].strip()
                if part:
                    chunks.append(part)
    return chunks or [content[:max_chars]]


def bootstrap_docs(docs_path: Path) -> int:
    """
    Index .md/.rst/.txt files from docs_path into R2R as doc_summary chunks.
    Call this before Pass 2 so the RAG search has domain vocabulary to work with.
    Returns the number of chunks indexed.
    """
    doc_files: list[Path] = []
    if docs_path.is_file():
        doc_files = [docs_path]
    else:
        for ext in (".md", ".rst", ".txt"):
            doc_files.extend(sorted(docs_path.rglob(f"*{ext}")))

    print(f"\n[Bootstrap] Indexing {len(doc_files)} doc file(s) into R2R...")
    client = _r2r_client()
    count = 0
    for doc_file in doc_files:
        try:
            content = doc_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [warn] {doc_file.name}: {e}")
            continue
        for chunk in _chunk_doc(content):
            try:
                client.documents.create(
                    raw_text=chunk,
                    metadata={
                        "source_file": doc_file.name,
                        "module":      "documentation",
                        "chunk_type":  "doc_summary",
                    },
                )
                count += 1
            except Exception as e:
                print(f"  [warn] {doc_file.name} chunk: {e}")
    print(f"[Bootstrap] Indexed {count} doc chunk(s)")
    return count


def search_kb(query: str, limit: int = 3) -> str:
    """
    Query R2R and return a formatted context string for prompt injection.
    Returns empty string if R2R is unavailable (degrades gracefully).
    """
    try:
        client = _r2r_client()
        results = client.retrieval.search(query=query, search_settings={"limit": limit})
        hits = results.results.chunk_search_results
        if not hits:
            return ""
        parts = []
        for hit in hits:
            src = (hit.metadata or {}).get("source_file", "?")
            parts.append(f"[{src}] {(hit.text or '').strip()[:300]}")
        return "\n".join(parts)
    except Exception:
        return ""


def index_summaries_to_r2r(summaries: list[dict]) -> None:
    """
    Index all summaries into R2R.  If an entry already has a 'doc_id' (from a
    previous pass), the old document is deleted first so there are no duplicates.
    The new doc_id is stored back into the entry dict.

    Also indexes raw code as a separate chunk (chunk_type: "raw_code") so that
    searches for exact identifiers can match the source code directly.
    """
    client = _r2r_client()
    print(f"\n[Index] Indexing {len(summaries)} summaries + raw code into R2R...")
    for i, entry in enumerate(summaries):
        # Delete old summary doc if re-indexing
        if entry.get("doc_id"):
            try:
                client.documents.delete(entry["doc_id"])
            except Exception:
                pass
        # Delete old raw_code doc if re-indexing
        if entry.get("code_doc_id"):
            try:
                client.documents.delete(entry["code_doc_id"])
            except Exception:
                pass

        src_file = entry.get("source_file", "")
        module   = entry.get("module", "")

        # Index the summary (embedded for semantic search)
        try:
            resp = client.documents.create(
                raw_text=entry["summary"],
                metadata={
                    "source_file": src_file,
                    "module":      module,
                    "chunk_type":  entry.get("chunk_type", "function_summary"),
                },
            )
            entry["doc_id"] = str(resp.results.document_id)
        except Exception as e:
            print(f"  [warn] {src_file}: summary index failed: {e}")

        # Index the raw code as a separate chunk (for identifier/keyword search)
        raw_code = entry.get("raw_code", "").strip()
        if raw_code and len(raw_code) > 30:
            try:
                resp = client.documents.create(
                    raw_text=raw_code,
                    metadata={
                        "source_file": src_file,
                        "module":      module,
                        "chunk_type":  "raw_code",
                    },
                )
                entry["code_doc_id"] = str(resp.results.document_id)
            except Exception as e:
                print(f"  [warn] {src_file}: code index failed: {e}")

        if i > 0 and i % 20 == 0:
            time.sleep(0.5)
    print("[Index] Done")


# ── Pass 3: Review ─────────────────────────────────────────────────────────────

REVIEW_SYSTEM = (
    "You are reviewing knowledge-base summaries for accuracy and quality. "
    "Output ONLY valid JSON — no markdown, no text outside the JSON object."
)


def _build_review_prompt(entry: dict, kb_context: str) -> str:
    return f"""Review this knowledge-base summary.

Source: {entry.get('source_file', '?')}  Module: {entry.get('module', '?')}

Summary:
{entry['summary']}

Original source code:
```
{entry.get('raw_code', '')[:1500]}
```

Similar entries already in the knowledge base (for vocabulary reference):
{kb_context or '(none found)'}

Criteria — a good summary must be:
1. Accurate: matches what the code actually does
2. Domain-specific: uses the project's own vocabulary, not generic terms
3. Useful: answers a real question a developer would ask

Output ONLY one of these two JSON forms:
  {{"keep": true}}
  {{"keep": false, "improved": "rewritten summary here"}}"""


def run_review(model: str, summaries: list[dict]) -> tuple[list[dict], float]:
    """
    Review every summary for quality. Queries the KB for domain context so
    the LLM can use proper vocabulary in rewrites.
    Summaries must be indexed in R2R before calling this.
    Returns (summaries_with_improvements, edit_rate).
    """
    print(f"\n[Review] Reviewing {len(summaries)} summaries with {model}...")
    edited = 0

    for i, entry in enumerate(summaries):
        query = f"{entry.get('module', '')} {entry.get('summary', '')[:120]}"
        kb_context = search_kb(query, limit=3)
        prompt = _build_review_prompt(entry, kb_context)

        try:
            raw = llm_call(model, REVIEW_SYSTEM, prompt, max_tokens=512, json_mode=True)
            data = json.loads(raw)
            if not data.get("keep") and data.get("improved", "").strip():
                entry["summary"] = data["improved"].strip()
                edited += 1
                print(f"  [{i+1:>4}/{len(summaries)}] edited  {entry.get('source_file', '?')}")
            else:
                print(f"  [{i+1:>4}/{len(summaries)}] kept    {entry.get('source_file', '?')}")
        except Exception as e:
            print(f"  [{i+1:>4}/{len(summaries)}] [warn] review failed: {e}")

        time.sleep(0.2)

    edit_rate = edited / len(summaries) if summaries else 0.0
    print(f"[Review] {edited}/{len(summaries)} summaries edited ({edit_rate:.1%})")
    return summaries, edit_rate


# ── Pass 1: Module Discovery ───────────────────────────────────────────────────

PASS1_SYSTEM = (
    "You are a senior software architect analyzing a codebase to build a semantic knowledge base. "
    "Output ONLY valid JSON — no markdown fences, no commentary, no text outside the JSON object."
)

def build_pass1_prompt(codebase_name: str, language: str,
                       tree: str, file_samples: dict[str, str],
                       docs_context: Optional[str] = None) -> str:
    samples_section = ""
    for fname, sample in list(file_samples.items())[:12]:
        samples_section += f"\n### {fname}\n```{language}\n{sample}\n```\n"

    docs_section = ""
    if docs_context:
        docs_section = f"\n## Design / Reference Documentation\n{docs_context}\n"

    return f"""Analyze this {language} codebase and produce a module map for a domain knowledge base.

## Directory Structure
```
{tree}
```
{docs_section}
## File Samples
{samples_section}

## Task
1. Group related files into logical modules/subsystems (2–6 modules total).
2. For each module, write 3–6 domain-specific questions a developer would ask.
   - Use the project's own vocabulary (class names, concepts, protocols).
   - Focus on HOW things work internally, not just WHAT they are.
   - Example for an HTTP library:
     "How does Session manage connection pools across requests?"
     "What retry/redirect logic does HTTPAdapter implement?"

Output ONLY this JSON (no markdown, no extra text):
{{
  "project": "{codebase_name}",
  "description": "one sentence describing what this codebase does",
  "modules": [
    {{
      "name": "short_module_name",
      "description": "what this module/subsystem does",
      "files": ["filename.py"],
      "questions": ["domain question 1", "question 2"]
    }}
  ]
}}"""


def run_pass1(model: str, codebase: Path, files: list[Path],
              language: str, docs_context: Optional[str] = None) -> ModuleMap:
    """Call the LLM to analyze directory structure and produce module_map."""

    print("\n[Pass 1] Building directory tree and reading file samples...")
    tree = build_directory_tree(codebase, files)

    file_samples: dict[str, str] = {}
    for f in files:
        sample = read_file_sample(f)
        if len(sample.strip()) > 20:
            file_samples[str(f.relative_to(codebase))] = sample

    prompt = build_pass1_prompt(
        codebase_name=codebase.name,
        language=language,
        tree=tree,
        file_samples=file_samples,
        docs_context=docs_context,
    )

    print(f"[Pass 1] Calling {model} for module discovery ({len(prompt)} char prompt)...")
    raw = llm_call(model, PASS1_SYSTEM, prompt, max_tokens=4096, json_mode=True)

    # Parse and validate with Pydantic
    try:
        data = json.loads(raw)
        module_map = ModuleMap(**data)
    except Exception as e:
        print(f"[Pass 1] ERROR: could not parse LLM output as ModuleMap: {e}")
        print("Raw output:", raw[:500])
        sys.exit(1)

    print(f"[Pass 1] Discovered {len(module_map.modules)} modules:")
    for m in module_map.modules:
        print(f"  - {m.name}: {m.description} ({len(m.files)} files, {len(m.questions)} questions)")

    return module_map


# ── Pass 2: Summarization ──────────────────────────────────────────────────────

PASS2_SYSTEM = (
    "You are building a semantic knowledge base for a software codebase. "
    "Generate concise, domain-aware summaries suitable for embedding and semantic search. "
    "Write plain prose — no markdown, no bullet points."
)

def build_pass2_prompt(project_desc: str, module_name: str,
                       module_desc: str, questions: list[str],
                       source_file: str, raw_code: str,
                       kb_context: str = "") -> str:
    questions_text = "\n".join(f"- {q}" for q in questions)
    kb_section = (
        f"\nRelevant context from the knowledge base (use this vocabulary):\n{kb_context}\n"
        if kb_context else ""
    )
    return f"""Project: {project_desc}
Module: {module_name} — {module_desc}

Domain questions this module answers:
{questions_text}
{kb_section}
Source file: {source_file}
Code chunk:
```
{raw_code}
```

Write a 2–4 sentence domain-aware summary of this code chunk.
Requirements:
- Use the project's own vocabulary (class names, function names, domain concepts)
- State what this code DOES, not just what it IS
- Mention which of the domain questions above this chunk addresses (if any)
- If trivial (imports-only, constants, empty stub), one sentence is enough
- Plain prose only — no markdown, no bullets

Summary:"""


def _make_chunk_key(source_file: str, chunk_index: int) -> str:
    """Unique key for a chunk, used to detect already-summarized chunks on resume."""
    return f"{source_file}::{chunk_index}"


def run_pass2(model: str, codebase: Path, module_map: ModuleMap,
              language: str, max_chunks: Optional[int] = None,
              rag: bool = False,
              summaries_path: Optional[Path] = None) -> list[dict]:
    """Chunk each file and call the LLM to generate a domain-aware summary per chunk.
    With rag=True, queries the KB before each chunk to inject relevant context.
    If summaries_path is set, writes incrementally for crash-safety."""

    # Load existing summaries for resume support
    summaries: list[dict] = []
    done_keys: set[str] = set()
    if summaries_path and summaries_path.exists():
        try:
            summaries = json.loads(summaries_path.read_text())
            for entry in summaries:
                done_keys.add(_make_chunk_key(
                    entry.get("source_file", ""),
                    entry.get("chunk_index", 0),
                ))
            if done_keys:
                print(f"\n[Pass 2] Resuming — {len(done_keys)} chunks already summarized")
        except Exception:
            summaries = []

    total_chunks = len(summaries)
    skipped = 0
    new_this_run = 0

    rag_label = " (RAG-augmented)" if rag else ""
    print(f"\n[Pass 2] Summarizing {len(module_map.modules)} modules with {model}{rag_label}...")

    for mod in module_map.modules:
        print(f"\n  Module: {mod.name} ({len(mod.files)} files)")

        for fname in mod.files:
            # Try direct relative path first (fast, deterministic)
            fpath = codebase / fname
            if not fpath.exists():
                # Fall back to rglob on the basename (handles LLM returning just filenames)
                candidates = list(codebase.rglob(Path(fname).name))
                if not candidates:
                    print(f"    [warn] {fname} not found under {codebase}, skipping")
                    skipped += 1
                    continue
                fpath = candidates[0]
                if len(candidates) > 1:
                    print(f"    [warn] {fname}: {len(candidates)} matches, using {fpath.relative_to(codebase)}")

            chunks = chunk_file(fpath, language=language)
            if not chunks:
                print(f"    [skip] {fname} — empty")
                skipped += 1
                continue

            rel_path = str(fpath.relative_to(codebase))
            print(f"    {fname}: {len(chunks)} chunk(s)")

            for i, chunk in enumerate(chunks):
                if max_chunks is not None and total_chunks >= max_chunks:
                    print(f"\n[Pass 2] Reached --max-chunks={max_chunks}, stopping.")
                    return summaries

                chunk_key = _make_chunk_key(rel_path, i)
                if chunk_key in done_keys:
                    total_chunks += 0  # already counted
                    print(f"      chunk {i+1}/{len(chunks)} → [cached]")
                    continue

                kb_context = search_kb(f"{mod.name} {chunk[:200]}", limit=3) if rag else ""
                prompt = build_pass2_prompt(
                    project_desc=f"{module_map.project}: {module_map.description}",
                    module_name=mod.name,
                    module_desc=mod.description,
                    questions=mod.questions,
                    source_file=rel_path,
                    raw_code=chunk,
                    kb_context=kb_context,
                )

                try:
                    # Stream + collect
                    summary_text = ""
                    for text in llm_call(model, PASS2_SYSTEM, prompt, max_tokens=512, stream=True):
                        summary_text += text

                    summary_text = summary_text.strip()
                    if summary_text.lower().startswith("summary:"):
                        summary_text = summary_text[len("summary:"):].strip()

                    entry = {
                        "summary":     summary_text,
                        "raw_code":    chunk,
                        "source_file": rel_path,
                        "module":      mod.name,
                        "chunk_type":  "function_summary",
                        "chunk_index": i,
                    }
                    summaries.append(entry)
                    total_chunks += 1
                    new_this_run += 1
                    print(f"      chunk {i+1}/{len(chunks)} → {len(summary_text)} chars")

                    # Write incrementally for crash safety
                    if summaries_path and new_this_run % 5 == 0:
                        summaries_path.write_text(json.dumps(summaries, indent=2))

                    time.sleep(0.3)  # avoid rate-limit bursts

                except Exception as e:
                    if "rate" in str(e).lower():
                        print(f"      [rate limit] chunk {i+1} — waiting 30s...")
                        time.sleep(30)
                        try:
                            summary_text = ""
                            for text in llm_call(model, PASS2_SYSTEM, prompt, max_tokens=512, stream=True):
                                summary_text += text
                            summaries.append({
                                "summary":     summary_text.strip(),
                                "raw_code":    chunk,
                                "source_file": rel_path,
                                "module":      mod.name,
                                "chunk_type":  "function_summary",
                                "chunk_index": i,
                            })
                            total_chunks += 1
                            new_this_run += 1
                        except Exception as retry_err:
                            print(f"      [error] retry failed: {retry_err}, skipping")
                            skipped += 1
                    else:
                        print(f"      [error] chunk {i+1}: {e}, skipping")
                        skipped += 1

    # Final write
    if summaries_path:
        summaries_path.write_text(json.dumps(summaries, indent=2))

    print(f"\n[Pass 2] Done: {total_chunks} summaries total "
          f"({new_this_run} new this run, {skipped} skipped)")
    return summaries


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Study agent: multi-pass codebase analysis → module_map.json + summaries.json"
    )
    parser.add_argument("--codebase", required=True,
                        help="Root directory of the codebase to analyze")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=(
                            "litellm model string (default: %(default)s). "
                            "Examples: openai/gpt-4o, ollama/llama3.1, "
                            "anthropic/claude-opus-4-6, groq/llama3-70b-8192"
                        ))
    parser.add_argument("--output-dir", default=None,
                        help="Where to write module_map.json and summaries.json "
                             "(default: same dir as this script)")
    parser.add_argument("--language", default="python",
                        choices=["python", "javascript", "typescript", "cpp", "java", "go", "rust"],
                        help="Primary language of the codebase (default: python)")
    parser.add_argument("--docs", default=None,
                        help="Path to a docs file/directory. Used in Pass 1 context. "
                             "Also indexed into R2R when --bootstrap-docs is set.")
    parser.add_argument("--bootstrap-docs", action="store_true",
                        help="Index --docs into R2R before Pass 2 so RAG has domain vocabulary")
    parser.add_argument("--rag", action="store_true",
                        help="Query the KB before each chunk in Pass 2 for richer context. "
                             "Most useful after --bootstrap-docs.")
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of summarize+review iterations (default: 1 = no review). "
                             "With --passes 3: runs Pass 2, indexes, reviews, repeats up to 3x "
                             "or until edit rate drops below 5%%.")
    parser.add_argument("--pass1-only", action="store_true",
                        help="Only run Pass 1 (module discovery)")
    parser.add_argument("--pass2-only", action="store_true",
                        help="Only run Pass 2 (requires existing module_map.json)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Cap total chunks in Pass 2 (good for quick demos)")
    parser.add_argument("--incremental", action="store_true",
                        help="Only re-summarize files whose content has changed since "
                             "the last run. Uses a hash manifest (file_hashes.json).")
    args = parser.parse_args()

    codebase = Path(args.codebase).resolve()
    if not codebase.is_dir():
        print(f"Error: --codebase '{codebase}' is not a directory")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    module_map_path = output_dir / "module_map.json"
    summaries_path  = output_dir / "summaries.json"

    # ── Collect source files ────────────────────────────────────────────────────
    files = collect_source_files(codebase, language=args.language)
    print(f"Found {len(files)} {args.language} source files under {codebase}")
    if not files:
        print("No source files found. Check --codebase and --language.")
        sys.exit(1)

    # ── Incremental: filter to changed files only ──────────────────────────────
    hash_manifest_path = output_dir / "file_hashes.json"
    new_manifest: Optional[dict] = None
    if args.incremental:
        changed_files, new_manifest = filter_changed_files(codebase, files, hash_manifest_path)
        if not changed_files:
            print("[Incremental] No files changed since last run. Nothing to do.")
            return
        print(f"[Incremental] {len(changed_files)}/{len(files)} files changed — "
              f"only these will be re-summarized")
        # Remove old summaries for changed files so they get regenerated
        if summaries_path.exists():
            try:
                existing = json.loads(summaries_path.read_text())
                changed_rels = {str(f.relative_to(codebase)) for f in changed_files}
                kept = [e for e in existing if e.get("source_file") not in changed_rels]
                summaries_path.write_text(json.dumps(kept, indent=2))
                print(f"[Incremental] Purged {len(existing) - len(kept)} stale summaries, "
                      f"kept {len(kept)}")
            except Exception:
                pass

    # ── Bootstrap docs into R2R (before Pass 1, so Pass 1 benefits too) ─────────
    # Auto-detect: if --rag is set and no --docs given, look for common doc dirs
    if args.rag and not args.docs:
        for candidate in ("docs", "doc", "documentation", "design"):
            candidate_path = codebase / candidate
            if candidate_path.is_dir() and any(candidate_path.rglob("*.md")):
                args.docs = str(candidate_path)
                args.bootstrap_docs = True
                print(f"[Auto] Found docs at {candidate_path}, will bootstrap")
                break

    if args.bootstrap_docs:
        if not args.docs:
            print("Error: --bootstrap-docs requires --docs PATH")
            sys.exit(1)
        bootstrap_docs(Path(args.docs))

    # ── Pass 1 ──────────────────────────────────────────────────────────────────
    if not args.pass2_only:
        docs_context = None
        if args.docs:
            docs_path = Path(args.docs)
            if docs_path.is_file():
                docs_context = docs_path.read_text(encoding="utf-8", errors="replace")[:8000]
            elif docs_path.is_dir():
                parts = []
                for doc_file in sorted(docs_path.rglob("*")):
                    if doc_file.suffix in {".md", ".rst", ".txt"} and doc_file.is_file():
                        parts.append(f"### {doc_file.name}\n{doc_file.read_text(errors='replace')[:2000]}")
                        if sum(len(p) for p in parts) > 8000:
                            break
                docs_context = "\n\n".join(parts)

        module_map = run_pass1(args.model, codebase, files, args.language, docs_context)
        module_map_path.write_text(json.dumps(module_map.model_dump(), indent=2))
        print(f"\n[Pass 1] Written to {module_map_path}")
    else:
        if not module_map_path.exists():
            print(f"Error: {module_map_path} not found. Run Pass 1 first.")
            sys.exit(1)
        module_map = ModuleMap(**json.loads(module_map_path.read_text()))
        print(f"[Pass 2] Loaded {len(module_map.modules)} modules from {module_map_path}")

    if args.pass1_only:
        print("\nPass 1 complete. Run with --pass2-only to generate summaries.")
        return

    # ── Iterative Pass 2 + Review ────────────────────────────────────────────────
    #
    # Pass 1 (pass_num=1): summarize all chunks → index → review → improve
    # Pass 2+ (pass_num>1): only review+improve (KB is richer from pass 1,
    #   so the reviewer has better context). Re-running Pass 2 from scratch
    #   would discard the review edits.
    summaries: list[dict] = []
    for pass_num in range(1, args.passes + 1):
        if args.passes > 1:
            print(f"\n{'='*60}")
            print(f"  PASS {pass_num} of {args.passes}")
            print(f"{'='*60}")

        # Only run full summarization on the first pass
        if pass_num == 1:
            summaries = run_pass2(
                args.model, codebase, module_map,
                language=args.language,
                max_chunks=args.max_chunks,
                rag=args.rag,
                summaries_path=summaries_path,
            )

        if args.passes > 1:
            # Index summaries so the review pass can query them
            index_summaries_to_r2r(summaries)

            # Review + improve
            summaries, edit_rate = run_review(args.model, summaries)

            # Update R2R with improved summaries
            index_summaries_to_r2r(summaries)

            summaries_path.write_text(json.dumps(summaries, indent=2))
            print(f"\n[Pass {pass_num}] {len(summaries)} summaries written to {summaries_path}")

            if edit_rate < CONVERGENCE_THRESHOLD:
                print(f"\nConverged after pass {pass_num} "
                      f"(edit rate {edit_rate:.1%} < {CONVERGENCE_THRESHOLD:.0%})")
                break
        else:
            summaries_path.write_text(json.dumps(summaries, indent=2))

    # Save hash manifest so --incremental can detect changes next time
    if new_manifest is not None:
        save_hash_manifest(hash_manifest_path, new_manifest)
        print(f"[Incremental] Saved file hashes to {hash_manifest_path}")

    if args.passes == 1:
        print(f"\n[Done] {len(summaries)} summaries written to {summaries_path}")
        print(f"       Next: python indexer.py --index {summaries_path}")
    else:
        print(f"\n[Done] Final summaries in {summaries_path} (already indexed in R2R)")


if __name__ == "__main__":
    main()
