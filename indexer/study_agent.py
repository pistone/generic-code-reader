"""
Study Agent — Two-pass codebase analysis for the domain knowledge base.

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
  - Output: summaries.json  →  feed to indexer.py

Usage:
  # OpenAI (get a key at platform.openai.com)
  OPENAI_API_KEY=sk-... python study_agent.py --codebase /tmp/requests_src/src/requests

  # Ollama — 100% local, no key needed
  # First: brew install ollama && ollama pull llama3.1
  python study_agent.py --codebase /tmp/requests_src/src/requests --model ollama/llama3.1

  # Quick demo with 20 chunks
  python study_agent.py --codebase /tmp/requests_src/src/requests --max-chunks 20

  # Skip Pass 1 if module_map.json already exists
  python study_agent.py --codebase /tmp/requests_src/src/requests --pass2-only
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "openai/gpt-4o"

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
        return result or [""]


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
            file_samples[f.name] = sample

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
                       source_file: str, raw_code: str) -> str:
    questions_text = "\n".join(f"- {q}" for q in questions)
    return f"""Project: {project_desc}
Module: {module_name} — {module_desc}

Domain questions this module answers:
{questions_text}

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


def run_pass2(model: str, codebase: Path, module_map: ModuleMap,
              language: str, max_chunks: Optional[int] = None) -> list[dict]:
    """Chunk each file and call the LLM to generate a domain-aware summary per chunk."""

    summaries = []
    total_chunks = 0
    skipped = 0

    print(f"\n[Pass 2] Summarizing {len(module_map.modules)} modules with {model}...")

    for mod in module_map.modules:
        print(f"\n  Module: {mod.name} ({len(mod.files)} files)")

        for fname in mod.files:
            candidates = list(codebase.rglob(fname))
            if not candidates:
                print(f"    [warn] {fname} not found under {codebase}, skipping")
                skipped += 1
                continue
            fpath = candidates[0]

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

                prompt = build_pass2_prompt(
                    project_desc=f"{module_map.project}: {module_map.description}",
                    module_name=mod.name,
                    module_desc=mod.description,
                    questions=mod.questions,
                    source_file=rel_path,
                    raw_code=chunk,
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
                    }
                    summaries.append(entry)
                    total_chunks += 1
                    print(f"      chunk {i+1}/{len(chunks)} → {len(summary_text)} chars")

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
                            })
                            total_chunks += 1
                        except Exception as retry_err:
                            print(f"      [error] retry failed: {retry_err}, skipping")
                            skipped += 1
                    else:
                        print(f"      [error] chunk {i+1}: {e}, skipping")
                        skipped += 1

    print(f"\n[Pass 2] Done: {total_chunks} summaries generated, {skipped} skipped")
    return summaries


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Study agent: two-pass codebase analysis → module_map.json + summaries.json"
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
                        help="Path to a docs file/directory to include in Pass 1 context")
    parser.add_argument("--pass1-only", action="store_true",
                        help="Only run Pass 1 (module discovery)")
    parser.add_argument("--pass2-only", action="store_true",
                        help="Only run Pass 2 (requires existing module_map.json)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Cap total chunks in Pass 2 (good for quick demos)")
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

    # ── Pass 2 ──────────────────────────────────────────────────────────────────
    summaries = run_pass2(
        args.model, codebase, module_map,
        language=args.language, max_chunks=args.max_chunks,
    )

    summaries_path.write_text(json.dumps(summaries, indent=2))
    print(f"\n[Done] {len(summaries)} summaries written to {summaries_path}")
    print(f"       Next: python indexer.py --index {summaries_path}")


if __name__ == "__main__":
    main()
