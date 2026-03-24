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
import sys
import time
from pathlib import Path
from typing import Optional

from datetime import datetime, timezone
from pydantic import BaseModel

# Shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codebase_shared.utils import TokenTracker, llm_call, llm_tool_loop  # noqa: E402

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
                ".mypy_cache", ".pytest_cache", ".tox",
                # Generated / third-party code
                "third_party", "thirdparty", "3rdparty", "vendor",
                "external", "deps",
                # Build artifacts common in C++ projects
                "cmake-build-debug", "cmake-build-release", "out",
                ".build", "_build"}
SKIP_TEST_DIRS = {"test", "tests", "testing", "test_data", "testdata",
                  "testcases", "test_fixtures", "fixtures"}

# TokenTracker and llm_call imported from codebase_shared.utils

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

# llm_call imported from codebase_shared.utils

# ── File utilities ─────────────────────────────────────────────────────────────

def collect_source_files(codebase: Path, language: str = "python",
                         include_tests: bool = False) -> list[Path]:
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

    # Filename patterns that indicate test files (even outside test dirs)
    test_prefixes = ("test_", "test-")
    test_suffixes = ("_test.", "-test.", "_spec.", ".spec.", "_unittest.", "_mock.")

    results = []
    for p in sorted(codebase.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not include_tests and any(part in SKIP_TEST_DIRS for part in p.parts):
            continue
        if p.suffix in SKIP_SUFFIXES:
            continue
        if p.suffix not in exts:
            continue
        # Skip test files by name pattern (unless --include-tests)
        if not include_tests:
            name_lower = p.name.lower()
            if name_lower.startswith(test_prefixes):
                continue
            if any(s in name_lower for s in test_suffixes):
                continue
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


def build_directory_tree(codebase: Path, files: list[Path],
                         max_depth: int = 3,
                         expand_dirs: Optional[set[str]] = None) -> str:
    """Build a depth-limited annotated directory tree from file list.

    Shows directories up to max_depth. Unexpanded nodes get annotations like
    "(45 files, 3 subdirs)". Leaf directories list file counts.
    Use expand_dirs to selectively expand specific directories deeper.

    For small codebases (≤200 files), shows every file (original behaviour).
    """
    if len(files) <= 200:
        lines = [str(codebase.name) + "/"]
        for rp in sorted(f.relative_to(codebase) for f in files):
            indent = "  " * (len(rp.parts) - 1)
            lines.append(f"{indent}  {rp.name}")
        return "\n".join(lines)

    expand_dirs = expand_dirs or set()

    # Build a tree structure: {dir_path: {files: int, subdirs: set}}
    from collections import defaultdict
    dir_files: dict[str, int] = defaultdict(int)     # files directly in this dir
    all_dirs: set[str] = set()

    for f in files:
        rel = f.relative_to(codebase)
        parts = rel.parts[:-1]  # directory components
        # Count file in its immediate directory
        dir_path = "/".join(parts) if parts else "."
        dir_files[dir_path] += 1
        # Register all ancestor directories
        for i in range(1, len(parts) + 1):
            all_dirs.add("/".join(parts[:i]))

    def _effective_depth(dir_path: str) -> int:
        """How deep to expand this directory."""
        depth = max_depth
        # If this dir or an ancestor is in expand_dirs, allow deeper
        for ed in expand_dirs:
            if dir_path.startswith(ed) or ed.startswith(dir_path):
                depth = max_depth + 2
                break
        return depth

    def _count_below(prefix: str) -> tuple[int, int]:
        """Count total files and immediate subdirs below a directory prefix."""
        total_files = 0
        immediate_subdirs = set()
        for dp, fc in dir_files.items():
            if dp == prefix or dp.startswith(prefix + "/"):
                total_files += fc
            if dp.startswith(prefix + "/"):
                # Get the immediate child directory name
                rest = dp[len(prefix) + 1:]
                immediate_subdirs.add(rest.split("/")[0])
        # Also count subdirs registered in all_dirs
        for d in all_dirs:
            if d.startswith(prefix + "/"):
                rest = d[len(prefix) + 1:]
                if "/" not in rest:
                    immediate_subdirs.add(rest)
        return total_files, len(immediate_subdirs)

    def _render_dir(dir_path: str, depth: int) -> list[str]:
        """Recursively render a directory node."""
        name = dir_path.split("/")[-1] if "/" in dir_path else dir_path
        indent = "  " * depth
        eff_depth = _effective_depth(dir_path)

        # Find immediate children (subdirs at this level)
        child_dirs = sorted(
            d for d in all_dirs
            if d.startswith(dir_path + "/") and d.count("/") == dir_path.count("/") + 1
        )

        # Files directly in this directory
        direct_files = dir_files.get(dir_path, 0)
        total_files, n_subdirs = _count_below(dir_path)

        if depth >= eff_depth and (child_dirs or total_files > direct_files):
            # Truncated: show summary annotation
            parts = [f"{indent}{name}/"]
            annotations = []
            if total_files:
                annotations.append(f"{total_files} files")
            if n_subdirs:
                annotations.append(f"{n_subdirs} subdirs")
            if annotations:
                parts[0] += f"  ({', '.join(annotations)})"
            return parts

        # Expanded: recurse into children
        lines = [f"{indent}{name}/"]
        if direct_files:
            lines.append(f"{indent}  ({direct_files} files here)")
        for child in child_dirs:
            lines.extend(_render_dir(child, depth + 1))
        return lines

    # Build from top-level directories
    result = [f"{codebase.name}/  ({len(files)} files total)"]

    # Root files
    root_files = dir_files.get(".", 0)
    if root_files:
        result.append(f"  ({root_files} files in root)")

    # Top-level directories
    top_dirs = sorted(d for d in all_dirs if "/" not in d)
    for td in top_dirs:
        result.extend(_render_dir(td, 1))

    return "\n".join(result)


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
    DEPRECATED: Use doc_agent for proper document ingestion instead.
    Run: python -m doc_agent.doc_agent --docs /path/to/docs

    doc_agent provides section-aware chunking, heading extraction, HTML parsing,
    incremental mode, and proper metadata (doc_title, source_type, last_modified).
    This function remains for backward compatibility.

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


def _create_or_replace(client, raw_text: str, metadata: dict) -> str:
    """Create a document in R2R; if it already exists, delete and retry."""
    import re as _re
    try:
        resp = client.documents.create(raw_text=raw_text, metadata=metadata)
        return str(resp.results.document_id)
    except Exception as e:
        err = str(e)
        if "already exists" not in err.lower():
            raise
        match = _re.search(r"document\s+([0-9a-f-]{36})", err, _re.IGNORECASE)
        if match:
            try:
                client.documents.delete(match.group(1))
            except Exception:
                pass
            resp = client.documents.create(raw_text=raw_text, metadata=metadata)
            return str(resp.results.document_id)
        raise


def index_summaries_to_r2r(summaries: list[dict]) -> None:
    """
    Index all summaries into R2R with upsert semantics.
    If a document already exists (same content hash), it is replaced.
    The new doc_id is stored back into the entry dict.

    Also indexes raw code as a separate chunk (chunk_type: "raw_code") so that
    searches for exact identifiers can match the source code directly.
    """
    client = _r2r_client()
    print(f"\n[Index] Indexing {len(summaries)} summaries + raw code into R2R...")
    for i, entry in enumerate(summaries):
        src_file = entry.get("source_file", "")
        module   = entry.get("module", "")

        # Index the summary (embedded for semantic search)
        try:
            doc_id = _create_or_replace(client, entry["summary"], {
                "source_file": src_file,
                "module":      module,
                "chunk_type":  entry.get("chunk_type", "function_summary"),
                "source_type": "code",
            })
            entry["doc_id"] = doc_id
        except Exception as e:
            print(f"  [warn] {src_file}: summary index failed: {e}")

        # Index the raw code as a separate chunk (for identifier/keyword search)
        raw_code = entry.get("raw_code", "").strip()
        if raw_code and len(raw_code) > 30:
            try:
                code_doc_id = _create_or_replace(client, raw_code, {
                    "source_file": src_file,
                    "module":      module,
                    "chunk_type":  "raw_code",
                    "source_type": "code",
                })
                entry["code_doc_id"] = code_doc_id
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


def run_review(model: str, summaries: list[dict],
               tracker: Optional[TokenTracker] = None) -> tuple[list[dict], float]:
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
            raw = llm_call(model, REVIEW_SYSTEM, prompt, max_tokens=512, json_mode=True,
                           tracker=tracker, phase="Review")
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


# ── Pass 1: Agent-based Module Discovery ───────────────────────────────────────

PASS1_SYSTEM = (
    "You are a senior software architect exploring a codebase to build a "
    "semantic knowledge base. Use the provided tools to explore the directory "
    "structure and read key files. When you understand the architecture, call "
    "define_modules to define the module map.\n\n"
    "Guidelines:\n"
    "- Each module should map to a directory or group of related directories\n"
    "- Every source file should belong to exactly one module\n"
    "- Write 3-6 domain-specific questions per module (use the project's vocabulary)\n"
    "- Focus questions on HOW things work internally, not just WHAT they are"
)

MAX_EXPLORE_ROUNDS = 8   # cap on interactive exploration rounds
MAX_FILES_PER_READ = 8   # max files per read_files call


def _group_files_by_dir(codebase: Path, files: list[Path],
                         dir_path: str) -> list[Path]:
    """Return files that live under a specific directory path."""
    result = []
    for f in files:
        rel = str(f.relative_to(codebase))
        if dir_path == ".":
            if "/" not in rel:
                result.append(f)
        elif rel.startswith(dir_path + "/") or rel == dir_path:
            result.append(f)
    return result


# ── Tool definitions ──────────────────────────────────────────────────────

PASS1_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "expand_dirs",
            "description": (
                "Expand directories to see their subdirectory structure deeper. "
                "Use this when you see a truncated directory with a large file/subdir count "
                "and want to understand its internal organization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dirs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Directory paths to expand (e.g. ['core/analysis', 'lib'])",
                    },
                },
                "required": ["dirs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List the source files in a specific directory. "
                "Use this to see what files exist before deciding to read some."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Directory to list files in (e.g. 'core/handlers')",
                    },
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": (
                "Read the first 80 lines of specific source files. "
                "Use this to understand what a module does by examining key files. "
                f"Up to {MAX_FILES_PER_READ} files per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths relative to codebase root",
                    },
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "Search the existing knowledge base (design docs, previously indexed content). "
                "Use this to find documentation about a module or concept."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'dataflow analysis architecture')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 3)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "define_modules",
            "description": (
                "Define the final module map. Call this when you have explored enough "
                "to understand the codebase architecture. Every source file should "
                "belong to exactly one module via dir_paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project name"},
                    "description": {"type": "string", "description": "One sentence about the project"},
                    "modules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Short module name"},
                                "description": {"type": "string", "description": "What this module does"},
                                "dir_paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Directory paths belonging to this module",
                                },
                                "questions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "3-6 domain-specific questions about this module",
                                },
                            },
                            "required": ["name", "description", "dir_paths", "questions"],
                        },
                    },
                },
                "required": ["project", "description", "modules"],
            },
        },
    },
]


def _make_pass1_dispatch(codebase: Path, files: list[Path], language: str):
    """Create a closure that dispatches Pass 1 tool calls."""

    def dispatch(name: str, args: dict) -> str:
        if name == "expand_dirs":
            dirs = args.get("dirs", [])
            expanded_set = set(str(d) for d in dirs)
            tree = build_directory_tree(codebase, files, max_depth=3,
                                         expand_dirs=expanded_set)
            return tree

        elif name == "list_files":
            dir_path = args.get("dir_path", ".")
            matches = _group_files_by_dir(codebase, files, dir_path)
            lines = [str(f.relative_to(codebase)) for f in matches[:80]]
            if len(matches) > 80:
                lines.append(f"... and {len(matches) - 80} more files")
            return "\n".join(lines) if lines else f"(no source files found in {dir_path})"

        elif name == "read_files":
            paths = args.get("paths", [])
            parts = []
            for rf in paths[:MAX_FILES_PER_READ]:
                rf_path = codebase / str(rf)
                if rf_path.exists() and rf_path.is_file():
                    content = read_file_sample(rf_path, max_lines=SAMPLE_LINES)
                    parts.append(f"=== {rf} ===\n{content}")
                else:
                    # Try rglob fallback
                    candidates = list(codebase.rglob(Path(rf).name))
                    if candidates:
                        content = read_file_sample(candidates[0], max_lines=SAMPLE_LINES)
                        actual = str(candidates[0].relative_to(codebase))
                        parts.append(f"=== {actual} (matched from {rf}) ===\n{content}")
                    else:
                        parts.append(f"=== {rf} ===\n(file not found)")
            return "\n\n".join(parts) if parts else "(no files specified)"

        elif name == "search_kb":
            query = args.get("query", "")
            limit = args.get("limit", 3)
            result = search_kb(query, limit=limit)
            return result if result else "(no results found in knowledge base)"

        else:
            return f"Unknown tool: {name}"

    return dispatch


def run_pass1(model: str, codebase: Path, files: list[Path],
              language: str, docs_context: Optional[str] = None,
              tracker: Optional[TokenTracker] = None) -> ModuleMap:
    """Agent-based module discovery using tool calling.

    The LLM explores the codebase interactively using tools:
    - expand_dirs: see deeper directory structure
    - list_files: see files in a directory
    - read_files: read first 80 lines of specific files
    - search_kb: search existing knowledge base / design docs
    - define_modules: terminal tool — outputs the final module map

    Bounded to MAX_EXPLORE_ROUNDS to cap cost.
    """
    print(f"\n[Pass 1] {len(files)} source files found.")

    # Build initial tree
    tree = build_directory_tree(codebase, files, max_depth=3)
    tree_lines = len(tree.split("\n"))
    print(f"[Pass 1] Directory tree: {tree_lines} lines, {len(tree)} chars")

    # Build initial prompt
    docs_section = ""
    if docs_context:
        docs_section = f"\n\nDesign documentation (excerpts):\n{docs_context}"

    initial_prompt = (
        f"Explore this {language} codebase and define its modules.\n\n"
        f"Directory tree (depth-limited, counts on truncated nodes):\n"
        f"```\n{tree}\n```"
        f"{docs_section}\n\n"
        f"Use the tools to explore directories and read key files. "
        f"When you understand the architecture, call define_modules. "
        f"You have up to {MAX_EXPLORE_ROUNDS} exploration rounds."
    )

    # Create dispatcher
    dispatch = _make_pass1_dispatch(codebase, files, language)

    # Logging callback
    def on_round(round_num: int, tool_name: str, args: dict):
        if tool_name == "expand_dirs":
            print(f"  Round {round_num}: expand_dirs({args.get('dirs', [])})")
        elif tool_name == "list_files":
            print(f"  Round {round_num}: list_files({args.get('dir_path', '?')})")
        elif tool_name == "read_files":
            paths = args.get("paths", [])
            print(f"  Round {round_num}: read_files({len(paths)} files)")
        elif tool_name == "search_kb":
            print(f"  Round {round_num}: search_kb({args.get('query', '?')[:60]})")
        elif tool_name == "define_modules":
            n = len(args.get("modules", []))
            print(f"  Round {round_num}: define_modules({n} modules)")

    print(f"[Pass 1] Starting agent exploration with {model}...")

    conversation, terminal_args = llm_tool_loop(
        model=model,
        system=PASS1_SYSTEM,
        initial_messages=[{"role": "user", "content": initial_prompt}],
        tools=PASS1_TOOLS,
        dispatch=dispatch,
        terminal_tools={"define_modules"},
        max_rounds=MAX_EXPLORE_ROUNDS,
        max_tokens=4096,
        tracker=tracker,
        phase="Pass 1",
        on_round=on_round,
    )

    # If agent didn't call define_modules, force it
    if terminal_args is None:
        print(f"  [Pass 1] Max rounds reached — forcing module definition...")
        from litellm import completion
        conversation.append({
            "role": "user",
            "content": ("You have used all exploration rounds. "
                        "Call define_modules now to define the final module map. "
                        "Assign ALL directories to modules."),
        })
        response = completion(
            model=model, messages=conversation, max_tokens=4096,
            tools=PASS1_TOOLS,
            tool_choice={"type": "function", "function": {"name": "define_modules"}},
        )
        if tracker:
            tracker.record("Pass 1 (forced)", response)
        msg = response.choices[0].message
        if msg.tool_calls:
            try:
                terminal_args = json.loads(msg.tool_calls[0].function.arguments)
            except (json.JSONDecodeError, TypeError):
                terminal_args = None

    if terminal_args is None:
        print("  [ERROR] Agent failed to define modules — falling back to directory-based")
        terminal_args = _fallback_module_map(codebase, files, language)

    # ── Resolve dir_paths → actual file lists ─────────────────────────────
    project_desc = terminal_args.get("description", "")
    raw_modules = terminal_args.get("modules", [])
    print(f"\n[Pass 1] Resolving {len(raw_modules)} modules to file lists...")

    assigned_files: set[str] = set()
    modules: list[ModuleDefinition] = []

    for mod_raw in raw_modules:
        name = mod_raw.get("name", "unknown")
        desc = mod_raw.get("description", "")
        dir_paths = mod_raw.get("dir_paths", [])
        questions = mod_raw.get("questions", [])

        mod_files: list[str] = []
        for dp in dir_paths:
            for f in _group_files_by_dir(codebase, files, str(dp)):
                rel = str(f.relative_to(codebase))
                if rel not in assigned_files:
                    mod_files.append(rel)
                    assigned_files.add(rel)

        if not mod_files:
            continue

        if not questions:
            questions = [f"What does the {name} module do?",
                         f"How is {name} structured internally?"]

        modules.append(ModuleDefinition(
            name=name,
            description=desc,
            files=mod_files,
            questions=questions if isinstance(questions, list) else [str(questions)],
        ))

    # Catch orphans
    orphaned = [str(f.relative_to(codebase)) for f in files
                if str(f.relative_to(codebase)) not in assigned_files]
    if orphaned:
        print(f"[Pass 1] {len(orphaned)} orphaned files → 'other' module")
        modules.append(ModuleDefinition(
            name="other",
            description="Files not assigned to a specific module",
            files=orphaned,
            questions=["What do these miscellaneous files do?"],
        ))

    for m in modules:
        print(f"  - {m.name}: {len(m.files)} files, {len(m.questions)} Qs — {m.description[:80]}")

    module_map = ModuleMap(
        project=codebase.name,
        description=project_desc or f"A {language} codebase with {len(files)} files",
        modules=modules,
    )

    total_assigned = sum(len(m.files) for m in modules)
    print(f"\n[Pass 1] {len(modules)} modules, {total_assigned}/{len(files)} files assigned.")
    return module_map


def _fallback_module_map(codebase: Path, files: list[Path],
                          language: str) -> dict:
    """Emergency fallback: one module per top-level directory."""
    groups: dict[str, list[str]] = {}
    for f in files:
        rel = f.relative_to(codebase)
        bucket = rel.parts[0] if len(rel.parts) > 1 else "."
        groups.setdefault(bucket, []).append(str(rel))

    modules = []
    for dir_name, dir_files in sorted(groups.items()):
        modules.append({
            "name": dir_name,
            "description": f"Files under {dir_name}/",
            "dir_paths": [dir_name],
            "questions": [f"What does {dir_name} do?"],
        })

    return {
        "project": codebase.name,
        "description": f"A {language} codebase with {len(files)} files",
        "modules": modules,
    }


# ── Pass 2: Summarization ──────────────────────────────────────────────────────

PASS2_SYSTEM = (
    "You are building a semantic knowledge base for a software codebase. "
    "Generate concise, domain-aware summaries suitable for embedding and semantic search. "
    "Write plain prose — no markdown, no bullet points."
)

# Phrases that indicate the summary has unresolved references
_VAGUE_MARKERS = [
    "calls an external", "delegates to", "uses a helper",
    "defined elsewhere", "another component", "not shown here",
    "presumably", "likely", "unclear", "unknown function",
    "some kind of", "appears to", "seems to",
]


def _needs_reference_resolution(summary: str) -> Optional[str]:
    """Check if a summary contains vague references that could be resolved.

    Returns the vague phrase found, or None if the summary is specific enough.
    """
    lower = summary.lower()
    for marker in _VAGUE_MARKERS:
        if marker in lower:
            return marker
    return None


def _resolve_references(model: str, summary: str, raw_code: str,
                         codebase: Path, rel_path: str,
                         tracker: Optional[TokenTracker] = None) -> str:
    """Try to resolve vague references in a summary by reading referenced files.

    Extracts identifiers from the code chunk that look like cross-file references
    (includes, imports, class prefixes), reads those files, and re-generates the
    summary with extra context. Returns the improved summary, or the original
    if resolution fails or finds nothing useful.
    """
    import re

    # Extract cross-file references from the code
    refs: list[str] = []

    # C/C++ includes
    for m in re.finditer(r'#include\s*[<"]([^>"]+)[>"]', raw_code):
        refs.append(m.group(1))

    # Python/JS imports
    for m in re.finditer(r'(?:from|import)\s+([\w.]+)', raw_code):
        refs.append(m.group(1).replace(".", "/"))

    # Class::Method or Namespace::Class patterns (C++)
    for m in re.finditer(r'(\w+)::\w+', raw_code):
        refs.append(m.group(1))

    if not refs:
        return summary

    # Try to read referenced files (up to 3)
    extra_context_parts: list[str] = []
    seen: set[str] = set()
    for ref in refs[:6]:
        if ref in seen:
            continue
        seen.add(ref)

        # Try direct path
        ref_path = codebase / ref
        if not ref_path.exists():
            # Try common patterns: same directory, .h/.hpp extension
            parent = (codebase / rel_path).parent
            for candidate in [parent / ref, parent / (ref + ".h"),
                              parent / (ref + ".hpp")]:
                if candidate.exists():
                    ref_path = candidate
                    break
            else:
                # Try rglob on the basename
                basename = Path(ref).name
                candidates = list(codebase.rglob(basename))
                if not candidates:
                    candidates = list(codebase.rglob(basename + ".h"))
                if not candidates:
                    candidates = list(codebase.rglob(basename + ".hpp"))
                if candidates:
                    ref_path = candidates[0]
                else:
                    continue

        if ref_path.exists() and ref_path.is_file():
            content = read_file_sample(ref_path, max_lines=40)
            if len(content.strip()) > 20:
                extra_context_parts.append(
                    f"[{ref_path.relative_to(codebase)}]\n{content}"
                )
                if len(extra_context_parts) >= 3:
                    break

    if not extra_context_parts:
        return summary

    # Re-generate with extra context
    extra_context = "\n\n".join(extra_context_parts)
    refine_prompt = f"""The following summary of a code chunk has vague references.
Rewrite it to be more specific using the referenced source files provided below.

Original summary:
{summary}

Code chunk:
```
{raw_code[:2000]}
```

Referenced files:
{extra_context}

Write an improved 2-4 sentence summary. Be specific about what the referenced
code does. Plain prose only."""

    try:
        improved = llm_call(model, PASS2_SYSTEM, refine_prompt,
                            max_tokens=256,
                            tracker=tracker, phase="Pass 2 (refine)")
        improved = improved.strip()
        if improved and len(improved) > 20:
            return improved
    except Exception:
        pass

    return summary

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
              summaries_path: Optional[Path] = None,
              tracker: Optional[TokenTracker] = None) -> list[dict]:
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

    total_chunks = len(summaries)  # includes cached — for display only
    skipped = 0
    new_this_run = 0  # only new chunks — used for --max-chunks cap

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
                if max_chunks is not None and new_this_run >= max_chunks:
                    print(f"\n[Pass 2] Reached --max-chunks={max_chunks}, stopping.")
                    return summaries

                chunk_key = _make_chunk_key(rel_path, i)
                if chunk_key in done_keys:
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
                    for text in llm_call(model, PASS2_SYSTEM, prompt, max_tokens=512, stream=True,
                                         tracker=tracker, phase="Pass 2"):
                        summary_text += text

                    summary_text = summary_text.strip()
                    if summary_text.lower().startswith("summary:"):
                        summary_text = summary_text[len("summary:"):].strip()

                    # Resolve vague references if detected
                    vague = _needs_reference_resolution(summary_text)
                    refined = False
                    if vague:
                        improved = _resolve_references(
                            model, summary_text, chunk, codebase, rel_path,
                            tracker=tracker,
                        )
                        if improved != summary_text:
                            summary_text = improved
                            refined = True

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
                    ref_tag = " [refined]" if refined else ""
                    print(f"      chunk {i+1}/{len(chunks)} → {len(summary_text)} chars{ref_tag}")

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
                            for text in llm_call(model, PASS2_SYSTEM, prompt, max_tokens=512, stream=True,
                                                 tracker=tracker, phase="Pass 2"):
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
                            # Incremental write after retry (crash safety)
                            if summaries_path:
                                summaries_path.write_text(json.dumps(summaries, indent=2))
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
    parser.add_argument("--include-tests", action="store_true",
                        help="Include test files and test directories (skipped by default)")
    args = parser.parse_args()

    codebase = Path(args.codebase).resolve()
    if not codebase.is_dir():
        print(f"Error: --codebase '{codebase}' is not a directory")
        sys.exit(1)

    if args.passes < 1:
        args.passes = 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    module_map_path = output_dir / "module_map.json"
    summaries_path  = output_dir / "summaries.json"

    # ── Collect source files ────────────────────────────────────────────────────
    files = collect_source_files(codebase, language=args.language,
                                 include_tests=args.include_tests)
    skip_note = "" if args.include_tests else " (excluding tests — use --include-tests to change)"
    print(f"Found {len(files)} {args.language} source files under {codebase}{skip_note}")
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

    tracker = TokenTracker()

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

        module_map = run_pass1(args.model, codebase, files, args.language, docs_context,
                              tracker=tracker)
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
                tracker=tracker,
            )

        if args.passes > 1:
            # Index summaries so the review pass can query them
            index_summaries_to_r2r(summaries)

            # Review + improve
            summaries, edit_rate = run_review(args.model, summaries, tracker=tracker)

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

    # ── Token usage summary ────────────────────────────────────────────────────
    if tracker.phases:
        print(f"\n{tracker.summary()}")
        cost_log_path = output_dir / "cost_log.jsonl"
        entry = tracker.to_log_entry(model=args.model, codebase=codebase.name)
        with cost_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Tokens] Logged to {cost_log_path}")


if __name__ == "__main__":
    main()
