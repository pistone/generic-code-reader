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
import asyncio
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
from codebase_shared.utils import (  # noqa: E402
    TokenTracker, llm_call, allm_call, llm_tool_loop,
    RateLimitedExecutor, AsyncRateLimiter,
    detect_rpm_from_proxy,
)
try:
    from codebase_shared.colors import green, yellow, red, bold, dim, ok, warn, err  # noqa: E402
except ImportError:
    green = yellow = red = bold = dim = ok = warn = err = lambda x: x  # type: ignore

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
# Prefixes for test directory names — matches "testutil", "test_helpers", etc.
_TEST_DIR_PREFIXES = ("test", "mock", "fake", "stub")
# Keywords that identify a module as test-related (case-insensitive match)
_TEST_MODULE_KEYWORDS = {"test", "testing", "mock", "fake", "stub",
                         "fixture", "harness", "benchmark"}


def _is_test_path(p: Path) -> bool:
    """Check if a path is test-related by directory name or prefix."""
    for part in p.parts:
        part_lower = part.lower()
        # Exact match on known test dirs
        if part_lower in SKIP_TEST_DIRS:
            return True
        # Prefix match: testutil, test_helpers, tests_integration, etc.
        if any(part_lower.startswith(pfx) for pfx in _TEST_DIR_PREFIXES):
            # Avoid matching "testament", "testimony" — require _ or digit or
            # end-of-string after the prefix
            for pfx in _TEST_DIR_PREFIXES:
                if part_lower.startswith(pfx):
                    rest = part_lower[len(pfx):]
                    if not rest or rest[0] in ("_", "-", "s") or rest[0].isdigit():
                        return True
    return False


def _is_test_module(name: str, description: str, dir_paths: list) -> bool:
    """Detect test modules by name, description, or directory paths.

    Catches cases where the LLM didn't set is_test but the module is
    clearly test-related.
    """
    name_lower = name.lower()
    desc_lower = description.lower()
    # Name-based: "TestFramework", "UnitTests", "MockServices"
    if any(kw in name_lower for kw in _TEST_MODULE_KEYWORDS):
        return True
    # Description-based: "contains unit tests for..."
    if any(kw in desc_lower for kw in ("unit test", "integration test",
                                        "test suite", "test harness",
                                        "test fixture", "mock")):
        return True
    # All dir_paths are test-related
    if dir_paths and all(_is_test_dir_path(dp) for dp in dir_paths):
        return True
    return False


def _is_test_dir_path(dp: str) -> bool:
    """Check if a directory path looks test-related."""
    parts = dp.strip("/").split("/")
    for part in parts:
        part_lower = part.lower()
        if part_lower in SKIP_TEST_DIRS:
            return True
        for pfx in _TEST_DIR_PREFIXES:
            if part_lower.startswith(pfx):
                rest = part_lower[len(pfx):]
                if not rest or rest[0] in ("_", "-", "s") or rest[0].isdigit():
                    return True
    return False


_search_kb_failures = 0
_quota_exhausted = False  # module-level flag, set when any phase hits quota

# Regex for extracting function calls from summaries (for call graph inversion)
import re as _re_mod
_CALL_PATTERNS = [
    # "calls Foo::bar()", "invokes Foo::bar", "delegates to Foo::bar"
    _re_mod.compile(r'(?:calls?|invokes?|delegates?\s+to|forwards?\s+to)\s+'
                    r'(?:`)?(\w[\w:]*(?:::\w+)?)\s*(?:\(\))?(?:`)?', _re_mod.IGNORECASE),
    # "via Foo::bar()" or "through Foo::bar()"
    _re_mod.compile(r'(?:via|through)\s+(?:`)?(\w[\w:]*(?:::\w+)?)\s*(?:\(\))?(?:`)?',
                    _re_mod.IGNORECASE),
    # "uses Foo::bar()" — only if followed by () to avoid "uses the cache"
    _re_mod.compile(r'(?:uses)\s+(?:`)?(\w[\w:]*(?:::\w+)?)\s*\(\)(?:`)?', _re_mod.IGNORECASE),
]

# Cache for file lookups to avoid repeated rglob walks
_file_index_cache: dict[Path, dict[str, list[Path]]] = {}

def _find_file(codebase: Path, filename: str) -> Optional[Path]:
    """Find a file by name using a cached index. O(1) after first call.

    Only indexes source/config files — skips binaries, build artifacts,
    and directories in SKIP_DIRS.
    """
    if codebase not in _file_index_cache:
        index: dict[str, list[Path]] = {}
        for p in codebase.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix in SKIP_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            # Only index files with known extensions (skip binaries, images, etc.)
            if p.suffix and p.suffix not in _INDEXABLE_SUFFIXES:
                continue
            index.setdefault(p.name, []).append(p)
        _file_index_cache[codebase] = index

    matches = _file_index_cache.get(codebase, {}).get(filename, [])
    return matches[0] if matches else None

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

LANG_EXT_MAP = {
    "python": {".py"},
    "javascript": {".js", ".mjs"},
    "typescript": {".ts", ".tsx"},
    "cpp": {".cpp", ".cc", ".cxx", ".h", ".hpp"},
    "java": {".java"},
    "go": {".go"},
    "rust": {".rs"},
}

# Reverse map: extension → language
_EXT_TO_LANG: dict[str, str] = {}
for _lang, _exts in LANG_EXT_MAP.items():
    for _ext in _exts:
        _EXT_TO_LANG[_ext] = _lang

# Extensions worth indexing for _find_file (source + headers + configs)
_INDEXABLE_SUFFIXES = set()
for _exts in LANG_EXT_MAP.values():
    _INDEXABLE_SUFFIXES.update(_exts)
_INDEXABLE_SUFFIXES.update({".h", ".hpp", ".hxx", ".hh", ".inl",
                            ".json", ".yaml", ".yml", ".toml", ".cfg",
                            ".md", ".rst", ".txt"})


def detect_language(codebase: Path) -> Optional[str]:
    """Auto-detect the dominant language by counting file extensions.

    Scans the top 3 directory levels (fast) and returns the language
    with the most files, or None if no known language found.
    """
    counts: dict[str, int] = {}
    max_depth = 3

    def _walk(directory: Path, depth: int):
        if depth > max_depth:
            return
        try:
            entries = list(directory.iterdir())
        except PermissionError:
            return
        for p in entries:
            if p.is_file():
                lang = _EXT_TO_LANG.get(p.suffix)
                if lang:
                    counts[lang] = counts.get(lang, 0) + 1
            elif p.is_dir() and p.name not in SKIP_DIRS:
                _walk(p, depth + 1)

    _walk(codebase, 0)
    if not counts:
        return None
    winner = max(counts, key=counts.get)  # type: ignore[arg-type]
    total = counts[winner]
    # Report detection
    runner_up = sorted(((c, l) for l, c in counts.items() if l != winner), reverse=True)
    detail = f"{total} files"
    if runner_up:
        detail += f", also found {runner_up[0][1]}({runner_up[0][0]})"
    print(f"{green('[Auto]')} Detected language: {bold(winner)} ({detail})")
    return winner


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
        if not include_tests and _is_test_path(p):
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


# ── Class and function detection ───────────────────────────────────────────────

import re as _re

# Patterns for class/struct declarations across languages
_CLASS_PATTERNS = [
    # C++: class Foo : public Bar {  or  struct Foo {
    _re.compile(r'^\s*(?:class|struct)\s+(\w+)\s*(?:final\s*)?(?::\s*(?:public|protected|private)\s+[\w:]+\s*(?:,\s*(?:public|protected|private)\s+[\w:]+\s*)*)?\s*\{', _re.MULTILINE),
    # Python: class Foo(Bar):  or  class Foo:
    _re.compile(r'^\s*class\s+(\w+)\s*(?:\([^)]*\))?\s*:', _re.MULTILINE),
    # Java/C#: public class Foo extends Bar implements Baz {
    _re.compile(r'^\s*(?:public|private|protected)?\s*(?:abstract|final|static)?\s*class\s+(\w+)', _re.MULTILINE),
    # Rust: struct Foo { or impl Foo {
    _re.compile(r'^\s*(?:pub\s+)?(?:struct|impl|trait|enum)\s+(\w+)', _re.MULTILINE),
    # TypeScript/JavaScript: class Foo extends Bar {
    _re.compile(r'^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)', _re.MULTILINE),
]

# Pattern for C++ ClassName::method — extracts the class name
_CPP_CLASS_METHOD = _re.compile(r'(\w+)::(\w+)\s*\(')

# Patterns for function/method signatures across supported languages
_FUNC_SIG_PATTERNS = [
    # C/C++: return_type func_name(args) { or ::method(args) {
    _re.compile(r'^\s*(?:[\w:*&<>,\s]+?)\s+(\w[\w:]*)\s*\([^)]*\)\s*(?:const)?\s*\{?\s*$', _re.MULTILINE),
    # Python: def func_name(args):
    _re.compile(r'^\s*(?:async\s+)?def\s+(\w+)\s*\(', _re.MULTILINE),
    # Java/Go/Rust: func/fn/public void method_name(
    _re.compile(r'^\s*(?:pub(?:lic)?\s+)?(?:static\s+)?(?:func|fn)\s+(\w+)\s*[(<]', _re.MULTILINE),
    # JavaScript/TypeScript: function name( or async function name( or name(
    _re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(', _re.MULTILINE),
    # Class methods: public/private/protected Type method_name(
    _re.compile(r'^\s*(?:public|private|protected)\s+[\w<>\[\],\s]+\s+(\w+)\s*\(', _re.MULTILINE),
]


_KEYWORDS = frozenset({
    "if", "else", "for", "while", "do", "switch", "case", "return", "break",
    "continue", "try", "catch", "throw", "new", "delete", "class", "struct",
    "enum", "typedef", "namespace", "using", "template", "typename", "const",
    "static", "virtual", "override", "final", "auto", "register", "volatile",
    "extern", "inline", "sizeof", "true", "false", "null", "nullptr", "this",
    "self", "import", "from", "with", "as", "in", "is", "not", "and", "or",
    "var", "let", "const", "elif", "except", "finally", "raise", "yield",
    "lambda", "pass", "assert", "global", "nonlocal", "del",
})


def _extract_function_name(chunk_text: str) -> Optional[str]:
    """Extract the primary function/method name from the first few lines of a chunk.

    Looks at the first 5 lines for a function signature. Returns the function
    name or None if no clear signature is found.
    """
    # Only look at first 5 lines to find the enclosing function
    first_lines = "\n".join(chunk_text.splitlines()[:5])

    for pattern in _FUNC_SIG_PATTERNS:
        m = pattern.search(first_lines)
        if m:
            name = m.group(1)
            # Filter out language keywords that matched the pattern
            bare_name = name.split("::")[-1] if "::" in name else name
            if bare_name.lower() in _KEYWORDS:
                continue
            return name
    return None


def _group_chunks_by_function(chunks: list[str]) -> list[list[int]]:
    """Group consecutive chunk indices that belong to the same function.

    Returns a list of groups, where each group is a list of chunk indices.
    Single-chunk functions get a group of [i]. Multi-chunk functions get
    [i, i+1, i+2, ...].

    Detection heuristic: if chunk N+1 starts with indented code (no new
    function signature at indent level 0) AND the overlap region matches
    the end of chunk N, it's a continuation.
    """
    if not chunks:
        return []

    func_names = [_extract_function_name(c) for c in chunks]
    groups: list[list[int]] = []
    current_group: list[int] = [0]

    for i in range(1, len(chunks)):
        # Check if this chunk continues the same function as the previous
        is_continuation = False

        if func_names[i] and func_names[i - 1]:
            # Same function name in consecutive chunks = continuation
            if func_names[i] == func_names[i - 1]:
                is_continuation = True

        if not is_continuation:
            # Check overlap: if the first lines of this chunk appear at the
            # end of the previous chunk, CodeSplitter split mid-function
            overlap_lines = chunks[i].splitlines()[:CHUNK_LINES_OVERLAP]
            if overlap_lines:
                prev_end = "\n".join(chunks[i - 1].splitlines()[-CHUNK_LINES_OVERLAP:])
                overlap_text = "\n".join(overlap_lines)
                if overlap_text.strip() and overlap_text.strip() in prev_end:
                    # Has overlap AND first non-overlap line is indented
                    # (still inside a function body)
                    non_overlap = chunks[i].splitlines()[CHUNK_LINES_OVERLAP:]
                    if non_overlap and non_overlap[0].startswith((" ", "\t")):
                        is_continuation = True

        if is_continuation:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]

    groups.append(current_group)
    return groups


FUNC_SUMMARY_SYSTEM = (
    "You are a code analyst. Write a single sentence describing what this "
    "function/method does. Use specific names and domain concepts."
)

# Feature 3: Pre-pass context system — generates both a summary and
# per-chunk targeted questions for multi-chunk functions.
FUNC_PREPASS_SYSTEM = (
    "You are a senior code analyst performing a pre-scan of a function. "
    "Output ONLY valid JSON — no markdown, no text outside the JSON."
)


async def _generate_function_summaries(
    model: str, chunks: list[str], groups: list[list[int]],
    rel_path: str, file_summary: str,
    limiter: "AsyncRateLimiter",
    tracker: Optional[TokenTracker] = None,
) -> dict[int, str]:
    """Generate function context for multi-chunk functions (Feature 3: pre-pass).

    For each multi-chunk function, scans the function signature + structure
    to generate:
    - A 1-sentence summary (applied to continuation chunks)
    - Per-chunk targeted questions (injected into chunk summarization prompts)

    Returns a dict mapping chunk_index → context string.
    The first chunk gets context too (with questions), not just continuations.
    """
    # Identify multi-chunk groups
    multi_groups = [g for g in groups if len(g) > 1]
    if not multi_groups:
        return {}

    result: dict[int, str] = {}

    async def _prepass_group(group: list[int]):
        """Pre-scan a multi-chunk function: generate summary + per-chunk questions."""
        first_chunk = chunks[group[0]]
        func_name = _extract_function_name(first_chunk) or "unknown"

        # Build a structural overview of each chunk (cheap, no LLM)
        chunk_overviews = []
        for i, idx in enumerate(group):
            chunk_text = chunks[idx] if idx < len(chunks) else ""
            lines = chunk_text.splitlines()
            n_lines = len(lines)
            # Extract key structural features
            has_if = sum(1 for l in lines if _re.search(r'\bif\s*\(', l))
            has_loop = sum(1 for l in lines if _re.search(r'\b(?:for|while)\s*\(', l))
            has_try = sum(1 for l in lines if _re.search(r'\b(?:try|catch)\b', l))
            has_return = sum(1 for l in lines if _re.search(r'\breturn\b', l))
            # First 2 non-empty lines as preview
            preview_lines = [l.strip() for l in lines if l.strip()][:2]
            preview = "; ".join(preview_lines)[:120]

            chunk_overviews.append(
                f"Chunk {i+1}/{len(group)} (idx={idx}, {n_lines} lines): "
                f"{has_if} branches, {has_loop} loops, {has_try} try/catch, "
                f"{has_return} returns. Preview: {preview}"
            )

        prompt = f"""File: {rel_path}
{f'File purpose: {file_summary}' if file_summary else ''}
Function: {func_name}
Chunks: {len(group)}

Function signature and opening:
```
{first_chunk[:2000]}
```

Chunk structure:
{chr(10).join(chunk_overviews)}

Analyze this function and output JSON:
{{
  "summary": "1 sentence: what this function does",
  "params": ["param_name: brief role"],
  "chunk_questions": [
    ["question to answer when reading chunk 1"],
    ["question to answer when reading chunk 2"],
    ...
  ]
}}

For chunk_questions, generate 1-2 targeted questions per chunk based on the
structural hints above. Examples of good questions:
- "What validation rules are checked before processing?"
- "Under what conditions does the retry loop give up?"
- "What state is mutated in the error handling path?"
- "What cleanup happens when the function exits early?"

These questions will guide a per-chunk summarizer to focus on what matters."""

        try:
            raw = await allm_call(model, FUNC_PREPASS_SYSTEM, prompt,
                                  max_tokens=400, tracker=tracker,
                                  phase="Func pre-pass")
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            # Fallback: basic summary, no questions
            for idx in group[1:]:
                result[idx] = f"[Continuation of {func_name}()]"
            return

        summary = data.get("summary", f"{func_name}: unknown purpose")
        params = data.get("params", [])
        chunk_questions = data.get("chunk_questions", [])

        params_str = "; ".join(params) if params else ""
        base_context = f"[Function: {func_name}] {summary}"
        if params_str:
            base_context += f"\nParameters: {params_str}"

        # Apply context to ALL chunks in the group (including the first)
        for i, idx in enumerate(group):
            parts = [base_context]

            if i > 0:
                parts[0] = f"[Continuation of {func_name}()] {summary}"
                if params_str:
                    parts.append(f"Parameters: {params_str}")

            # Add chunk-specific questions
            if i < len(chunk_questions) and chunk_questions[i]:
                questions = chunk_questions[i]
                if isinstance(questions, list):
                    q_text = "; ".join(questions)
                else:
                    q_text = str(questions)
                parts.append(f"Focus on: {q_text}")

            result[idx] = "\n".join(parts)

    factories = [
        (lambda g=group: _prepass_group(g))
        for group in multi_groups
    ]
    await limiter.run_many(factories)

    return result


# ── Feature 2: Function card synthesis (post-pass, 1 LLM call per function) ──

FUNC_CARD_SYSTEM = (
    "You are a senior code analyst. Generate a structured function card that "
    "captures the contract, phases, and key decisions of a function. "
    "Use the project's own vocabulary. Output ONLY valid JSON."
)


async def _synthesize_function_cards(
    model: str, summaries: list[dict], call_graph: dict,
    per_file_chunks: list[tuple[str, list[str], "ModuleDefinition"]],
    limiter: "AsyncRateLimiter",
    tracker: Optional[TokenTracker] = None,
    quiet: bool = False,
) -> list[dict]:
    """Generate rich function cards for multi-chunk functions.

    After all chunks are summarized, this reads their summaries + raw code
    to produce a comprehensive function card with contract, phases, data flow.
    Returns a list of function_card summary entries to append.
    """
    # Build a lookup: (source_file, chunk_index) → summary entry
    chunk_lookup: dict[tuple[str, int], dict] = {}
    for entry in summaries:
        key = (entry.get("source_file", ""), entry.get("chunk_index", -1))
        if key[1] >= 0:
            chunk_lookup[key] = entry

    # Build chunk lists per file
    file_chunks_map: dict[str, list[str]] = {}
    for rel_path, chunks, _mod in per_file_chunks:
        file_chunks_map[rel_path] = chunks

    # Identify multi-chunk functions
    called_by = call_graph.get("called_by", {})
    calls_map = call_graph.get("calls", {})
    cards: list[dict] = []

    async def _gen_card(rel_path: str, group: list[int], mod_name: str):
        chunks = file_chunks_map.get(rel_path, [])
        if not chunks or group[0] >= len(chunks):
            return

        first_chunk = chunks[group[0]]
        func_name = _extract_function_name(first_chunk) or "unknown"

        # Gather all chunk summaries for this function
        chunk_summaries = []
        combined_code = []
        for idx in group:
            entry = chunk_lookup.get((rel_path, idx))
            if entry:
                chunk_summaries.append(f"Chunk {idx}: {entry.get('summary', '')}")
            if idx < len(chunks):
                combined_code.append(chunks[idx])

        if not chunk_summaries:
            return

        # Gather call graph info
        callers = called_by.get(func_name, [])
        callees = calls_map.get(func_name, [])
        callers_str = ", ".join(callers[:5]) if callers else "unknown"
        callees_str = ", ".join(callees[:5]) if callees else "none detected"

        # Combine code (truncated) for the LLM to see the full function
        full_code = "\n".join(combined_code)
        # Truncate to ~6K chars to stay within context limits
        if len(full_code) > 6000:
            full_code = full_code[:3000] + "\n\n... [truncated] ...\n\n" + full_code[-2000:]

        prompt = f"""Function: {func_name}
File: {rel_path}

Chunk-level summaries (already generated):
{chr(10).join(chunk_summaries)}

Call graph:
  Called by: {callers_str}
  Calls: {callees_str}

Full source code:
```
{full_code}
```

Generate a function card as a JSON object with these fields:
{{
  "name": "{func_name}",
  "purpose": "1-2 sentence description of what this function does and why",
  "contract": {{
    "params": ["param_name: description", ...],
    "returns": "what it returns",
    "preconditions": ["condition that must be true before calling"],
    "postconditions": ["what is guaranteed after successful return"],
    "throws": ["ExceptionType: when"],
    "side_effects": ["what state it mutates beyond return value"]
  }},
  "phases": [
    {{"name": "phase name", "description": "what happens", "lines_approx": "1-80"}}
  ],
  "key_decisions": ["notable design choices, hardcoded values, TODOs"],
  "complexity": "low|medium|high"
}}

Be specific. Use actual parameter names, types, exception types from the code.
If you can't determine a field, use an empty list or "unknown"."""

        try:
            raw = await allm_call(model, FUNC_CARD_SYSTEM, prompt,
                                  max_tokens=800, tracker=tracker,
                                  phase="Func cards")
            raw = raw.strip()
            # Try to parse JSON
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            card_data = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            # Fallback: create a basic card from chunk summaries
            card_data = {
                "name": func_name,
                "purpose": " ".join(s.split(": ", 1)[-1] for s in chunk_summaries[:2]),
                "contract": {},
                "phases": [],
                "key_decisions": [],
                "complexity": "medium",
            }

        # Format as a readable summary for the KB
        card_text = _format_function_card(card_data, rel_path, callers, callees)

        cards.append({
            "summary": card_text,
            "raw_code": "",  # Don't duplicate — raw code already indexed per-chunk
            "source_file": rel_path,
            "module": mod_name,
            "chunk_type": "function_card",
            "chunk_index": -3,  # Negative to distinguish from real chunks
            "function_card": card_data,  # Structured data for programmatic use
        })

    # Collect multi-chunk function groups
    factories = []
    for rel_path, chunks, mod in per_file_chunks:
        if len(chunks) <= 1:
            continue
        groups = _group_chunks_by_function(chunks)
        multi_groups = [g for g in groups if len(g) > 1]
        for group in multi_groups:
            factories.append(
                (lambda rp=rel_path, g=group, mn=mod.name: _gen_card(rp, g, mn))
            )

    if not factories:
        return []

    if not quiet:
        print(f"\n[Function cards] Generating {len(factories)} function cards...")

    await limiter.run_many(factories)

    if not quiet:
        print(f"[Function cards] Generated {len(cards)} cards")

    return cards


def _format_function_card(card: dict, rel_path: str,
                          callers: list[str], callees: list[str]) -> str:
    """Format a function card dict into readable text for the KB."""
    parts = [f"[Function card: {card.get('name', '?')} in {rel_path}]"]
    parts.append(f"Purpose: {card.get('purpose', 'unknown')}")

    contract = card.get("contract", {})
    if contract:
        if contract.get("params"):
            parts.append("Parameters: " + "; ".join(contract["params"][:8]))
        if contract.get("returns"):
            parts.append(f"Returns: {contract['returns']}")
        if contract.get("preconditions"):
            parts.append("Preconditions: " + "; ".join(contract["preconditions"]))
        if contract.get("postconditions"):
            parts.append("Postconditions: " + "; ".join(contract["postconditions"]))
        if contract.get("throws"):
            parts.append("Throws: " + "; ".join(contract["throws"]))
        if contract.get("side_effects"):
            parts.append("Side effects: " + "; ".join(contract["side_effects"]))

    phases = card.get("phases", [])
    if phases:
        phase_strs = [f"{p.get('name', '?')}: {p.get('description', '?')}"
                      for p in phases[:6]]
        parts.append("Phases: " + " → ".join(phase_strs))

    decisions = card.get("key_decisions", [])
    if decisions:
        parts.append("Key decisions: " + "; ".join(decisions[:5]))

    if callers:
        parts.append("Called by: " + ", ".join(callers[:5]))
    if callees:
        parts.append("Calls: " + ", ".join(callees[:5]))

    parts.append(f"Complexity: {card.get('complexity', 'unknown')}")

    return "\n".join(parts)


# ── Class-level summaries ──────────────────────────────────────────────────────

def _extract_class_name(chunk_text: str) -> Optional[str]:
    """Extract a class/struct declaration name from a chunk."""
    first_lines = "\n".join(chunk_text.splitlines()[:10])
    for pattern in _CLASS_PATTERNS:
        m = pattern.search(first_lines)
        if m:
            name = m.group(1)
            if name.lower() not in _KEYWORDS:
                return name
    return None


def _extract_class_from_methods(chunk_text: str) -> Optional[str]:
    """Extract the owning class from C++ ClassName::method patterns."""
    matches = _CPP_CLASS_METHOD.findall(chunk_text)
    if not matches:
        return None
    # Count occurrences of each class name, return the most common
    from collections import Counter
    class_counts = Counter(cls for cls, _ in matches if cls.lower() not in _KEYWORDS)
    if class_counts:
        return class_counts.most_common(1)[0][0]
    return None


def _detect_classes_in_file(chunks: list[str], rel_path: str,
                            codebase: Path) -> dict[str, str]:
    """Detect classes referenced in a file's chunks.

    Returns a dict mapping class_name → header snippet (first ~30 lines of
    the class declaration). For C++, also looks up the .h file if chunks
    contain ClassName::method patterns without a class declaration.
    """
    classes: dict[str, str] = {}  # class_name → declaration snippet

    # 1. Find class declarations directly in the chunks
    for chunk in chunks:
        name = _extract_class_name(chunk)
        if name and name not in classes:
            # Grab the declaration context (from class line to ~30 lines after)
            lines = chunk.splitlines()
            for j, line in enumerate(lines):
                if name in line and ("class " in line or "struct " in line):
                    snippet = "\n".join(lines[j:j + 30])
                    classes[name] = snippet
                    break

    # 2. For C++ method implementations (Foo::bar), find the class in headers
    method_classes: set[str] = set()
    for chunk in chunks:
        cls = _extract_class_from_methods(chunk)
        if cls and cls not in classes:
            method_classes.add(cls)

    # Look up header files for method classes
    for cls_name in method_classes:
        # Common patterns: ClassName.h, class_name.h, ClassName.hpp
        candidates = [
            f"{cls_name}.h", f"{cls_name}.hpp",
            f"{cls_name.lower()}.h", f"{cls_name.lower()}.hpp",
        ]
        # Also try: convert CamelCase to snake_case
        snake = _re.sub(r'(?<!^)(?=[A-Z])', '_', cls_name).lower()
        if snake != cls_name.lower():
            candidates.extend([f"{snake}.h", f"{snake}.hpp"])

        for candidate in candidates:
            match = _find_file(codebase, candidate)
            if match:
                content = read_file_sample(match, max_lines=60)
                # Find the class declaration in the header
                for pattern in _CLASS_PATTERNS:
                    m = pattern.search(content)
                    if m and m.group(1) == cls_name:
                        # Grab from class line to ~30 lines after
                        lines = content.splitlines()
                        for j, line in enumerate(lines):
                            if cls_name in line:
                                classes[cls_name] = "\n".join(lines[j:j + 30])
                                break
                        break
                if cls_name in classes:
                    break

    return classes


CLASS_SUMMARY_SYSTEM = (
    "You are a code analyst. Write 1-2 sentences describing what this class does "
    "and its key responsibilities. Use the project's own vocabulary."
)


async def _generate_class_summaries(
    model: str, classes: dict[str, str], rel_path: str,
    file_summary: str, limiter: "AsyncRateLimiter",
    tracker: Optional[TokenTracker] = None,
) -> dict[str, str]:
    """Generate summaries for classes found in a file.

    Returns a dict mapping class_name → 1-2 sentence summary.
    """
    if not classes:
        return {}

    result: dict[str, str] = {}

    async def _summarize_class(cls_name: str, snippet: str):
        prompt = f"""File: {rel_path}
{f'File purpose: {file_summary}' if file_summary else ''}

Class/struct declaration:
```
{snippet[:2500]}
```

1-2 sentences describing what {cls_name} does and its key responsibilities:"""

        try:
            summary = await allm_call(model, CLASS_SUMMARY_SYSTEM, prompt,
                                      max_tokens=100, tracker=tracker,
                                      phase="Class summaries")
            result[cls_name] = summary.strip()
        except Exception:
            pass

    factories = [
        (lambda cn=cls_name, sn=snippet: _summarize_class(cn, sn))
        for cls_name, snippet in classes.items()
    ]
    await limiter.run_many(factories)
    return result


def _match_chunk_to_class(chunk_text: str, class_names: set[str]) -> Optional[str]:
    """Determine which class a chunk belongs to.

    Checks for: class declaration in chunk, ClassName::method patterns,
    or self/this usage within a known class context.
    """
    # Direct class declaration
    name = _extract_class_name(chunk_text)
    if name and name in class_names:
        return name

    # C++ ClassName::method
    cls = _extract_class_from_methods(chunk_text)
    if cls and cls in class_names:
        return cls

    return None


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
        global _search_kb_failures
        _search_kb_failures += 1
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


def _index_one_entry(client, entry: dict) -> tuple[str, Optional[str], Optional[str]]:
    """Index one summary + raw_code into R2R. Returns (src_file, doc_id, code_doc_id).

    Entries with skip_index=True are not indexed (summary is kept in
    summaries.json for completeness but doesn't go into the vector DB).
    """
    src_file = entry.get("source_file", "")
    module = entry.get("module", "")
    doc_id = None
    code_doc_id = None

    # Low-value chunks flagged by the LLM — skip indexing
    if entry.get("skip_index"):
        return (src_file, None, None)

    chunk_type = entry.get("chunk_type", "function_summary")
    # Function cards get a dedicated source_kind for better search scoping
    source_kind = "reference"
    if chunk_type == "function_card":
        source_kind = "specification"
    elif chunk_type in ("file_overview", "class_overview"):
        source_kind = "overview"

    # Map chunk_category to source_kind for better search scoping
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
        # Add classification metadata if present
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


def index_summaries_to_r2r(summaries: list[dict], index_workers: int = 8) -> None:
    """
    Index all summaries into R2R with upsert semantics, using parallel workers.
    If a document already exists (same content hash), it is replaced.
    The new doc_id is stored back into the entry dict.

    Also indexes raw code as a separate chunk (chunk_type: "raw_code") so that
    searches for exact identifiers can match the source code directly.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = _r2r_client()
    skip_count = sum(1 for e in summaries if e.get("skip_index"))
    n = len(summaries)
    if skip_count:
        print(f"\n[Index] Indexing {n - skip_count} summaries into R2R "
              f"({skip_count} low-value skipped, {index_workers} workers)...")
    else:
        print(f"\n[Index] Indexing {n} summaries + raw code into R2R ({index_workers} workers)...")

    # Map future → index so we can write back doc_ids
    with ThreadPoolExecutor(max_workers=index_workers) as pool:
        future_to_idx = {
            pool.submit(_index_one_entry, client, entry): i
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


def _review_score(entry: dict) -> float:
    """Score a summary for review priority. Higher = more likely to need review.
    Returns 0.0-1.0. Summaries scoring below a threshold can be skipped."""
    score = 0.0
    summary = entry.get("summary", "")

    # Short summaries are likely shallow
    if len(summary) < 100:
        score += 0.4
    elif len(summary) < 200:
        score += 0.2

    # Vague markers suggest the summary needs improvement
    vague_count = sum(1 for marker in _VAGUE_MARKERS if marker in summary.lower())
    score += min(0.4, vague_count * 0.15)

    # Generic terms suggest lack of domain specificity
    generic = {"function", "method", "class", "handles", "processes", "implements",
               "this code", "this function", "this method"}
    generic_count = sum(1 for g in generic if g in summary.lower())
    score += min(0.2, generic_count * 0.05)

    return min(1.0, score)


async def _async_review_one_summary(model: str, entry: dict,
                                     tracker: Optional[TokenTracker] = None) -> tuple[bool, str]:
    """Async review of a single summary. Returns (was_edited, new_summary)."""
    query = f"{entry.get('module', '')} {entry.get('summary', '')[:120]}"
    kb_context = await asyncio.to_thread(search_kb, query, 3)
    prompt = _build_review_prompt(entry, kb_context)

    raw = await allm_call(model, REVIEW_SYSTEM, prompt, max_tokens=512,
                          json_mode=True, tracker=tracker, phase="Review")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Malformed LLM response — treat as "keep, no edit"
        return (False, "")
    if not data.get("keep") and data.get("improved", "").strip():
        return (True, data["improved"].strip())
    return (False, "")


async def run_review(model: str, summaries: list[dict],
                     tracker: Optional[TokenTracker] = None,
                     workers: int = 4, rpm: int = 60,
                     max_concurrent: int = 50,
                     quiet: bool = False) -> tuple[list[dict], float]:
    """
    Review summaries for quality using async concurrency.
    Smart targeting: only reviews summaries likely to need improvement.
    Returns (summaries_with_improvements, edit_rate).
    """
    # Smart targeting: score and filter
    scored = [(i, _review_score(entry)) for i, entry in enumerate(summaries)]
    threshold = 0.15
    review_indices = [i for i, score in scored if score >= threshold]
    skipped = len(summaries) - len(review_indices)

    print(f"\n[Review] {len(review_indices)} summaries to review "
          f"({skipped} skipped as likely good), async {min(max_concurrent, len(review_indices))} concurrent, {rpm} RPM")

    if not review_indices:
        return summaries, 0.0

    limiter = AsyncRateLimiter(
        max_concurrent=min(max_concurrent, len(review_indices)),
        calls_per_minute=rpm,
    )

    edited_count = 0
    completed_count = 0
    start_time = time.monotonic()

    def on_result(factory_idx, result):
        nonlocal edited_count, completed_count
        was_edited, new_summary = result
        real_idx = review_indices[factory_idx]
        completed_count += 1
        n = completed_count

        if was_edited:
            summaries[real_idx]["summary"] = new_summary
            edited_count += 1
            tag = "edited"
        else:
            tag = "kept"

        if not quiet:
            elapsed = time.monotonic() - start_time
            if n > 1 and elapsed > 0:
                rate = n / elapsed * 60
                remaining = (len(review_indices) - n) / (n / elapsed)
                eta = f"  [{rate:.1f}/min, ~{int(remaining//60)}m{int(remaining%60):02d}s left]"
            else:
                eta = ""
            print(f"  [{n:>4}/{len(review_indices)}] {tag:<7} "
                  f"{summaries[real_idx].get('source_file', '?')}{eta}")

    def on_error(factory_idx, exc):
        nonlocal completed_count
        completed_count += 1
        if not quiet:
            print(f"  [warn] review failed for index {review_indices[factory_idx]}: {exc}")

    factories = [
        (lambda i=i: _async_review_one_summary(model, summaries[i], tracker))
        for i in review_indices
    ]

    await limiter.run_many(factories, on_result=on_result, on_error=on_error)

    edit_rate = edited_count / len(summaries) if summaries else 0.0
    print(f"[Review] {edited_count}/{len(review_indices)} reviewed summaries edited "
          f"({skipped} skipped) — overall edit rate {edit_rate:.1%}")
    return summaries, edit_rate


# ── Pass 1: Agent-based Module Discovery ───────────────────────────────────────

PASS1_SYSTEM = (
    "You are a senior software architect exploring a codebase to build a "
    "semantic knowledge base. Use the provided tools to explore the directory "
    "structure and read key files. When you have thoroughly explored the "
    "codebase, call define_modules to define the module map.\n\n"
    "Guidelines:\n"
    "- IMPORTANT: Explore broadly BEFORE defining modules. Use expand_dirs on "
    "every major directory. Read at least one key file per area. Do NOT rush "
    "to call define_modules after only 2-3 exploration rounds.\n"
    "- A large codebase typically has 5-20+ modules — if you only see 2-3, "
    "you haven't explored enough. Keep exploring.\n"
    "- Each module should map to a directory or group of related directories\n"
    "- Every source file should belong to exactly one module\n"
    "- Write 3-6 domain-specific questions per module (use the project's vocabulary)\n"
    "- Focus questions on HOW things work internally, not just WHAT they are\n"
    "- Mark test modules with is_test: true (unit tests, test frameworks, mocks, "
    "benchmarks, fixtures). These will be excluded from the knowledge base."
)

MAX_EXPLORE_ROUNDS = 20  # cap on interactive exploration rounds
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
                                "is_test": {
                                    "type": "boolean",
                                    "description": (
                                        "True if this module is primarily test code: "
                                        "unit tests, integration tests, test fixtures, "
                                        "test harnesses, mocks, fakes, benchmarks. "
                                        "False for production code."
                                    ),
                                    "default": False,
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
    # Accumulate all expanded dirs across rounds so the tree keeps growing
    _all_expanded: set[str] = set()
    # Use same depth as initial tree
    _base_depth = 3 if len(files) < 500 else 4 if len(files) < 5000 else 5

    def dispatch(name: str, args: dict) -> str:
        if name == "expand_dirs":
            dirs = args.get("dirs", [])
            _all_expanded.update(str(d) for d in dirs)
            tree = build_directory_tree(codebase, files, max_depth=_base_depth,
                                         expand_dirs=_all_expanded)
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
                    # Try cached file index fallback
                    match = _find_file(codebase, Path(rf).name)
                    candidates = [match] if match else []
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

    # Build initial tree — deeper for larger codebases
    initial_depth = 3 if len(files) < 500 else 4 if len(files) < 5000 else 5
    tree = build_directory_tree(codebase, files, max_depth=initial_depth)
    tree_lines = len(tree.split("\n"))
    print(f"[Pass 1] Directory tree: {tree_lines} lines, {len(tree)} chars")

    # Build initial prompt
    docs_section = ""
    if docs_context:
        docs_section = f"\n\nDesign documentation (excerpts):\n{docs_context}"

    # Count top-level dirs to set expectations
    top_dirs = set()
    for f in files:
        rel = f.relative_to(codebase)
        if len(rel.parts) > 1:
            top_dirs.add(rel.parts[0])

    initial_prompt = (
        f"Explore this {language} codebase and define its modules.\n\n"
        f"Directory tree (depth-limited, counts on truncated nodes):\n"
        f"```\n{tree}\n```"
        f"{docs_section}\n\n"
        f"This codebase has {len(files)} files across {len(top_dirs)} "
        f"top-level directories. Explore broadly first — expand the major "
        f"directories, read a few key files in each area, then define modules. "
        f"A project this size likely has 5-20+ distinct modules.\n\n"
        f"You have {MAX_EXPLORE_ROUNDS} exploration rounds — use at least "
        f"half of them exploring before calling define_modules."
    )

    # Create dispatcher
    dispatch = _make_pass1_dispatch(codebase, files, language)

    # Logging callback — user-friendly progress
    _explored_dirs: list[str] = []
    _files_read = 0
    _modules_so_far = 0

    def on_round(round_num: int, tool_name: str, args: dict):
        nonlocal _files_read, _modules_so_far
        if tool_name == "expand_dirs":
            dirs = args.get("dirs", [])
            _explored_dirs.extend(dirs)
            area = dirs[0] if dirs else "..."
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Exploring {area}/ "
                  f"({len(_explored_dirs)} dirs explored)")
        elif tool_name == "list_files":
            dir_path = args.get("dir_path", "?")
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Listing {dir_path}/")
        elif tool_name == "read_files":
            paths = args.get("paths", [])
            _files_read += len(paths)
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Reading {len(paths)} files "
                  f"({_files_read} total read)")
        elif tool_name == "search_kb":
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Searching KB: "
                  f"{args.get('query', '?')[:50]}")
        elif tool_name == "define_modules":
            n = len(args.get("modules", []))
            _modules_so_far = n
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Defining {n} modules")

    print(f"[Pass 1] Exploring codebase structure with {model}...")

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

    test_modules_skipped = 0
    for mod_raw in raw_modules:
        name = mod_raw.get("name", "unknown")
        desc = mod_raw.get("description", "")
        dir_paths = mod_raw.get("dir_paths", [])
        questions = mod_raw.get("questions", [])
        is_test = mod_raw.get("is_test", False)

        # Skip test modules — flagged by LLM or detected by name/keywords
        if is_test or _is_test_module(name, desc, dir_paths):
            test_modules_skipped += 1
            # Still count the files as assigned so they don't become orphans
            for dp in dir_paths:
                for f in _group_files_by_dir(codebase, files, str(dp)):
                    assigned_files.add(str(f.relative_to(codebase)))
            continue

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

    if test_modules_skipped:
        print(f"[Pass 1] Skipped {test_modules_skipped} test module(s)")

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


def build_pass2_prompt(project_desc: str, module_name: str,
                       module_desc: str, questions: list[str],
                       source_file: str, raw_code: str,
                       kb_context: str = "",
                       file_summary: str = "") -> str:
    questions_text = "\n".join(f"- {q}" for q in questions)
    kb_section = (
        f"\nRelevant context from the knowledge base (use this vocabulary):\n{kb_context}\n"
        if kb_context else ""
    )
    # The file_summary field may contain:
    # - A plain file summary ("This file implements X")
    # - Function pre-pass context ("[Function: foo] ... Focus on: ...")
    # - Combined context from multiple sources
    context_section = ""
    focus_section = ""
    if file_summary:
        # Split out "Focus on:" directives for special handling
        lines = file_summary.split("\n")
        context_lines = []
        focus_lines = []
        for line in lines:
            if line.startswith("Focus on:"):
                focus_lines.append(line[len("Focus on:"):].strip())
            else:
                context_lines.append(line)
        if context_lines:
            context_section = "\nContext:\n" + "\n".join(context_lines) + "\n"
        if focus_lines:
            focus_section = (
                "\n\nAdditionally, address these specific questions about this chunk:\n"
                + "\n".join(f"- {q}" for q in focus_lines)
            )

    return f"""Project: {project_desc}
Module: {module_name} — {module_desc}

Domain questions this module answers:
{questions_text}
{kb_section}{context_section}
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
- Note what other functions/methods this code calls (if any)
- If trivial (imports-only, constants, empty stub), one sentence is enough
- Plain prose only — no markdown, no bullets{focus_section}

After the summary, on a new line, write a classification tag in this exact format:
[category:CATEGORY search_value:VALUE]

CATEGORY — what kind of question this chunk answers:
- algorithm: Core logic, state machines, decision trees — "how does X work internally?"
- contract: Interface definition, API, protocol — "how do I use X?" / "what does X expect?"
- glue: Wiring, delegation, configuration, initialization — "how are X and Y connected?"
- error_handling: Error paths, recovery, validation — "what happens when X fails?"
- data_model: Type definitions, schemas, data structures — "what does X look like?"
- boilerplate: Getters/setters, imports, forward declarations, logging — rarely searched for

VALUE — would a developer search for this?
- high: Core business logic, complex algorithms, important APIs
- medium: Supporting logic, internal helpers, secondary interfaces
- low: Trivial code, boilerplate, generated code

Example output:
This function validates incoming TxnRequest objects by checking account balance, applying fraud rules, and verifying spending limits. It calls FraudChecker::evaluate() for ML-based fraud scoring and throws TxnValidationError if any check fails.
[category:algorithm search_value:high]

Summary:"""


def _make_chunk_key(source_file: str, chunk_index: int) -> str:
    """Unique key for a chunk, used to detect already-summarized chunks on resume."""
    return f"{source_file}::{chunk_index}"


def _chunk_content_hash(text: str) -> str:
    """Short hash of chunk content, used to verify chunk hasn't shifted on resume."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ── File-level summaries (pre-Pass 2) ────────────────────────────────────────

FILE_SUMMARY_SYSTEM = (
    "You are a code analyst. Write a single sentence describing what this "
    "source file does in the context of its project. Use the project's own "
    "vocabulary (class names, domain concepts). Be specific, not generic."
)


async def _async_summarize_one_file(model: str, rel_path: str, sample: str,
                                    mod_name: str, mod_desc: str,
                                    tracker: Optional[TokenTracker] = None) -> tuple[str, str]:
    """Generate a 1-sentence file summary (async). Returns (rel_path, summary)."""
    prompt = f"""Module: {mod_name} — {mod_desc}
File: {rel_path}

```
{sample[:3000]}
```

One sentence describing what this file does:"""
    result = await allm_call(model, FILE_SUMMARY_SYSTEM, prompt, max_tokens=100,
                             tracker=tracker, phase="File summaries")
    return (rel_path, result.strip())


async def _generate_file_summaries(
    model: str, codebase: Path, module_map: "ModuleMap",
    max_concurrent: int = 50, rpm: int = 60,
    tracker: Optional[TokenTracker] = None,
    quiet: bool = False,
) -> dict[str, str]:
    """Generate 1-sentence summaries for each file before chunk-level Pass 2.

    Uses async concurrency for high throughput.
    Returns a dict mapping rel_path → file summary.
    """
    # Collect unique files with their module context
    file_info: dict[str, tuple[str, str, str]] = {}  # rel_path → (fpath, mod_name, mod_desc)
    for mod in module_map.modules:
        for fname in mod.files:
            fpath = codebase / fname
            if not fpath.exists():
                match = _find_file(codebase, Path(fname).name)
                if match:
                    fpath = match
                else:
                    continue
            rel = str(fpath.relative_to(codebase))
            if rel not in file_info:
                file_info[rel] = (str(fpath), mod.name, mod.description)

    n = len(file_info)
    if n == 0:
        return {}

    print(f"\n[File summaries] Generating 1-sentence summaries for {n} files "
          f"(async, {min(max_concurrent, n)} concurrent, {rpm} RPM)...")

    limiter = AsyncRateLimiter(
        max_concurrent=min(max_concurrent, n),
        calls_per_minute=rpm,
    )

    results: dict[str, str] = {}
    done_count = 0
    start = time.monotonic()

    # Pre-read all file samples (disk I/O — fast, do it synchronously)
    items: list[tuple[str, str, str, str]] = []  # (rel, sample, mod_name, mod_desc)
    for rel, (fpath_str, mod_name, mod_desc) in file_info.items():
        sample = read_file_sample(Path(fpath_str), max_lines=SAMPLE_LINES)
        items.append((rel, sample, mod_name, mod_desc))

    def on_result(idx, result):
        nonlocal done_count
        rel_path, summary = result
        if summary:
            results[rel_path] = summary
        done_count += 1
        if not quiet and done_count % 50 == 0:
            elapsed = time.monotonic() - start
            rate = done_count / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{done_count}/{n}] file summaries generated ({rate:.0f}/min)")

    def on_error(idx, exc):
        nonlocal done_count
        done_count += 1

    factories = [
        (lambda r=rel, s=sample, mn=mod_name, md=mod_desc:
            _async_summarize_one_file(model, r, s, mn, md, tracker))
        for rel, sample, mod_name, mod_desc in items
    ]

    await limiter.run_many(factories, on_result=on_result, on_error=on_error)
    if limiter.quota_exhausted:
        global _quota_exhausted
        _quota_exhausted = True
        print(f"[File summaries] ⚠ Quota exhausted after {len(results)}/{n} "
              f"file summaries. Saving progress.")
    else:
        print(f"[File summaries] Done: {len(results)}/{n} files summarized")
    return results


def _classify_chunk(chunk_text: str) -> tuple[str, Optional[str]]:
    """Classify a chunk and extract the primary symbol name.

    Returns (chunk_type, symbol_name_or_none).
    chunk_type is one of: "class_definition", "method_implementation",
    "function_implementation", "file_level_code".
    """
    cls = _extract_class_name(chunk_text)
    if cls:
        return ("class_definition", cls)

    func = _extract_function_name(chunk_text)
    if func:
        if "::" in func:
            return ("method_implementation", func)
        return ("function_implementation", func)

    # Check for class method patterns without a clear signature on line 1
    cls_from_methods = _extract_class_from_methods(chunk_text)
    if cls_from_methods:
        func = _extract_function_name(chunk_text)
        label = f"{cls_from_methods}::{func}" if func else cls_from_methods
        return ("method_implementation", label)

    return ("file_level_code", None)


def _prefix_summary(summary: str, chunk_type: str,
                    symbol_name: Optional[str]) -> str:
    """Prepend symbol name to summary if not already present."""
    if not symbol_name:
        return summary
    # Don't double-prefix if LLM already included the name
    if symbol_name in summary[:80]:
        return summary
    # Format: "ClassName::method — <summary>"
    return f"{symbol_name} — {summary}"



# ── Feature 1: Call graph inversion (zero LLM cost) ──────────────────────────

def _extract_calls_from_summary(summary: str) -> list[str]:
    """Extract function/method names that a summary says this code calls."""
    calls: list[str] = []
    for pattern in _CALL_PATTERNS:
        for m in pattern.finditer(summary):
            name = m.group(1).strip("`")
            # Filter out common false positives
            bare = name.split("::")[-1] if "::" in name else name
            if bare.lower() in _KEYWORDS or len(bare) < 2:
                continue
            if name not in calls:
                calls.append(name)
    return calls


def _extract_calls_from_code(raw_code: str) -> list[str]:
    """Extract function/method call targets from raw source code.

    Complements summary-based extraction — catches calls the LLM
    didn't mention in the summary.
    """
    calls: list[str] = []
    seen: set[str] = set()

    # C++: Namespace::Class::method( or function(
    for m in _re.finditer(r'(?<![.\w])(\w[\w:]*(?:::\w+)+)\s*\(', raw_code):
        name = m.group(1)
        bare = name.split("::")[-1]
        if bare.lower() not in _KEYWORDS and name not in seen and len(bare) >= 2:
            calls.append(name)
            seen.add(name)

    # Plain function calls: func_name( — filter out control-flow keywords
    for m in _re.finditer(r'(?<![.\w:])([a-zA-Z_]\w{2,})\s*\(', raw_code):
        name = m.group(1)
        if (name.lower() not in _KEYWORDS and name not in seen
                and not name[0].isupper()):  # skip type constructors like String(
            # Heuristic: skip ALL_CAPS (likely macros)
            if name != name.upper():
                calls.append(name)
                seen.add(name)

    return calls


def build_call_graph(summaries: list[dict]) -> dict:
    """Build a call graph from summaries: who calls whom and who is called by whom.

    Returns:
        {
            "calls": {"FileA::func1": ["FileB::func2", "FileC::func3"], ...},
            "called_by": {"FileB::func2": ["FileA::func1"], ...},
            "functions": {"func1": {"file": "path/a.cpp", "module": "mod", ...}, ...},
        }
    """
    # Step 1: Build a registry of all known functions/methods
    # Maps symbol_name → {file, module, chunk_type, summary_preview}
    functions: dict[str, dict] = {}
    # Also build basename → list of qualified names for fuzzy matching
    basename_index: dict[str, list[str]] = {}

    for entry in summaries:
        chunk_type = entry.get("chunk_type", "")
        if chunk_type in ("file_overview", "class_overview"):
            continue

        summary = entry.get("summary", "")
        source_file = entry.get("source_file", "")
        module = entry.get("module", "")
        raw_code = entry.get("raw_code", "")

        # Extract the defining symbol from this chunk
        symbol = None
        if " — " in summary[:120]:
            # Our summaries are prefixed: "ClassName::method — description"
            symbol = summary.split(" — ", 1)[0].strip()
        if not symbol and raw_code:
            func = _extract_function_name(raw_code)
            if func:
                symbol = func

        if symbol and symbol not in functions:
            functions[symbol] = {
                "file": source_file,
                "module": module,
                "chunk_type": chunk_type,
                "preview": summary[:120],
            }
            # Index by basename for fuzzy matching
            bare = symbol.split("::")[-1] if "::" in symbol else symbol
            basename_index.setdefault(bare, []).append(symbol)

    # Step 2: For each summary, extract what it calls
    calls: dict[str, list[str]] = {}
    called_by: dict[str, list[str]] = {}

    for entry in summaries:
        chunk_type = entry.get("chunk_type", "")
        if chunk_type in ("file_overview", "class_overview"):
            continue

        summary = entry.get("summary", "")
        raw_code = entry.get("raw_code", "")
        source_file = entry.get("source_file", "")

        # Identify the caller
        caller = None
        if " — " in summary[:120]:
            caller = summary.split(" — ", 1)[0].strip()
        if not caller and raw_code:
            caller = _extract_function_name(raw_code)
        if not caller:
            caller = f"{source_file}:chunk{entry.get('chunk_index', '?')}"

        # Extract callees from both summary text and code
        callees_from_summary = _extract_calls_from_summary(summary)
        callees_from_code = _extract_calls_from_code(raw_code) if raw_code else []

        # Merge and resolve callees
        all_callees: list[str] = []
        seen_callees: set[str] = set()

        for callee in callees_from_summary + callees_from_code:
            if callee in seen_callees or callee == caller:
                continue
            seen_callees.add(callee)

            # Try to resolve to a known function
            resolved = None
            if callee in functions:
                resolved = callee
            else:
                # Fuzzy: match by basename
                bare = callee.split("::")[-1] if "::" in callee else callee
                candidates = basename_index.get(bare, [])
                if len(candidates) == 1:
                    resolved = candidates[0]
                elif candidates:
                    # Multiple matches — prefer one in a different file (cross-file ref)
                    cross = [c for c in candidates if functions[c]["file"] != source_file]
                    resolved = cross[0] if cross else candidates[0]

            if resolved:
                all_callees.append(resolved)

        if all_callees:
            calls[caller] = all_callees
            for callee in all_callees:
                called_by.setdefault(callee, [])
                if caller not in called_by[callee]:
                    called_by[callee].append(caller)

    return {
        "calls": calls,
        "called_by": called_by,
        "functions": functions,
    }


def _enrich_summaries_with_call_graph(summaries: list[dict],
                                       call_graph: dict) -> int:
    """Append 'Called by: ...' to summaries that have known callers.

    Modifies summaries in-place. Returns count of enriched entries.
    """
    called_by = call_graph.get("called_by", {})
    functions = call_graph.get("functions", {})
    enriched = 0

    for entry in summaries:
        chunk_type = entry.get("chunk_type", "")
        if chunk_type in ("file_overview", "class_overview"):
            continue

        summary = entry.get("summary", "")
        raw_code = entry.get("raw_code", "")

        # Identify this entry's symbol
        symbol = None
        if " — " in summary[:120]:
            symbol = summary.split(" — ", 1)[0].strip()
        if not symbol and raw_code:
            symbol = _extract_function_name(raw_code)

        if not symbol or symbol not in called_by:
            continue

        callers = called_by[symbol]
        if not callers:
            continue

        # Format caller list with files
        caller_strs = []
        for c in callers[:5]:  # cap at 5 to keep summary readable
            info = functions.get(c, {})
            f = info.get("file", "")
            if f:
                caller_strs.append(f"{c} ({f})")
            else:
                caller_strs.append(c)

        suffix = f" Called by: {', '.join(caller_strs)}"
        if len(callers) > 5:
            suffix += f" and {len(callers) - 5} more"
        suffix += "."

        # Don't duplicate if already has "Called by"
        if "Called by:" not in summary:
            entry["summary"] = summary.rstrip() + suffix
            enriched += 1

    return enriched


def _extract_cross_file_refs(raw_code: str) -> list[str]:
    """Extract cross-file references from code (includes, imports, namespaces)."""
    import re
    refs: list[str] = []
    for m in re.finditer(r'#include\s*[<"]([^>"]+)[>"]', raw_code):
        refs.append(m.group(1))
    for m in re.finditer(r'(?:from|import)\s+([\w.]+)', raw_code):
        refs.append(m.group(1).replace(".", "/"))
    for m in re.finditer(r'(\w+)::\w+', raw_code):
        refs.append(m.group(1))
    return refs


def _resolve_ref_to_path(ref: str, codebase: Path, rel_path: str) -> Optional[Path]:
    """Try to resolve a cross-file reference to an actual file path."""
    ref_path = codebase / ref
    if ref_path.exists():
        return ref_path
    parent = (codebase / rel_path).parent
    for candidate in [parent / ref, parent / (ref + ".h"),
                      parent / (ref + ".hpp")]:
        if candidate.exists():
            return candidate
    basename = Path(ref).name
    for suffix in [basename, basename + ".h", basename + ".hpp"]:
        match = _find_file(codebase, suffix)
        if match:
            return match
    return None


def _collect_ref_context(raw_code: str, codebase: Path,
                         rel_path: str, max_refs: int = 3) -> str:
    """Read referenced files and return combined context string."""
    refs = _extract_cross_file_refs(raw_code)
    if not refs:
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    for ref in refs[:6]:
        if ref in seen:
            continue
        seen.add(ref)
        ref_path = _resolve_ref_to_path(ref, codebase, rel_path)
        if ref_path and ref_path.is_file():
            content = read_file_sample(ref_path, max_lines=40)
            if len(content.strip()) > 20:
                parts.append(f"[{ref_path.relative_to(codebase)}]\n{content}")
                if len(parts) >= max_refs:
                    break
    return "\n\n".join(parts)


async def _async_resolve_references(model: str, summary: str, raw_code: str,
                                     codebase: Path, rel_path: str,
                                     tracker: Optional[TokenTracker] = None) -> str:
    """Resolve vague references in a summary by reading referenced files
    and re-generating with extra context."""
    extra_context = _collect_ref_context(raw_code, codebase, rel_path)
    if not extra_context:
        return summary

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
        improved = await allm_call(model, PASS2_SYSTEM, refine_prompt,
                                   max_tokens=256,
                                   tracker=tracker, phase="Pass 2 (refine)")
        improved = improved.strip()
        if improved and len(improved) > 20:
            return improved
    except Exception:
        pass
    return summary


async def _async_summarize_one_chunk(
        model: str, codebase: Path, project_desc: str,
        mod_name: str, mod_desc: str, questions: list[str],
        rel_path: str, chunk: str, chunk_index: int,
        rag: bool = False, file_summary: str = "",
        tracker: Optional[TokenTracker] = None) -> dict:
    """Async version: summarize a single code chunk."""
    kb_context = (await asyncio.to_thread(search_kb, f"{mod_name} {chunk[:200]}", 3)
                  if rag else "")
    prompt = build_pass2_prompt(
        project_desc=project_desc,
        module_name=mod_name,
        module_desc=mod_desc,
        questions=questions,
        source_file=rel_path,
        raw_code=chunk,
        kb_context=kb_context,
        file_summary=file_summary,
    )

    summary_text = await allm_call(model, PASS2_SYSTEM, prompt, max_tokens=512,
                                   tracker=tracker, phase="Pass 2")
    summary_text = summary_text.strip()
    if summary_text.lower().startswith("summary:"):
        summary_text = summary_text[len("summary:"):].strip()

    # Parse classification tag: [category:X search_value:Y]
    chunk_category = "unknown"
    search_value = "medium"
    tag_match = _re.search(
        r'\[category:(\w+)\s+search_value:(\w+)\]', summary_text)
    if tag_match:
        chunk_category = tag_match.group(1)
        search_value = tag_match.group(2)
        # Remove the tag from the summary text
        summary_text = summary_text[:tag_match.start()].rstrip()

    # Check if the LLM tagged this as low-value
    skip_index = False
    if summary_text.upper().startswith("SKIP:"):
        skip_index = True
        summary_text = summary_text[5:].strip()
    elif chunk_category == "boilerplate" and search_value == "low":
        skip_index = True

    refined = False
    if not skip_index and _needs_reference_resolution(summary_text):
        improved = await _async_resolve_references(
            model, summary_text, chunk, codebase, rel_path,
            tracker=tracker,
        )
        if improved != summary_text:
            summary_text = improved
            refined = True

    chunk_type, symbol_name = _classify_chunk(chunk)
    summary_text = _prefix_summary(summary_text, chunk_type, symbol_name)

    return {
        "summary":        summary_text,
        "raw_code":       chunk,
        "source_file":    rel_path,
        "module":         mod_name,
        "chunk_type":     chunk_type,
        "chunk_index":    chunk_index,
        "refined":        refined,
        "skip_index":     skip_index,
        "chunk_category": chunk_category,
        "search_value":   search_value,
        "content_hash":   _chunk_content_hash(chunk),
    }


async def run_pass2(model: str, codebase: Path, module_map: ModuleMap,
                    language: str, max_chunks: Optional[int] = None,
                    rag: bool = False,
                    summaries_path: Optional[Path] = None,
                    tracker: Optional[TokenTracker] = None,
                    workers: int = 4, rpm: int = 60,
                    max_concurrent: int = 50,
                    quiet: bool = False, verbose: bool = False) -> list[dict]:
    """Chunk each file and call the LLM to generate a domain-aware summary per chunk.

    Uses async concurrency with rate limiting for high throughput.
    With rag=True, queries the KB before each chunk to inject relevant context.
    If summaries_path is set, writes incrementally for crash-safety.
    """
    global _quota_exhausted

    # Load existing summaries for resume support
    # done_hashes maps chunk_key → content_hash so we can detect shifted chunks
    summaries: list[dict] = []
    done_keys: set[str] = set()
    done_hashes: dict[str, str] = {}
    if summaries_path and summaries_path.exists():
        try:
            summaries = json.loads(summaries_path.read_text())
            for entry in summaries:
                key = _make_chunk_key(
                    entry.get("source_file", ""),
                    entry.get("chunk_index", 0),
                )
                done_keys.add(key)
                if "content_hash" in entry:
                    done_hashes[key] = entry["content_hash"]
            if done_keys:
                print(f"\n[Pass 2] Resuming — {len(done_keys)} chunks already summarized")
        except Exception:
            summaries = []

    # ── Load or generate file-level summaries ────────────────────────────
    # Cache file/class/function summaries so resume doesn't re-generate them
    context_cache_path = (summaries_path.parent / "context_cache.json"
                          if summaries_path else None)
    cached_context = {}
    if context_cache_path and context_cache_path.exists():
        try:
            cached_context = json.loads(context_cache_path.read_text())
            print(f"[Pass 2] Loaded context cache "
                  f"({len(cached_context.get('file_summaries', {}))} file, "
                  f"{len(cached_context.get('class_summaries', {}))} class, "
                  f"{len(cached_context.get('func_summaries', {}))} func summaries)")
        except Exception:
            cached_context = {}

    project_desc = f"{module_map.project}: {module_map.description}"

    if cached_context.get("file_summaries"):
        file_summaries = cached_context["file_summaries"]
        print(f"[File summaries] Loaded {len(file_summaries)} from cache")
    else:
        file_summaries = await _generate_file_summaries(
            model, codebase, module_map,
            max_concurrent=max_concurrent, rpm=rpm, tracker=tracker, quiet=quiet,
        )

    # ── Collect chunks and detect multi-chunk functions ──────────────────
    work_items: list[dict] = []
    skipped = 0
    # Collect per-file chunk data for function grouping
    per_file_chunks: list[tuple[str, list[str], "ModuleDefinition"]] = []

    rag_label = " (RAG-augmented)" if rag else ""
    print(f"\n[Pass 2] Collecting chunks from {len(module_map.modules)} modules{rag_label}...")

    for mod in module_map.modules:
        for fname in mod.files:
            fpath = codebase / fname
            if not fpath.exists():
                match = _find_file(codebase, Path(fname).name)
                candidates = [match] if match else []
                if not candidates:
                    skipped += 1
                    continue
                fpath = candidates[0]

            chunks = chunk_file(fpath, language=language)
            if not chunks:
                skipped += 1
                continue

            rel_path = str(fpath.relative_to(codebase))
            per_file_chunks.append((rel_path, chunks, mod))

    # ── Generate or load function summaries for multi-chunk functions ───
    func_summaries: dict[tuple[str, int], str] = {}

    if not _quota_exhausted:
        if cached_context.get("func_summaries"):
            # Restore from cache — keys are "rel_path|chunk_idx" strings
            for key, summary in cached_context["func_summaries"].items():
                sep = "|" if "|" in key else "::"  # back-compat
                rp, ci = key.rsplit(sep, 1)
                func_summaries[(rp, int(ci))] = summary
            print(f"[Pass 2] Loaded {len(func_summaries)} function summaries from cache")
        else:
            func_limiter = AsyncRateLimiter(
                max_concurrent=max_concurrent, calls_per_minute=rpm,
            )
            multi_chunk_count = 0

            for rel_path, chunks, mod in per_file_chunks:
                if _quota_exhausted:
                    break
                if len(chunks) > 1:
                    groups = _group_chunks_by_function(chunks)
                    multi_groups = [g for g in groups if len(g) > 1]
                    if multi_groups:
                        multi_chunk_count += sum(len(g) - 1 for g in multi_groups)
                        fsums = await _generate_function_summaries(
                            model, chunks, groups, rel_path,
                            file_summaries.get(rel_path, ""),
                            limiter=func_limiter, tracker=tracker,
                        )
                        for chunk_idx, summary in fsums.items():
                            func_summaries[(rel_path, chunk_idx)] = summary
                        if func_limiter.quota_exhausted:
                            _quota_exhausted = True

            if multi_chunk_count > 0:
                print(f"[Pass 2] Generated function context for {multi_chunk_count} "
                      f"continuation chunks across {len(func_summaries)} multi-chunk functions")

    # ── Generate or load class summaries ──────────────────────────────────
    class_summaries: dict[tuple[str, str], str] = {}

    if not _quota_exhausted:
        if cached_context.get("class_summaries"):
            # Restore from cache — keys are "rel_path|class_name" strings
            for key, summary in cached_context["class_summaries"].items():
                sep = "|" if "|" in key else "::"  # back-compat
                rp, cn = key.rsplit(sep, 1)
                class_summaries[(rp, cn)] = summary
            print(f"[Pass 2] Loaded {len(class_summaries)} class summaries from cache")
        else:
            class_limiter = AsyncRateLimiter(
                max_concurrent=max_concurrent, calls_per_minute=rpm,
            )
            class_count = 0

            for rel_path, chunks, mod in per_file_chunks:
                if _quota_exhausted:
                    break
                classes = _detect_classes_in_file(chunks, rel_path, codebase)
                if classes:
                    csums = await _generate_class_summaries(
                        model, classes, rel_path,
                        file_summaries.get(rel_path, ""),
                        limiter=class_limiter, tracker=tracker,
                    )
                    for cls_name, summary in csums.items():
                        class_summaries[(rel_path, cls_name)] = summary
                        class_count += 1
                    if class_limiter.quota_exhausted:
                        _quota_exhausted = True

            if class_count > 0:
                print(f"[Pass 2] Generated summaries for {class_count} classes")

    # Build global class summary lookup (class_name → summary)
    # so chunks in .cpp files can find classes defined in .h files
    global_class_summaries: dict[str, str] = {}
    for (rp, cn), summary in class_summaries.items():
        # Prefer longer summaries if same class found in multiple files
        if cn not in global_class_summaries or len(summary) > len(global_class_summaries[cn]):
            global_class_summaries[cn] = summary

    # Per-file class names for direct matching
    file_class_names: dict[str, set[str]] = {}
    for (rp, cn) in class_summaries:
        file_class_names.setdefault(rp, set()).add(cn)

    # ── Save context cache (resume support) ──────────────────────────────
    # Serialize tuple keys to strings for JSON compatibility
    if context_cache_path:
        _ctx = {
            "file_summaries": file_summaries,
            "class_summaries": {
                f"{rp}|{cn}": s for (rp, cn), s in class_summaries.items()
            },
            "func_summaries": {
                f"{rp}|{ci}": s for (rp, ci), s in func_summaries.items()
            },
        }
        context_cache_path.write_text(json.dumps(_ctx, indent=2))
        print(f"[Pass 2] Saved context cache ({len(file_summaries)} file, "
              f"{len(class_summaries)} class, {len(func_summaries)} func)")

    # ── Build work items ──────────────────────────────────────────────────
    stale_count = 0
    for rel_path, chunks, mod in per_file_chunks:
        for i, chunk in enumerate(chunks):
            chunk_key = _make_chunk_key(rel_path, i)
            if chunk_key in done_keys:
                # Verify content hasn't shifted since last run
                if chunk_key in done_hashes:
                    current_hash = _chunk_content_hash(chunk)
                    if done_hashes[chunk_key] != current_hash:
                        # Chunk content changed — remove stale summary, re-summarize
                        stale_count += 1
                        summaries = [
                            s for s in summaries
                            if _make_chunk_key(
                                s.get("source_file", ""),
                                s.get("chunk_index", 0),
                            ) != chunk_key
                        ]
                        done_keys.discard(chunk_key)
                        # Fall through to add as work item
                    else:
                        continue
                else:
                    continue
            # Combine file summary + class context + function context
            context_parts: list[str] = []
            fs = file_summaries.get(rel_path, "")
            if fs:
                context_parts.append(fs)

            # Match chunk to a class and inject class summary
            # First check classes found in this file, then global lookup
            # (handles .cpp files referencing classes defined in .h files)
            all_known = file_class_names.get(rel_path, set()) | set(global_class_summaries.keys())
            if all_known:
                cls = _match_chunk_to_class(chunk, all_known)
                if cls:
                    cls_summary = (class_summaries.get((rel_path, cls), "")
                                   or global_class_summaries.get(cls, ""))
                    if cls_summary:
                        context_parts.append(f"Class {cls}: {cls_summary}")

            func_ctx = func_summaries.get((rel_path, i), "")
            if func_ctx:
                context_parts.append(func_ctx)

            work_items.append({
                "mod_name": mod.name,
                "mod_desc": mod.description,
                "questions": mod.questions,
                "rel_path": rel_path,
                "chunk": chunk,
                "chunk_index": i,
                "file_summary": "\n".join(context_parts),
            })

    # ── Dedup near-identical chunks (boilerplate, generated code) ────────
    # Uses content hash + length + structure to avoid false matches.
    def _chunk_signature(text: str) -> tuple:
        h = hashlib.sha256(text.strip().encode()).hexdigest()[:16]
        tokens = text.split()
        lines = text.splitlines()
        return (h, len(tokens), len(lines))

    seen_sigs: dict[tuple, str] = {}
    deduped: list[dict] = []
    dedup_count = 0
    for item in work_items:
        sig = _chunk_signature(item["chunk"])
        if sig in seen_sigs:
            dedup_count += 1
            if not quiet:
                print(f"  [dedup] {item['rel_path']}:{item['chunk_index']} "
                      f"≈ {seen_sigs[sig]}")
            continue
        seen_sigs[sig] = f"{item['rel_path']}:{item['chunk_index']}"
        deduped.append(item)
    if dedup_count > 0:
        print(f"[Pass 2] Deduped {dedup_count} exact-duplicate chunks")
    work_items = deduped

    if max_chunks is not None:
        work_items = work_items[:max_chunks]

    total_cached = len(done_keys)
    if stale_count > 0:
        print(f"[Pass 2] {stale_count} cached chunks had stale content — re-summarizing")
    print(f"[Pass 2] {len(work_items)} new chunks to summarize "
          f"({total_cached} cached, {skipped} files skipped)")

    if not work_items:
        print("[Pass 2] Nothing to do.")
        return summaries

    if _quota_exhausted:
        print(f"\n⚠ Quota was exhausted during context generation. "
              f"Saving {len(summaries)} existing summaries. "
              f"Re-run to resume with {len(work_items)} remaining chunks.")
        if summaries_path:
            summaries_path.write_text(json.dumps(summaries, indent=2))
        return summaries

    # ── Process with async concurrency ────────────────────────────────────
    effective_concurrent = min(max_concurrent, len(work_items))
    print(f"[Pass 2] Async: {effective_concurrent} concurrent, {rpm} RPM\n")

    limiter = AsyncRateLimiter(
        max_concurrent=effective_concurrent,
        calls_per_minute=rpm,
    )

    start_time = time.monotonic()
    completed_count = 0
    refined_count = 0
    error_count = 0
    # Lock protects summaries list + counters against concurrent on_result calls
    _result_lock = asyncio.Lock()

    async def on_result(idx, entry):
        nonlocal completed_count, refined_count
        async with _result_lock:
            ref_tag = ""
            if entry.pop("refined", False):
                ref_tag = " [refined]"
                refined_count += 1

            summaries.append(entry)
            completed_count += 1
            n = completed_count

            elapsed = time.monotonic() - start_time
            if n > 1 and elapsed > 0:
                rate = n / elapsed * 60
                remaining = (len(work_items) - n) / (n / elapsed)
                eta = f"  [{rate:.1f}/min, ~{int(remaining//60)}m{int(remaining%60):02d}s left]"
            else:
                eta = ""
            if not quiet:
                print(f"  [{n:>4}/{len(work_items)}] "
                      f"{work_items[idx]['rel_path']}:{work_items[idx]['chunk_index']} → "
                      f"{len(entry['summary'])} chars{ref_tag}{eta}")

            # Incremental write every 10 completions — snapshot while holding lock
            if summaries_path and n % 10 == 0:
                summaries_path.write_text(json.dumps(list(summaries), indent=2))

    def on_error(idx, exc):
        nonlocal error_count
        error_count += 1
        item = work_items[idx]
        print(f"  [error] {item['rel_path']}:{item['chunk_index']}: {exc}")

    factories = [
        (lambda it=item: _async_summarize_one_chunk(
            model, codebase, project_desc,
            it["mod_name"], it["mod_desc"], it["questions"],
            it["rel_path"], it["chunk"], it["chunk_index"],
            rag=rag, file_summary=it.get("file_summary", ""),
            tracker=tracker,
        ))
        for item in work_items
    ]

    await limiter.run_many(factories, on_result=on_result, on_error=on_error)

    # Save immediately after run_many completes (especially important on quota halt)
    if summaries_path:
        summaries_path.write_text(json.dumps(summaries, indent=2))

    if limiter.quota_exhausted:
        print(f"\n⚠ Quota exhausted after {completed_count} chunk summaries. "
              f"{limiter.skipped_count} chunks skipped.")
        print(f"  Progress saved to {summaries_path}. Re-run to resume.")
        return summaries

    # ── Append file and class overview summaries for indexing ────────────
    overview_count = 0
    for rel_path, _chunks, mod in per_file_chunks:
        fs = file_summaries.get(rel_path, "")
        if fs:
            summaries.append({
                "summary": f"[File overview: {rel_path}] {fs}",
                "raw_code": "",
                "source_file": rel_path,
                "module": mod.name,
                "chunk_type": "file_overview",
                "chunk_index": -1,
            })
            overview_count += 1

    for (rel_path, cls_name), cls_summary in class_summaries.items():
        if cls_summary:
            # Find the module for this file
            mod_name = ""
            for rp, _ch, mod in per_file_chunks:
                if rp == rel_path:
                    mod_name = mod.name
                    break
            summaries.append({
                "summary": f"[Class overview: {cls_name} in {rel_path}] {cls_summary}",
                "raw_code": "",
                "source_file": rel_path,
                "module": mod_name,
                "chunk_type": "class_overview",
                "chunk_index": -2,
            })
            overview_count += 1

    # ── Feature 1: Call graph inversion (no LLM cost) ──────────────────
    call_graph = build_call_graph(summaries)
    n_functions = len(call_graph.get("functions", {}))
    n_edges = sum(len(v) for v in call_graph.get("called_by", {}).values())
    enriched = _enrich_summaries_with_call_graph(summaries, call_graph)
    print(f"[Call graph] {n_functions} functions, {n_edges} edges, "
          f"{enriched} summaries enriched with 'Called by'")

    # Save call graph as a standalone artifact
    if summaries_path:
        cg_path = summaries_path.parent / "call_graph.json"
        # Serialize only the calls/called_by (functions dict is large)
        cg_export = {
            "calls": call_graph.get("calls", {}),
            "called_by": call_graph.get("called_by", {}),
            "stats": {
                "functions": n_functions,
                "edges": n_edges,
                "enriched_summaries": enriched,
            },
        }
        cg_path.write_text(json.dumps(cg_export, indent=2))

    # ── Feature 2: Function card synthesis ────────────────────────────────
    func_card_count = 0
    if not _quota_exhausted:
        card_limiter = AsyncRateLimiter(
            max_concurrent=min(max_concurrent, 20),  # conservative for card gen
            calls_per_minute=rpm,
        )
        func_cards = await _synthesize_function_cards(
            model, summaries, call_graph, per_file_chunks,
            limiter=card_limiter, tracker=tracker, quiet=quiet,
        )
        if func_cards:
            summaries.extend(func_cards)
            func_card_count = len(func_cards)
            if card_limiter.quota_exhausted:
                _quota_exhausted = True

    # Final write
    if summaries_path:
        summaries_path.write_text(json.dumps(summaries, indent=2))

    # Classification breakdown
    cat_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}
    skip_count = 0
    for e in summaries:
        cat = e.get("chunk_category", "")
        val = e.get("search_value", "")
        if cat:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if val:
            val_counts[val] = val_counts.get(val, 0) + 1
        if e.get("skip_index"):
            skip_count += 1

    print(f"\n[Pass 2] Done: {len(summaries)} summaries total "
          f"({completed_count} new, {total_cached} cached, "
          f"{error_count} errors, {refined_count} refined, "
          f"{overview_count} overviews, {func_card_count} function cards)")
    if cat_counts:
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]))
        print(f"[Pass 2] Categories: {cat_str}")
    if val_counts:
        val_str = ", ".join(f"{k}={v}" for k, v in sorted(val_counts.items(), key=lambda x: -x[1]))
        print(f"[Pass 2] Search value: {val_str}")
    if skip_count:
        print(f"[Pass 2] {skip_count} chunks marked low-value (will skip indexing)")
    if _search_kb_failures > 0:
        print(f"  [warn] {_search_kb_failures} RAG queries failed — "
              f"is R2R running? (summaries were generated without KB context)")
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
    parser.add_argument("--model-fast", default=None,
                        help="Cheaper model for bulk Pass 2 summarization. "
                             "If set, --model is used for Pass 1 and Review only. "
                             "Example: --model openai/gpt-4o --model-fast openai/gpt-4o-mini")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write module_map.json and summaries.json "
                             "(default: same dir as this script)")
    parser.add_argument("--language", default=None,
                        choices=["python", "javascript", "typescript", "cpp", "java", "go", "rust"],
                        help="Primary language (auto-detected if omitted)")
    parser.add_argument("--docs", nargs="+", default=None,
                        help="Path(s) to docs files/directories. Used in Pass 1 context. "
                             "Also indexed into R2R when --bootstrap-docs is set. "
                             "Can specify multiple: --docs /path/a /path/b")
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
    parser.add_argument("--exclude", nargs="+", default=None,
                        help="Additional directories to skip. Can specify multiple: "
                             "--exclude generated proto_out experimental")
    parser.add_argument("--include-tests", action="store_true",
                        help="Include test files and test directories (skipped by default)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for Pass 2 summarization (default: 4)")
    parser.add_argument("--rpm", type=int, default=60,
                        help="Rate limit: max LLM calls per minute (default: 60)")
    parser.add_argument("--max-concurrent", type=int, default=50,
                        help="Max concurrent async requests (default: 50). "
                             "Higher values use more memory but better saturate high RPM limits.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate token cost without running any LLM calls")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip cost confirmation prompt")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress (tool calls, reference resolution)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-chunk progress, show only phase summaries")
    args = parser.parse_args()

    codebase = Path(args.codebase).resolve()
    if not codebase.is_dir():
        print(f"Error: --codebase '{codebase}' is not a directory")
        sys.exit(1)

    if args.passes < 1:
        args.passes = 1

    # Add user-specified exclusions to SKIP_DIRS
    if args.exclude:
        SKIP_DIRS.update(args.exclude)
        print(f"[Config] Excluding additional dirs: {', '.join(args.exclude)}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    module_map_path = output_dir / "module_map.json"
    summaries_path  = output_dir / "summaries.json"

    # ── Auto-detect language if not specified ────────────────────────────────
    if args.language is None:
        detected = detect_language(codebase)
        if detected:
            args.language = detected
        else:
            print("Could not auto-detect language. Use --language to specify.")
            sys.exit(1)

    # ── Collect source files ────────────────────────────────────────────────────
    files = collect_source_files(codebase, language=args.language,
                                 include_tests=args.include_tests)
    skip_note = "" if args.include_tests else " (excluding tests — use --include-tests to change)"
    print(f"Found {len(files)} {args.language} source files under {codebase}{skip_note}")
    if args.model_fast:
        print(f"Models: {args.model} (Pass 1 + Review), {args.model_fast} (Pass 2 bulk)")
    else:
        print(f"Model: {args.model}")
    if not files:
        print(red("No source files found.") + " Check --codebase and --language.")
        sys.exit(1)

    # ── Cost estimate ─────────────────────────────────────────────────────────
    est_chunks = len(files) * 3
    est_prompt_tokens = est_chunks * 2000
    est_completion_tokens = est_chunks * 500
    est_total = est_prompt_tokens + est_completion_tokens
    file_summary_tokens = len(files) * 500
    est_total += file_summary_tokens
    func_prepass_tokens = int(est_chunks * 0.2 * 600)   # pre-pass with questions
    est_total += func_prepass_tokens
    func_card_tokens = int(est_chunks * 0.15 * 1200)   # function card synthesis
    est_total += func_card_tokens
    class_summary_tokens = int(len(files) / 5 * 500)
    est_total += class_summary_tokens
    pass1_tokens = 50000
    est_total += pass1_tokens
    # Try to get real pricing from litellm's model database
    rate_per_mtok = 5.0  # fallback: ~GPT-4o blended rate
    _pricing_source = "default estimate"
    try:
        from litellm import get_model_info
        _cost_model = args.model_fast or args.model
        info = get_model_info(_cost_model)
        input_cpt = info.get("input_cost_per_token", 0)
        output_cpt = info.get("output_cost_per_token", 0)
        if input_cpt and output_cpt:
            # Weighted blend: ~80% prompt, ~20% completion
            rate_per_mtok = (input_cpt * 0.8 + output_cpt * 0.2) * 1_000_000
            _pricing_source = _cost_model
    except Exception:
        pass
    est_cost = est_total / 1_000_000 * rate_per_mtok

    total_calls = est_chunks
    if args.passes > 1:
        review_tokens = est_chunks * 1500
        est_total += review_tokens * (args.passes - 1)
        total_calls += int(est_chunks * 0.7)
    est_minutes = total_calls / args.rpm

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN — Cost Estimate")
        print(f"{'='*60}")
        print(f"  Source files:      {len(files):,}")
        print(f"  Est. chunks:       ~{est_chunks:,} (avg 3/file)")
        print(f"  Est. Pass 1:       ~{pass1_tokens:,} tokens")
        print(f"  Est. file summaries: ~{file_summary_tokens:,} tokens ({len(files):,} files × ~500 tok)")
        print(f"  Est. func pre-pass:  ~{func_prepass_tokens:,} tokens (~20% multi-chunk funcs × ~600 tok)")
        print(f"  Est. func cards:     ~{func_card_tokens:,} tokens (~15% multi-chunk funcs × ~1200 tok)")
        print(f"  Est. class summaries: ~{class_summary_tokens:,} tokens (~1 class per 5 files)")
        print(f"  Est. Pass 2:       ~{est_prompt_tokens + est_completion_tokens:,} tokens")
        print(f"                     ({est_chunks:,} chunks × ~2,500 tok/chunk)")
        print(f"  Est. total tokens: ~{est_total:,}")
        print(f"  Est. cost:         ~${est_cost:.2f} (at ${rate_per_mtok:.2f}/MTok — {_pricing_source})")
        if args.passes > 1:
            print(f"  Est. review:       ~{review_tokens:,} tokens/pass × {args.passes - 1} pass(es)")
            print(f"  Est. total w/review: ~{est_total:,} tokens (~${est_total / 1_000_000 * rate_per_mtok:.2f})")
        if est_minutes > 60:
            print(f"  Est. wall time:    ~{est_minutes / 60:.1f} hours (at {args.rpm} RPM)")
        else:
            print(f"  Est. wall time:    ~{int(est_minutes)} minutes (at {args.rpm} RPM)")
        print(f"{'='*60}")
        if not args.model_fast:
            print(f"\n  Tip: Use --model-fast openai/gpt-4o-mini to reduce Pass 2 cost by ~90%")
        print(f"\n  Note: Actual cost depends on model, file sizes, and content.")
        print(f"  Run without --dry-run to proceed.")
        return

    # Show cost estimate and confirm (unless --yes)
    if not args.yes:
        est_cost_total = est_total / 1_000_000 * rate_per_mtok
        if est_minutes > 60:
            time_str = f"~{est_minutes / 60:.1f} hours"
        else:
            time_str = f"~{int(est_minutes)} minutes"
        print(f"\nEstimate: ~{est_chunks:,} chunks, ~${est_cost_total:.2f}, {time_str} at {args.rpm} RPM")
        if not args.model_fast and len(files) > 100:
            print(f"  Tip: --model-fast openai/gpt-4o-mini cuts Pass 2 cost ~90%")
        try:
            answer = input("Proceed? [Y/n] ").strip().lower()
            if answer and answer not in ("y", "yes"):
                print("Aborted.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return

    # ── Smart RPM detection ─────────────────────────────────────────────────────
    # If user didn't explicitly set --rpm, try to detect from proxy headers
    if args.rpm == parser.get_default("rpm"):
        detected_rpm = detect_rpm_from_proxy(args.model_fast or args.model)
        if detected_rpm and detected_rpm != args.rpm:
            print(f"{green('[Auto]')} Detected rate limit: {bold(str(detected_rpm))} RPM from proxy headers")
            args.rpm = detected_rpm
            # Recalculate ETA with new RPM
            est_minutes = total_calls / args.rpm

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
                args.docs = [str(candidate_path)]
                args.bootstrap_docs = True
                print(f"[Auto] Found docs at {candidate_path}, will bootstrap")
                break

    if args.bootstrap_docs:
        if not args.docs:
            print("Error: --bootstrap-docs requires --docs PATH")
            sys.exit(1)
        for doc_path in args.docs:
            bootstrap_docs(Path(doc_path))

    tracker = TokenTracker()
    _start_time = time.time()

    # ── Pass 1 ──────────────────────────────────────────────────────────────────
    if not args.pass2_only:
        docs_context = None
        if args.docs:
            parts = []
            for doc_arg in args.docs:
                docs_path = Path(doc_arg)
                if docs_path.is_file():
                    parts.append(f"### {docs_path.name}\n"
                                 f"{docs_path.read_text(encoding='utf-8', errors='replace')[:4000]}")
                elif docs_path.is_dir():
                    for doc_file in sorted(docs_path.rglob("*")):
                        if doc_file.suffix in {".md", ".rst", ".txt"} and doc_file.is_file():
                            parts.append(f"### {doc_file.name}\n"
                                         f"{doc_file.read_text(errors='replace')[:2000]}")
                        if sum(len(p) for p in parts) > 8000:
                            break
                if sum(len(p) for p in parts) > 8000:
                    break
            docs_context = "\n\n".join(parts) if parts else None

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

    # ── Show ETA ─────────────────────────────────────────────────────────────
    if est_minutes > 60:
        print(f"\nEstimated time remaining: ~{est_minutes / 60:.1f} hours at {args.rpm} RPM")
    elif est_minutes > 1:
        print(f"\nEstimated time remaining: ~{int(est_minutes)} minutes at {args.rpm} RPM")

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
            summaries = asyncio.run(run_pass2(
                args.model_fast or args.model, codebase, module_map,
                language=args.language,
                max_chunks=args.max_chunks,
                rag=args.rag,
                summaries_path=summaries_path,
                tracker=tracker,
                workers=args.workers,
                rpm=args.rpm,
                max_concurrent=args.max_concurrent,
                quiet=args.quiet,
                verbose=args.verbose,
            ))

        if args.passes > 1:
            # Index summaries so the review pass can query them
            index_summaries_to_r2r(summaries)

            # Review + improve
            summaries, edit_rate = asyncio.run(run_review(
                args.model, summaries, tracker=tracker,
                workers=args.workers, rpm=args.rpm,
                max_concurrent=args.max_concurrent,
                quiet=args.quiet,
            ))

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

    # Always index into R2R (previously only done for --passes > 1)
    if summaries:
        print(f"\n[Index] Indexing {len(summaries)} summaries into R2R...")
        index_summaries_to_r2r(summaries)

    # ── Token usage + cost log ─────────────────────────────────────────────────
    total_tokens = 0
    if tracker.phases:
        total_tokens = sum(
            p["prompt"] + p["completion"] for p in tracker.phases.values()
        )
        cost_log_path = output_dir / "cost_log.jsonl"
        entry = tracker.to_log_entry(model=args.model, codebase=codebase.name)
        with cost_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Post-run summary card ──────────────────────────────────────────────────
    elapsed = time.time() - _start_time

    n_modules = len(module_map.modules) if module_map else 0
    cost_str = f"~${total_tokens / 1_000_000 * rate_per_mtok:.2f}" if total_tokens else "n/a"
    if elapsed > 3600:
        time_str = f"{elapsed / 3600:.1f}h"
    elif elapsed > 60:
        time_str = f"{elapsed / 60:.0f}m"
    else:
        time_str = f"{elapsed:.0f}s"

    w = 44
    hr = green("=" * w)
    print(f"\n{hr}")
    print(f"  {ok('Study Complete')}")
    print(f"  Files: {len(files):,}  Chunks: {len(summaries):,}  Modules: {n_modules}")
    print(f"  Tokens: {total_tokens:,}  Cost: {cost_str}  Time: {time_str}")
    if _quota_exhausted:
        print(f"  {warn('Quota exhausted')} — partial results saved. Re-run to resume.")
    print(f"{hr}")
    print(f"\n  Next: Open Claude Code in this directory")
    print(f"  Try: {dim('\"How does X work?\"')} or {dim('\"Where is Y implemented?\"')}")
    print()

    if tracker.phases:
        print(dim(tracker.summary()))


if __name__ == "__main__":
    main()
