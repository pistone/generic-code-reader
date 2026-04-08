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
  - LLM can call search_kb during summarization to look up unfamiliar types/concepts
  - Output: summaries.json  →  auto-indexed to R2R after Pass 2

Pass 3 — Review  (runs automatically when --passes > 1)
  - LLM reviews each summary for accuracy and domain vocabulary
  - Rewrites weak summaries; improves the vector DB in-place
  - Stops early when edit rate drops below 5% (convergence)

Usage:
  # Standard run
  OPENAI_API_KEY=sk-... python study_agent.py --codebase /path/to/src

  # Skip Pass 1 if module_map.json already exists
  python study_agent.py --codebase /path/to/src --summarize
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
# Keywords that identify a module as test-related (case-insensitive match)
_TEST_MODULE_KEYWORDS = {"test", "testing", "mock", "fake", "stub",
                         "fixture", "harness", "benchmark"}

# Compiled regex for test directory detection.
# Matches: test, tests, testing, test_*, *test, *tests, *testing
# Also: jtest, cstest, rstest, ktest, jstest, unittest, testutil, etc.
# Does NOT match: "latest", "fastest", "greatest", "contest", "attest",
# Test directory detection: explicit prefix allowlist avoids false positives
# on English words like "latest", "fastest", "attest", "contest", "detest".
# Add entries to _XTEST_PREFIXES for project-specific conventions.
import re as _re_test

# Known short prefixes that form test dir names: {prefix}test(s)
# e.g. jtest (Java), cstest (C#), rstest (Rust), gtest (Google Test), unittest
_XTEST_PREFIXES = (
    'j', 'k', 'g', 'c', 'u', 'e', 'i', 'p', 'n',   # single-letter
    'js', 'cs', 'rs', 'ts', 'go',                      # two-letter
    'cpp', 'sys', 'api', 'gui', 'cli', 'net',          # three-letter
    'unit', 'perf', 'fuzz', 'load', 'func',            # four-letter
)
_xtest_alt = '|'.join(sorted(_XTEST_PREFIXES, key=len, reverse=True))

_TEST_DIR_RE = _re_test.compile(
    r'^(?:tests?|testing)(?:[_\-s]|$|[a-z])'           # starts with test: test, testutil, testcases
    r'|^(?:' + _xtest_alt + r')tests?(?:[_\-]|$)'      # known prefix + test: jtest, cstest, gtest
    r'|^(?:mock|fake|stub|fixture|benchmark)',           # known non-test prefixes
    _re_test.IGNORECASE,
)


def _is_test_path(p: Path) -> bool:
    """Check if a path is test-related by directory name pattern.

    Catches: test, tests, testing, test_utils, testutil, jtest, cstest,
    rstest, ktest, jstest, unittest, mock_*, fake_*, etc.
    """
    for part in p.parts:
        if part.lower() in SKIP_TEST_DIRS:
            return True
        if _TEST_DIR_RE.search(part):
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
        if part.lower() in SKIP_TEST_DIRS:
            return True
        if _TEST_DIR_RE.search(part):
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

# ── Caller frequency map ──────────────────────────────────────────────────
# Single-pass scan of all source files to count how often each function/method
# is called. Used to inject "widely used" hints into Pass 2 prompts.

# Regex to extract function/method call sites from raw source code.
# Matches: foo(, Foo::bar(, obj.method(, obj->method(, namespace::func(
_SOURCE_CALL_RE = _re_mod.compile(
    r'(?:(\w+)(?:::|\.|->))?(\w+)\s*\(',
)

# Names too generic to be useful — skip these in the frequency map
_CALL_NOISE = frozenset({
    # C/C++ stdlib & common
    "if", "for", "while", "switch", "return", "sizeof", "typeof", "decltype",
    "static_cast", "dynamic_cast", "reinterpret_cast", "const_cast",
    "new", "delete", "throw", "catch", "try",
    "get", "set", "put", "add", "end", "begin", "size", "push", "pop",
    "find", "erase", "clear", "empty", "front", "back", "insert", "remove",
    "open", "close", "read", "write", "lock", "unlock",
    "printf", "sprintf", "fprintf", "snprintf", "scanf", "sscanf",
    "malloc", "calloc", "realloc", "free",
    "memcpy", "memset", "memmove", "memcmp",
    "strlen", "strcmp", "strncmp", "strcpy", "strncpy", "strcat",
    "assert", "abort", "exit",
    "make_shared", "make_unique", "make_pair", "make_tuple",
    "move", "forward", "swap",
    # Python common
    "len", "str", "int", "float", "bool", "list", "dict", "tuple",
    "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "print", "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "super", "property", "classmethod", "staticmethod",
    "append", "extend", "update", "items", "keys", "values", "join", "split",
    "strip", "replace", "format", "encode", "decode",
    # JS/TS common
    "log", "error", "warn", "info", "debug",
    "then", "catch", "finally", "resolve", "reject",
    "push", "splice", "slice", "concat", "forEach", "map", "filter", "reduce",
    "toString", "valueOf", "constructor",
    # Macros / preprocessor
    "define", "undef", "include", "ifdef", "ifndef", "endif", "elif",
    "LOG", "ASSERT", "CHECK", "DCHECK", "VLOG", "TRACE",
})

# Minimum name length and minimum call count to qualify as "frequently called"
_MIN_CALL_NAME_LEN = 4
_FREQUENT_CALL_THRESHOLD = 5  # called from at least this many distinct files


def build_caller_frequency_map(files: list[Path],
                                codebases: Optional[list[Path]] = None,
                                codebase: Optional[Path] = None,
                                ) -> dict[str, int]:
    """Scan all source files once and count how many distinct files call each function.

    Returns a dict mapping "Class::method" or "function" → number of distinct
    files that contain a call to it. Only includes names that appear in
    >= _FREQUENT_CALL_THRESHOLD distinct files and aren't noise.
    """
    # symbol → set of files that call it
    call_sites: dict[str, set[str]] = {}
    all_cb = codebases if codebases else ([codebase] if codebase else [])

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue

        # Compute a short file key for dedup
        if all_cb:
            fkey = _relative_to_any(f, all_cb)
        else:
            fkey = f.name

        seen_in_file: set[str] = set()
        for match in _SOURCE_CALL_RE.finditer(text):
            qualifier = match.group(1)  # Class/namespace or None
            name = match.group(2)

            if len(name) < _MIN_CALL_NAME_LEN:
                continue
            if name in _CALL_NOISE or name.upper() == name:
                # Skip all-caps (likely macros)
                continue

            # Build the symbol key
            if qualifier and len(qualifier) >= 2:
                symbol = f"{qualifier}::{name}"
            else:
                symbol = name

            if symbol not in seen_in_file:
                seen_in_file.add(symbol)
                call_sites.setdefault(symbol, set()).add(fkey)

    # Filter to frequently called only
    return {sym: len(callers) for sym, callers in call_sites.items()
            if len(callers) >= _FREQUENT_CALL_THRESHOLD}


def _get_chunk_functions(raw_code: str) -> list[str]:
    """Extract function/method names defined in a code chunk.

    Returns a list of names like "ClassName::methodName" or "functionName".
    Handles C/C++, Python, JS/TS, Java, Go, Rust.
    """
    names: list[str] = []
    for line in raw_code.split("\n"):
        stripped = line.strip()
        # C/C++: ReturnType ClassName::methodName(
        m = _re_mod.match(r'(?:\w[\w:<>*&\s]*?\s+)?(\w+)::(\w+)\s*\(', stripped)
        if m:
            names.append(f"{m.group(1)}::{m.group(2)}")
            continue
        # C/C++ free function: ReturnType functionName(
        m = _re_mod.match(r'(?:\w[\w:<>*&\s]*?\s+)(\w{4,})\s*\([^)]*\)\s*(?:\{|const|override|=)',
                          stripped)
        if m and not m.group(1) in ("if", "for", "while", "switch", "return", "class", "struct"):
            names.append(m.group(1))
            continue
        # Python: def functionName(
        m = _re_mod.match(r'def\s+(\w{4,})\s*\(', stripped)
        if m:
            names.append(m.group(1))
            continue
        # JS/TS: function name( or name( {  or name = function/arrow
        m = _re_mod.match(r'(?:export\s+)?(?:async\s+)?function\s+(\w{4,})\s*\(', stripped)
        if m:
            names.append(m.group(1))
            continue
    return names


def _lookup_doc_mentions(symbols: list[str]) -> dict[str, str]:
    """Query R2R for documentation that mentions frequently-called functions.

    For each symbol, searches R2R for doc-type chunks containing that name.
    Returns a dict mapping symbol → doc snippet (first match, truncated).
    Degrades gracefully if R2R is unavailable.
    """
    if not symbols:
        return {}
    try:
        client = _r2r_client()
    except Exception:
        return {}

    doc_mentions: dict[str, str] = {}
    for sym in symbols:
        # Search for the bare function name (most likely to match docs)
        bare_name = sym.split("::")[-1] if "::" in sym else sym
        try:
            results = client.retrieval.search(
                query=bare_name,
                search_settings={
                    "limit": 2,
                    "filters": {"chunk_type": {"$in": ["doc_summary", "doc_section"]}},
                },
            )
            hits = results.results.chunk_search_results
            if not hits:
                continue
            # Check that the hit actually mentions the function name
            for hit in hits:
                text = (hit.text or "").strip()
                if bare_name.lower() in text.lower():
                    src = (hit.metadata or {}).get("source_file", "docs")
                    # Extract the sentence containing the function name
                    snippet = _extract_mention_sentence(text, bare_name)
                    doc_mentions[sym] = f'[{src}] {snippet}'
                    break
        except Exception:
            continue
    return doc_mentions


def _extract_mention_sentence(text: str, name: str) -> str:
    """Extract the sentence(s) from text that mention the given name."""
    # Split into sentences (rough)
    import re as _re_sent
    sentences = _re_sent.split(r'(?<=[.!?])\s+', text)
    name_lower = name.lower()
    matching = [s for s in sentences if name_lower in s.lower()]
    if matching:
        # Return first matching sentence, truncated
        result = matching[0].strip()
        if len(result) > 200:
            result = result[:200] + "..."
        return result
    # Fallback: return first 150 chars
    return text[:150] + ("..." if len(text) > 150 else "")


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


def _resolve_file(fname: str, codebase: Path,
                  codebases: Optional[list[Path]] = None) -> Optional[Path]:
    """Resolve a relative filename across one or more codebase roots.

    Tries direct join with each codebase, then falls back to _find_file.
    Returns the absolute path or None.
    """
    roots = codebases if codebases else [codebase]
    # Direct path match
    for cb in roots:
        candidate = cb / fname
        if candidate.exists() and candidate.is_file():
            return candidate
    # Filename-only fallback via cached index
    basename = Path(fname).name
    for cb in roots:
        match = _find_file(cb, basename)
        if match:
            return match
    return None


def _relative_to_any(f: Path, codebases: list[Path]) -> str:
    """Compute relative path from whichever codebase contains this file."""
    for cb in codebases:
        try:
            return str(f.relative_to(cb))
        except ValueError:
            continue
    return f.name


# TokenTracker and llm_call imported from codebase_shared.utils

# ── Pydantic models (used to validate Pass 1 JSON output) ─────────────────────

class ModuleDefinition(BaseModel):
    name:        str
    description: str
    files:       list[str]
    questions:   list[str]
    dir_paths:   list[str] = []   # original dir_paths from define_modules (for focused refinement)

class ModuleMap(BaseModel):
    project:      str
    description:  str
    modules:      list[ModuleDefinition]
    review_issues: list[dict] = []  # populated by review_module_map(); persisted in module_map.json

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


def detect_language(codebase: Path,
                    codebases: Optional[list[Path]] = None) -> Optional[str]:
    """Auto-detect the dominant language by counting file extensions.

    Scans the top 3 directory levels of each codebase (fast) and returns
    the language with the most files, or None if no known language found.
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

    roots = codebases if codebases else [codebase]
    for root in roots:
        _walk(root, 0)
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
                         manifest_path: Path,
                         codebases: Optional[list[Path]] = None,
                         ) -> tuple[list[Path], dict[str, str]]:
    """
    Compare current file hashes against a saved manifest.
    Returns (changed_files, new_manifest).
    Only returns files whose content has changed since the last run.
    """
    all_cb = codebases if codebases else [codebase]
    old_manifest = load_hash_manifest(manifest_path)
    new_manifest: dict[str, str] = {}
    changed: list[Path] = []

    for f in files:
        rel = _relative_to_any(f, all_cb)
        h = file_hash(f)
        new_manifest[rel] = h
        if h != old_manifest.get(rel, ""):
            changed.append(f)

    return changed, new_manifest


def build_directory_tree(codebase: Path, files: list[Path],
                         max_depth: int = 3,
                         expand_dirs: Optional[set[str]] = None,
                         codebases: Optional[list[Path]] = None) -> str:
    """Build a depth-limited annotated directory tree from file list.

    Shows directories up to max_depth. Unexpanded nodes get annotations like
    "(45 files, 3 subdirs)". Leaf directories list file counts.
    Use expand_dirs to selectively expand specific directories deeper.

    For small codebases (≤200 files), shows every file (original behaviour).

    When codebases is provided (multiple codebase paths), files are matched
    to their respective codebase for relative path computation.
    """
    # Helper to compute relative path when multiple codebases are possible
    all_codebases = codebases if codebases else [codebase]

    def _relative_to_any(f: Path) -> Path:
        """Compute relative path from whichever codebase contains this file."""
        for cb in all_codebases:
            try:
                return f.relative_to(cb)
            except ValueError:
                continue
        # Fallback: use filename only
        return Path(f.name)

    if len(all_codebases) > 1:
        root_label = " + ".join(cb.name for cb in all_codebases)
    else:
        root_label = codebase.name

    if len(files) <= 200:
        lines = [f"{root_label}/"]
        for rp in sorted(_relative_to_any(f) for f in files):
            indent = "  " * (len(rp.parts) - 1)
            lines.append(f"{indent}  {rp.name}")
        return "\n".join(lines)

    expand_dirs = expand_dirs or set()

    # Build a tree structure: {dir_path: {files: int, subdirs: set}}
    from collections import defaultdict
    dir_files: dict[str, int] = defaultdict(int)     # files directly in this dir
    all_dirs: set[str] = set()

    for f in files:
        rel = _relative_to_any(f)
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
    result = [f"{root_label}/  ({len(files)} files total)"]

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


def index_summaries_to_r2r(summaries: list[dict], index_workers: int = 8) -> None:
    """Index all summaries into R2R. Delegates to codebase_shared.r2r_indexer."""
    from codebase_shared.r2r_indexer import index_entries
    index_entries(summaries, index_workers=index_workers)


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
    "- MANDATORY: You must explore EVERY major directory (those with many files) "
    "using expand_dirs or list_files before calling define_modules. The system "
    "tracks which directories you've visited and will REJECT define_modules if "
    "major directories are unexplored.\n"
    "- Read at least one key file per major area to understand its purpose.\n"
    "- A large codebase typically has 5-20+ modules — if you only see 2-3, "
    "you haven't explored enough. Keep exploring.\n"
    "- Each module should map to a directory or group of related directories\n"
    "- Every source file should belong to exactly one module\n"
    "- NO catch-all modules. Do NOT create a single 'other' or 'libs' module that "
    "contains most of the codebase. Every major directory should be its own module "
    "or grouped with closely related directories only.\n"
    "- Write 3-6 domain-specific questions per module (use the project's vocabulary)\n"
    "- Focus questions on HOW things work internally, not just WHAT they are\n"
    "- Mark test modules with is_test: true (unit tests, test frameworks, mocks, "
    "benchmarks, fixtures). These will be excluded from the knowledge base.\n"
    "- If define_modules is rejected, read the feedback carefully and fix every "
    "issue mentioned before trying again."
)

MAX_EXPLORE_ROUNDS = 30  # cap on interactive exploration rounds
MAX_FILES_PER_READ = 8   # max files per read_files call


def _group_files_by_dir(codebase: Path, files: list[Path],
                         dir_path: str, codebases: Optional[list[Path]] = None) -> list[Path]:
    """Return files that live under a specific directory path."""
    all_codebases = codebases if codebases else [codebase]

    def _relative_to_any(f: Path) -> str:
        for cb in all_codebases:
            try:
                return str(f.relative_to(cb))
            except ValueError:
                continue
        return f.name

    result = []
    for f in files:
        rel = _relative_to_any(f)
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


def _validate_modules(modules: list[dict], total_files: int,
                      top_dir_count: int) -> Optional[str]:
    """Validate a define_modules result. Returns error message or None if OK."""
    if not modules:
        return "You defined 0 modules. Every codebase has at least a few modules."

    # Expected minimum: rough heuristic based on codebase size
    min_modules = max(3, min(top_dir_count // 2, 8))
    non_test = [m for m in modules if not m.get("is_test", False)]

    if len(non_test) < min_modules and total_files > 500:
        return (
            f"You only defined {len(non_test)} non-test modules, but this "
            f"codebase has {total_files} files across {top_dir_count} top-level "
            f"directories. A project this size should have at least {min_modules} "
            f"modules. Explore more directories and break it down further."
        )

    # Reject catch-all: any single module covering >40% of files
    for m in non_test:
        dir_paths = m.get("dir_paths", [])
        # Count how many top-level dirs this module claims
        claimed_top = set()
        for dp in dir_paths:
            top = dp.split("/")[0] if "/" in dp else dp
            claimed_top.add(top)
        if len(claimed_top) > top_dir_count * 0.4 and top_dir_count > 4:
            return (
                f"Module '{m.get('name')}' spans {len(claimed_top)} of "
                f"{top_dir_count} top-level directories — that's a catch-all, "
                f"not a real module. Break it into separate modules by functional "
                f"area. Each module should cover one cohesive subsystem."
            )

    return None  # OK


def _make_pass1_dispatch(codebase: Path, files: list[Path], language: str,
                         top_dir_count: int,
                         codebases: Optional[list[Path]] = None,
                         top_dirs: Optional[dict[str, int]] = None,
                         required_dirs: Optional[set[str]] = None):
    """Create a closure that dispatches Pass 1 tool calls.

    define_modules is validated here. If validation fails, an error message
    is returned to the LLM so it can fix and retry. If validation passes,
    the accepted result is stored in `dispatch.accepted_modules`.

    Coverage gate: before validation, checks that all required directories have
    been touched via expand_dirs, list_files, or read_files. If not, returns a
    specific list of unexplored directories. The coverage gate fires at most once.

    required_dirs: explicit set of top-level dirs that must be explored. When
    provided (e.g. for focused refinement rounds), overrides the default threshold
    (>= 30 files). Pass an empty set to disable the coverage gate entirely.
    """
    # Accumulate all expanded dirs across rounds so the tree keeps growing
    _all_expanded: set[str] = set()
    # Track which top-level dirs have been touched by any exploration tool
    _explored_top_dirs: set[str] = set()
    # Dirs that must be explored before define_modules is accepted.
    # required_dirs overrides the threshold-based computation when provided.
    if required_dirs is not None:
        _major_top_dirs: set[str] = required_dirs
    else:
        _major_top_dirs = {
            d for d, count in (top_dirs or {}).items() if count >= 30
        }
    # Use same depth as initial tree
    _base_depth = 3 if len(files) < 500 else 4 if len(files) < 5000 else 5
    all_codebases = codebases if codebases else [codebase]

    def _relative_to_any(f: Path) -> Path:
        """Compute relative path from whichever codebase contains this file."""
        for cb in all_codebases:
            try:
                return f.relative_to(cb)
            except ValueError:
                continue
        return Path(f.name)

    def _mark_explored(path: str):
        """Mark a path's top-level directory as touched."""
        top = path.split("/")[0] if "/" in path else path
        if top and top != ".":
            _explored_top_dirs.add(top)

    def dispatch(name: str, args: dict) -> str:
        if name == "expand_dirs":
            dirs = args.get("dirs", [])
            _all_expanded.update(str(d) for d in dirs)
            for d in dirs:
                _mark_explored(str(d))
            tree = build_directory_tree(codebase, files, max_depth=_base_depth,
                                         expand_dirs=_all_expanded, codebases=all_codebases)
            return tree

        elif name == "list_files":
            dir_path = args.get("dir_path", ".")
            _mark_explored(dir_path)
            matches = _group_files_by_dir(codebase, files, dir_path, codebases=all_codebases)
            lines = [str(_relative_to_any(f)) for f in matches[:80]]
            if len(matches) > 80:
                lines.append(f"... and {len(matches) - 80} more files")
            return "\n".join(lines) if lines else f"(no source files found in {dir_path})"

        elif name == "read_files":
            paths = args.get("paths", [])
            for p in paths:
                _mark_explored(str(p))
            parts = []
            for rf in paths[:MAX_FILES_PER_READ]:
                # Try each codebase to find the file
                rf_path = None
                for cb in all_codebases:
                    candidate = cb / str(rf)
                    if candidate.exists() and candidate.is_file():
                        rf_path = candidate
                        break
                if rf_path:
                    content = read_file_sample(rf_path, max_lines=SAMPLE_LINES)
                    parts.append(f"=== {rf} ===\n{content}")
                else:
                    # Try cached file index fallback across all codebases
                    found = None
                    for cb in all_codebases:
                        match = _find_file(cb, Path(rf).name)
                        if match:
                            found = match
                            break
                    if found:
                        content = read_file_sample(found, max_lines=SAMPLE_LINES)
                        actual = str(_relative_to_any(found))
                        parts.append(f"=== {actual} (matched from {rf}) ===\n{content}")
                    else:
                        parts.append(f"=== {rf} ===\n(file not found)")
            return "\n\n".join(parts) if parts else "(no files specified)"

        elif name == "search_kb":
            query = args.get("query", "")
            limit = args.get("limit", 3)
            result = search_kb(query, limit=limit)
            return result if result else "(no results found in knowledge base)"

        elif name == "define_modules":
            modules = args.get("modules", [])

            # Coverage gate: enforce exploration of major directories.
            # Fires at most once — after that, we skip it to avoid infinite loops.
            if not dispatch.coverage_gate_fired and _major_top_dirs:
                unexplored = _major_top_dirs - _explored_top_dirs
                if unexplored:
                    dispatch.coverage_gate_fired = True
                    unexplored_list = sorted(unexplored)
                    explored_count = len(_major_top_dirs - unexplored)
                    print(f"  [Pass 1] Coverage gate: {explored_count}/{len(_major_top_dirs)} "
                          f"major dirs explored, {len(unexplored)} missing")
                    return (
                        f"REJECTED: You have not explored these major directories yet:\n"
                        f"  {chr(10).join('  - ' + d for d in unexplored_list)}\n\n"
                        f"Use expand_dirs or list_files on each of them before calling "
                        f"define_modules. You have explored "
                        f"{explored_count}/{len(_major_top_dirs)} major directories so far.\n\n"
                        f"Major directories are those with 30+ source files. Each one likely "
                        f"contains a distinct module or sub-system."
                    )

            error = _validate_modules(modules, len(files), top_dir_count)
            if error:
                print(f"  [Pass 1] Module map REJECTED: {error[:120]}")
                dispatch.rejected_count = getattr(dispatch, "rejected_count", 0) + 1
                # After 2 rejections, accept whatever we get
                if dispatch.rejected_count >= 2:
                    print(f"  [Pass 1] Accepting after {dispatch.rejected_count} rejections")
                    dispatch.accepted_modules = args
                    return "__ACCEPTED__"
                return (
                    f"REJECTED: {error}\n\n"
                    "Use expand_dirs and list_files to explore the directories "
                    "you haven't looked at yet, then call define_modules again "
                    "with a more granular module breakdown."
                )
            dispatch.accepted_modules = args
            return "__ACCEPTED__"

        else:
            return f"Unknown tool: {name}"

    dispatch.accepted_modules = None
    dispatch.rejected_count = 0
    dispatch.coverage_gate_fired = False
    return dispatch


def run_pass1(model: str, codebase: Path, files: list[Path],
              language: str, docs_context: Optional[str] = None,
              tracker: Optional[TokenTracker] = None,
              codebases: Optional[list[Path]] = None) -> ModuleMap:
    """Agent-based module discovery using tool calling.

    The LLM explores the codebase interactively using tools:
    - expand_dirs: see deeper directory structure
    - list_files: see files in a directory
    - read_files: read first 80 lines of specific files
    - search_kb: search existing knowledge base / design docs
    - define_modules: terminal tool — outputs the final module map

    Bounded to MAX_EXPLORE_ROUNDS to cap cost.
    """
    all_codebases = codebases if codebases else [codebase]

    def _relative_to_any(f: Path) -> Path:
        """Compute relative path from whichever codebase contains this file."""
        for cb in all_codebases:
            try:
                return f.relative_to(cb)
            except ValueError:
                continue
        return Path(f.name)

    print(f"\n[Pass 1] {len(files)} source files found.")

    # Build initial tree — deeper for larger codebases
    initial_depth = 3 if len(files) < 500 else 4 if len(files) < 5000 else 5
    tree = build_directory_tree(codebase, files, max_depth=initial_depth, codebases=all_codebases)
    tree_lines = len(tree.split("\n"))
    print(f"[Pass 1] Directory tree: {tree_lines} lines, {len(tree)} chars")

    # Build initial prompt
    docs_section = ""
    if docs_context:
        docs_section = f"\n\nDesign documentation (excerpts):\n{docs_context}"

    # Count top-level dirs and file counts per dir for expectations
    top_dirs: dict[str, int] = {}  # dir_name → file_count
    for f in files:
        rel = _relative_to_any(f)
        if len(rel.parts) > 1:
            top = rel.parts[0]
            top_dirs[top] = top_dirs.get(top, 0) + 1

    # Build a compact summary of top-level directories for the LLM
    dir_summary_lines = []
    for d, count in sorted(top_dirs.items(), key=lambda x: -x[1]):
        dir_summary_lines.append(f"  {d}/ — {count} files")
    dir_summary = "\n".join(dir_summary_lines)

    min_modules = max(3, min(len(top_dirs) // 2, 8))
    # List the major dirs so the LLM knows exactly what it must explore
    major_dirs = sorted(d for d, count in top_dirs.items() if count >= 30)
    major_dirs_str = "\n".join(f"  - {d}/ ({top_dirs[d]} files)" for d in major_dirs)
    major_section = (
        f"\nYou MUST explore all of these major directories before calling "
        f"define_modules (the system will reject define_modules otherwise):\n"
        f"{major_dirs_str}\n"
    ) if major_dirs else ""

    initial_prompt = (
        f"Explore this {language} codebase and define its modules.\n\n"
        f"Top-level directories (by file count):\n{dir_summary}\n\n"
        f"Directory tree (depth-limited, counts on truncated nodes):\n"
        f"```\n{tree}\n```"
        f"{docs_section}\n"
        f"{major_section}\n"
        f"This codebase has {len(files)} files across {len(top_dirs)} "
        f"top-level directories. Each major directory above is likely its own "
        f"module or contains multiple sub-modules.\n\n"
        f"IMPORTANT RULES:\n"
        f"- Explore broadly first — expand every major directory, read key "
        f"files in each area. You have {MAX_EXPLORE_ROUNDS} rounds.\n"
        f"- Define at least {min_modules} modules. A project this size likely "
        f"has {min_modules}-{min_modules * 3}+ distinct modules.\n"
        f"- Do NOT create catch-all modules. A module named 'libs' or 'other' "
        f"that contains most of the codebase will be REJECTED.\n"
        f"- Each module should cover ONE cohesive subsystem.\n"
        f"- If define_modules is rejected, read the feedback and fix every issue."
    )

    # Create dispatcher — define_modules is validated inside dispatch
    n_top_dirs = len(top_dirs)
    dispatch = _make_pass1_dispatch(codebase, files, language, n_top_dirs,
                                    codebases=all_codebases, top_dirs=top_dirs)

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
            status = "validating" if dispatch.accepted_modules is None else "accepted"
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Defining {n} modules ({status})")

    print(f"[Pass 1] Exploring codebase structure with {model}...")

    # define_modules is NOT a terminal tool — it's validated inside dispatch.
    # The loop runs until dispatch.accepted_modules is set or rounds exhausted.
    conversation, _ = llm_tool_loop(
        model=model,
        system=PASS1_SYSTEM,
        initial_messages=[{"role": "user", "content": initial_prompt}],
        tools=PASS1_TOOLS,
        dispatch=dispatch,
        terminal_tools=set(),  # no terminal tools — validation is in dispatch
        max_rounds=MAX_EXPLORE_ROUNDS,
        max_tokens=4096,
        tracker=tracker,
        phase="Pass 1",
        on_round=on_round,
    )

    terminal_args = dispatch.accepted_modules

    # If agent never called define_modules successfully, force it
    if terminal_args is None:
        print(f"  [Pass 1] Max rounds reached — forcing module definition...")
        from litellm import completion
        conversation.append({
            "role": "user",
            "content": (
                "You have used all exploration rounds. "
                "Call define_modules now to define the final module map. "
                f"This codebase has {n_top_dirs} top-level directories — "
                f"define at least {min_modules} modules. "
                "Assign ALL directories to modules. "
                "Do NOT put everything in one module."
            ),
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
        terminal_args = _fallback_module_map(codebase, files, language,
                                                codebases=all_codebases)

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
                for f in _group_files_by_dir(codebase, files, str(dp), codebases=all_codebases):
                    assigned_files.add(str(_relative_to_any(f)))
            continue

        mod_files: list[str] = []
        for dp in dir_paths:
            for f in _group_files_by_dir(codebase, files, str(dp), codebases=all_codebases):
                rel = str(_relative_to_any(f))
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
            dir_paths=[str(dp) for dp in dir_paths],
        ))

    if test_modules_skipped:
        print(f"[Pass 1] Skipped {test_modules_skipped} test module(s)")

    # Catch orphans
    orphaned = [str(_relative_to_any(f)) for f in files
                if str(_relative_to_any(f)) not in assigned_files]
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

    # ── Review: advisory quality check ────────────────────────────────────────
    print(f"\n[Pass 1 review] Checking module map quality...")
    issues, overall = review_module_map(model, module_map, len(files), tracker=tracker)
    print(f"[Pass 1 review] {overall}")
    errors = [i for i in issues if i.get("severity") == "error"]
    warns  = [i for i in issues if i.get("severity") == "warn"]
    for i in warns:
        print(f"  {yellow('[warn]')} {i['module']}: {i['description']}")
    for i in errors:
        print(f"  {red('[error]')} {i['module']}: {i['description']}")
    if errors:
        print(f"\n  {bold(str(len(errors)))} issue(s) saved to module_map.json. "
              f"Re-run with --discover to fix them.")
    module_map.review_issues = issues

    return module_map


# ── Pass 1: Review & Focused Refinement ───────────────────────────────────────

PASS1_REVIEW_SYSTEM = (
    "You are reviewing a codebase module map for quality. "
    "Output ONLY valid JSON — no markdown, no text outside the JSON."
)

PASS1_FOCUSED_SYSTEM = (
    "You are a senior software architect refining a codebase module map. "
    "A reviewer identified problems with specific modules. The current module map "
    "is shown below. Use the exploration tools to investigate the problematic "
    "areas, then call define_modules with the complete revised map.\n\n"
    "Rules:\n"
    "- Keep well-structured modules exactly as they are.\n"
    "- Focus your exploration on the directories of the flagged modules.\n"
    "- Fix each flagged issue: split too-large modules, improve generic questions, "
    "or separate unrelated directories.\n"
    "- The output must include ALL modules (revised + unchanged).\n"
    "- If define_modules is rejected, read the feedback and fix every issue."
)


def review_module_map(
    model: str,
    module_map: "ModuleMap",
    total_files: int,
    tracker: Optional[TokenTracker] = None,
) -> tuple[list[dict], str]:
    """Single LLM call to review the module map quality.

    Returns (issues, overall_comment).
    issues: list of {"module", "issue_type", "description", "severity"}
      severity: "error" → re-run with --discover to fix
                "warn"  → advisory, noted but not blocking

    issue_type values:
      too_large      — module covers >35% of files and contains separable subsystems
      poor_questions — questions are generic instead of domain-specific
      bad_split      — module combines unrelated concerns
    """
    lines = []
    for m in module_map.modules:
        pct = len(m.files) / total_files * 100 if total_files else 0
        sample = ", ".join(m.files[:8])
        lines.append(
            f"Module: {m.name} ({len(m.files)} files, {pct:.0f}% of codebase)\n"
            f"  Dirs: {', '.join(m.dir_paths) or '(unknown)'}\n"
            f"  Sample files: {sample}\n"
            f"  Questions: {'; '.join(m.questions[:4])}"
        )

    prompt = (
        f"Review this module map ({total_files} source files total).\n\n"
        + "\n\n".join(lines)
        + f"""

Identify quality problems. Output JSON:
{{
  "issues": [
    {{
      "module": "name",
      "issue_type": "too_large|poor_questions|bad_split",
      "description": "specific problem and concrete suggestion (name dirs to split, example good questions)",
      "severity": "error|warn"
    }}
  ],
  "overall": "1-2 sentence overall assessment"
}}

Flag as error (re-run needed):
- too_large: module covers >35% of all files AND file samples show clearly separable subsystems
- bad_split: module mixes dirs with unrelated purposes visible from sample files

Flag as warn:
- poor_questions: questions use generic phrasing ("What does X do?") instead of domain vocabulary

Only flag real problems. Return empty issues list if the map is reasonable."""
    )

    try:
        raw = llm_call(model, PASS1_REVIEW_SYSTEM, prompt,
                       max_tokens=800, json_mode=True,
                       tracker=tracker, phase="Pass 1 review")
        data = json.loads(raw)
        return data.get("issues", []), data.get("overall", "")
    except Exception as e:
        print(f"  [Pass 1 review] Error: {e}")
        return [], "Review failed — skipping."


def run_pass1_focused(
    model: str,
    module_map: "ModuleMap",
    codebase: Path,
    files: list[Path],
    language: str,
    tracker: Optional[TokenTracker] = None,
    codebases: Optional[list[Path]] = None,
) -> "ModuleMap":
    """Targeted Pass 1 re-run that fixes modules flagged in module_map.review_issues.

    The LLM sees the full current module map plus the reviewer's feedback.
    It explores the problematic directories, then calls define_modules with
    a complete revised map (keeping good modules unchanged, fixing bad ones).
    The coverage gate is narrowed to only the flagged modules' directories.
    """
    all_codebases = codebases if codebases else [codebase]
    issues = module_map.review_issues
    error_issues = [i for i in issues if i.get("severity") == "error"]

    if not error_issues:
        print("[Pass 1 focused] No error-level issues to fix.")
        return module_map

    # Identify the flagged module names and their directories
    flagged_names = {i["module"] for i in error_issues}
    flagged_mods = {m.name: m for m in module_map.modules if m.name in flagged_names}

    # Compute required_dirs: top-level dirs of flagged modules that need exploration
    required_dirs: set[str] = set()
    for m in flagged_mods.values():
        for dp in m.dir_paths:
            top = dp.split("/")[0] if "/" in dp else dp
            if top:
                required_dirs.add(top)

    print(f"\n[Pass 1 focused] Fixing {len(error_issues)} flagged module(s): "
          f"{', '.join(sorted(flagged_names))}")
    print(f"[Pass 1 focused] Must explore: {', '.join(sorted(required_dirs))}")

    # Build the current module map summary for the prompt
    def _relative_to_any(f: Path) -> Path:
        for cb in all_codebases:
            try:
                return f.relative_to(cb)
            except ValueError:
                continue
        return Path(f.name)

    total_files = len(files)
    mod_lines = []
    for m in module_map.modules:
        pct = len(m.files) / total_files * 100 if total_files else 0
        flag = " ← FLAGGED" if m.name in flagged_names else ""
        mod_lines.append(
            f"  {m.name} ({len(m.files)} files, {pct:.0f}%){flag}\n"
            f"    Dirs: {', '.join(m.dir_paths) or '(unknown)'}\n"
            f"    Description: {m.description}"
        )

    issues_lines = []
    for i in error_issues:
        issues_lines.append(
            f"  [{i['issue_type']}] {i['module']}: {i['description']}"
        )

    # Build initial tree
    initial_depth = 3 if total_files < 500 else 4 if total_files < 5000 else 5
    tree = build_directory_tree(codebase, files, max_depth=initial_depth,
                                codebases=all_codebases)

    # Top-level dir counts (for validation)
    top_dirs: dict[str, int] = {}
    for f in files:
        rel = _relative_to_any(f)
        if len(rel.parts) > 1:
            top = rel.parts[0]
            top_dirs[top] = top_dirs.get(top, 0) + 1
    n_top_dirs = len(top_dirs)

    initial_prompt = (
        f"Refine this {language} codebase module map. "
        f"A reviewer flagged {len(error_issues)} module(s) as problematic.\n\n"
        f"CURRENT MODULE MAP:\n" + "\n".join(mod_lines) + "\n\n"
        f"REVIEWER ISSUES TO FIX:\n" + "\n".join(issues_lines) + "\n\n"
        f"Directory tree:\n```\n{tree}\n```\n\n"
        f"REQUIRED DIRECTORIES TO EXPLORE FIRST: {', '.join(sorted(required_dirs))}\n"
        f"(The system will reject define_modules until you explore these.)\n\n"
        f"After exploring, call define_modules with the COMPLETE revised map — "
        f"include all {len(module_map.modules)} modules (unchanged ones + fixed ones). "
        f"Keep good modules exactly as they are. Fix every flagged issue."
    )

    dispatch = _make_pass1_dispatch(
        codebase, files, language, n_top_dirs,
        codebases=all_codebases,
        top_dirs=top_dirs,
        required_dirs=required_dirs,
    )

    _explored_dirs: list[str] = []
    _files_read = 0

    def on_round(round_num: int, tool_name: str, args: dict):
        nonlocal _files_read
        if tool_name == "expand_dirs":
            dirs = args.get("dirs", [])
            _explored_dirs.extend(dirs)
            area = dirs[0] if dirs else "..."
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Exploring {area}/ "
                  f"({len(_explored_dirs)} dirs explored)")
        elif tool_name == "list_files":
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Listing {args.get('dir_path', '?')}/")
        elif tool_name == "read_files":
            paths = args.get("paths", [])
            _files_read += len(paths)
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Reading {len(paths)} files "
                  f"({_files_read} total)")
        elif tool_name == "define_modules":
            n = len(args.get("modules", []))
            status = "validating" if dispatch.accepted_modules is None else "accepted"
            print(f"  [{round_num}/{MAX_EXPLORE_ROUNDS}] Defining {n} modules ({status})")

    print(f"[Pass 1 focused] Exploring with {model}...")
    conversation, _ = llm_tool_loop(
        model=model,
        system=PASS1_FOCUSED_SYSTEM,
        initial_messages=[{"role": "user", "content": initial_prompt}],
        tools=PASS1_TOOLS,
        dispatch=dispatch,
        terminal_tools=set(),
        max_rounds=MAX_EXPLORE_ROUNDS,
        max_tokens=4096,
        tracker=tracker,
        phase="Pass 1 focused",
        on_round=on_round,
    )

    terminal_args = dispatch.accepted_modules

    if terminal_args is None:
        # Force define_modules if rounds exhausted
        print(f"  [Pass 1 focused] Max rounds reached — forcing module definition...")
        from litellm import completion
        conversation.append({
            "role": "user",
            "content": (
                "You have used all exploration rounds. Call define_modules now "
                "with the complete revised map. Include all modules."
            ),
        })
        response = completion(
            model=model, messages=conversation, max_tokens=4096,
            tools=PASS1_TOOLS,
            tool_choice={"type": "function", "function": {"name": "define_modules"}},
        )
        if tracker:
            tracker.record("Pass 1 focused (forced)", response)
        msg = response.choices[0].message
        if msg.tool_calls:
            try:
                terminal_args = json.loads(msg.tool_calls[0].function.arguments)
            except (json.JSONDecodeError, TypeError):
                terminal_args = None

    if terminal_args is None:
        print("  [Pass 1 focused] Failed — keeping original module map.")
        return module_map

    # Resolve the revised map using the same post-processing as run_pass1
    raw_modules = terminal_args.get("modules", [])
    print(f"\n[Pass 1 focused] Resolving {len(raw_modules)} modules to file lists...")

    assigned_files: set[str] = set()
    new_modules: list[ModuleDefinition] = []
    test_modules_skipped = 0

    for mod_raw in raw_modules:
        name = mod_raw.get("name", "unknown")
        desc = mod_raw.get("description", "")
        dir_paths = mod_raw.get("dir_paths", [])
        questions = mod_raw.get("questions", [])
        is_test = mod_raw.get("is_test", False)

        if is_test or _is_test_module(name, desc, dir_paths):
            test_modules_skipped += 1
            for dp in dir_paths:
                for f in _group_files_by_dir(codebase, files, str(dp),
                                              codebases=all_codebases):
                    assigned_files.add(str(_relative_to_any(f)))
            continue

        mod_files: list[str] = []
        for dp in dir_paths:
            for f in _group_files_by_dir(codebase, files, str(dp),
                                          codebases=all_codebases):
                rel = str(_relative_to_any(f))
                if rel not in assigned_files:
                    mod_files.append(rel)
                    assigned_files.add(rel)

        if not mod_files:
            continue

        if not questions:
            questions = [f"What does the {name} module do?",
                         f"How is {name} structured internally?"]

        new_modules.append(ModuleDefinition(
            name=name,
            description=desc,
            files=mod_files,
            questions=questions if isinstance(questions, list) else [str(questions)],
            dir_paths=[str(dp) for dp in dir_paths],
        ))

    if test_modules_skipped:
        print(f"[Pass 1 focused] Skipped {test_modules_skipped} test module(s)")

    # Orphaned files → other
    orphaned = [str(_relative_to_any(f)) for f in files
                if str(_relative_to_any(f)) not in assigned_files]
    if orphaned:
        print(f"[Pass 1 focused] {len(orphaned)} orphaned files → 'other'")
        new_modules.append(ModuleDefinition(
            name="other",
            description="Files not assigned to a specific module",
            files=orphaned,
            questions=["What do these miscellaneous files do?"],
        ))

    for m in new_modules:
        flag = " ← revised" if m.name in flagged_names else ""
        print(f"  - {m.name}: {len(m.files)} files{flag}")

    # Re-run advisory review on the refined map (issues cleared on success)
    refined_map = ModuleMap(
        project=module_map.project,
        description=terminal_args.get("description", module_map.description),
        modules=new_modules,
    )
    total_assigned = sum(len(m.files) for m in new_modules)
    print(f"\n[Pass 1 focused] {len(new_modules)} modules, "
          f"{total_assigned}/{len(files)} files assigned.")

    print(f"\n[Pass 1 review] Re-checking refined map...")
    issues, overall = review_module_map(model, refined_map, len(files), tracker=tracker)
    print(f"[Pass 1 review] {overall}")
    remaining_errors = [i for i in issues if i.get("severity") == "error"]
    for i in [x for x in issues if x.get("severity") == "warn"]:
        print(f"  {yellow('[warn]')} {i['module']}: {i['description']}")
    for i in remaining_errors:
        print(f"  {red('[error]')} {i['module']}: {i['description']}")
    if remaining_errors:
        print(f"  {len(remaining_errors)} issue(s) remain. Re-run --discover to continue refining.")
    else:
        print(f"  {green('Module map looks good.')} No more errors.")

    refined_map.review_issues = issues
    return refined_map


def _fallback_module_map(codebase: Path, files: list[Path],
                          language: str,
                          codebases: Optional[list[Path]] = None) -> dict:
    """Emergency fallback: one module per top-level directory."""
    all_cb = codebases if codebases else [codebase]
    groups: dict[str, list[str]] = {}
    for f in files:
        rel_str = _relative_to_any(f, all_cb)
        rel = Path(rel_str)
        bucket = rel.parts[0] if len(rel.parts) > 1 else "."
        groups.setdefault(bucket, []).append(rel_str)

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
    "Write plain prose — no markdown, no bullet points.\n\n"
    "You have access to a search_kb tool to query the knowledge base when you need "
    "to understand a concept, type, or module referenced in the code. Use it when "
    "you encounter unfamiliar types, function calls into other modules, or domain "
    "concepts that would make the summary richer if understood. For simple or "
    "self-contained chunks, go directly to write_summary."
)

PASS2_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "Search the knowledge base for context about a concept, type, "
                "module, or function referenced in the code chunk. Use this when "
                "you need to understand what an unfamiliar symbol does before "
                "writing the summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query, e.g. 'DeviceManager initialization contract' or 'what is TxnRequest'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_summary",
            "description": "Commit the final summary for this code chunk. Call this when you are ready to write the summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "2-4 sentence domain-aware summary of the chunk",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["algorithm", "contract", "glue", "error_handling",
                                 "data_model", "boilerplate"],
                        "description": "What kind of question this chunk answers",
                    },
                    "search_value": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Would a developer search for this?",
                    },
                },
                "required": ["summary", "category", "search_value"],
            },
        },
    },
]

MAX_PASS2_SEARCH_ROUNDS = 3  # max search_kb calls per chunk before forcing write_summary

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
                       file_summary: str = "",
                       usage_hints: Optional[list[str]] = None) -> str:
    questions_text = "\n".join(f"- {q}" for q in questions)

    # The file_summary field may contain:
    # - A plain file summary ("This file implements X")
    # - Function pre-pass context ("[Function: foo] ... Focus on: ...")
    # - Combined context from multiple sources
    context_section = ""
    focus_section = ""
    if file_summary:
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

    usage_section = ""
    if usage_hints:
        usage_section = (
            "\n\nUsage note — these functions defined in this chunk are widely "
            "called across the codebase:\n"
            + "\n".join(f"- {h}" for h in usage_hints)
            + "\n"
        )

    return f"""Project: {project_desc}
Module: {module_name} — {module_desc}

Domain questions this module answers:
{questions_text}
{context_section}
Source file: {source_file}
Code chunk:
```
{raw_code}
```
{usage_section}
If you encounter unfamiliar types, modules, or domain concepts referenced in this \
code, call search_kb to look them up before writing the summary.

When ready, call write_summary with:
- summary: 2-4 sentence domain-aware description
  - Use the project's own vocabulary (class names, function names, domain concepts)
  - State what this code DOES, not just what it IS
  - Mention which domain questions above this chunk addresses (if any)
  - Note what other functions/methods this code calls (if any)
  - If a function is widely called (see usage note above), include WHEN and WHY a developer would call it
  - If trivial (imports-only, constants, empty stub), one sentence is enough{focus_section}
- category: what kind of question this chunk answers
  - algorithm: core logic, state machines — "how does X work internally?"
  - contract: interface, API, protocol — "how do I use X?"
  - glue: wiring, delegation, config — "how are X and Y connected?"
  - error_handling: error paths, recovery — "what happens when X fails?"
  - data_model: type definitions, schemas — "what does X look like?"
  - boilerplate: getters, imports, logging — rarely searched for
- search_value: high (core logic/APIs), medium (supporting code), low (trivial/boilerplate)"""


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
    codebases: Optional[list[Path]] = None,
) -> dict[str, str]:
    """Generate 1-sentence summaries for each file before chunk-level Pass 2.

    Uses async concurrency for high throughput.
    Returns a dict mapping rel_path → file summary.
    """
    all_codebases = codebases if codebases else [codebase]

    # Collect unique files with their module context
    file_info: dict[str, tuple[str, str, str]] = {}  # rel_path → (fpath, mod_name, mod_desc)
    for mod in module_map.modules:
        for fname in mod.files:
            fpath = _resolve_file(fname, codebase, all_codebases)
            if not fpath:
                continue
            rel = _relative_to_any(fpath, all_codebases)
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
                try:
                    ref_rel = ref_path.relative_to(codebase)
                except ValueError:
                    ref_rel = ref_path.name
                parts.append(f"[{ref_rel}]\n{content}")
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


def _make_pass2_dispatch() -> tuple[dict, callable]:
    """Create a dispatch closure for the Pass 2 tool loop.

    Returns (state_dict, dispatch_fn).
    state_dict holds the terminal result once write_summary is called.
    """
    state: dict = {"result": None, "kb_hits": 0}

    def dispatch(name: str, args: dict) -> str:
        if name == "search_kb":
            query = args.get("query", "")
            state["kb_hits"] += 1
            result = search_kb(query, limit=3)
            return result if result else "(no results found)"

        elif name == "write_summary":
            state["result"] = {
                "summary":        args.get("summary", "").strip(),
                "chunk_category": args.get("category", "unknown"),
                "search_value":   args.get("search_value", "medium"),
            }
            return "__ACCEPTED__"

        return f"Unknown tool: {name}"

    return state, dispatch


async def _async_summarize_one_chunk(
        model: str, codebase: Path, project_desc: str,
        mod_name: str, mod_desc: str, questions: list[str],
        rel_path: str, chunk: str, chunk_index: int,
        file_summary: str = "",
        tracker: Optional[TokenTracker] = None,
        caller_freq: Optional[dict[str, int]] = None,
        doc_mentions: Optional[dict[str, str]] = None) -> dict:
    """Summarize a single code chunk using a tool loop.

    The LLM can call search_kb as many times as needed to look up unfamiliar
    types, modules, or domain concepts before committing the summary via
    write_summary. Simple chunks typically go straight to write_summary.
    """
    # Build usage hints for frequently-called functions in this chunk
    usage_hints: Optional[list[str]] = None
    if caller_freq:
        func_names = _get_chunk_functions(chunk)
        hints = []
        for fn in func_names:
            count = caller_freq.get(fn)
            matched_sym = fn
            if not count:
                bare = fn.split("::")[-1] if "::" in fn else None
                if bare:
                    count = caller_freq.get(bare)
                    if count:
                        matched_sym = bare
            if count:
                hint = f"{fn} is called from {count} different files"
                doc_snip = (doc_mentions or {}).get(matched_sym) or \
                           (doc_mentions or {}).get(fn)
                if doc_snip:
                    hint += f". Documentation says: {doc_snip}"
                hints.append(hint)
        if hints:
            usage_hints = hints

    prompt = build_pass2_prompt(
        project_desc=project_desc,
        module_name=mod_name,
        module_desc=mod_desc,
        questions=questions,
        source_file=rel_path,
        raw_code=chunk,
        file_summary=file_summary,
        usage_hints=usage_hints,
    )

    state, dispatch = _make_pass2_dispatch()

    # Run tool loop in a thread (llm_tool_loop is synchronous)
    def _run_tool_loop():
        return llm_tool_loop(
            model=model,
            system=PASS2_SYSTEM,
            initial_messages=[{"role": "user", "content": prompt}],
            tools=PASS2_TOOLS,
            dispatch=dispatch,
            terminal_tools={"write_summary"},
            max_rounds=MAX_PASS2_SEARCH_ROUNDS + 1,  # +1 for the write_summary call
            max_tokens=600,
            tracker=tracker,
            phase="Pass 2",
        )

    await asyncio.to_thread(_run_tool_loop)

    # If the LLM never called write_summary (exhausted rounds), force it
    if state["result"] is None:
        force_prompt = (
            "You have used all search rounds. Call write_summary now with your "
            "best summary of the code chunk based on what you've seen."
        )
        def _force():
            from litellm import completion
            msgs = [
                {"role": "user", "content": prompt},
                {"role": "user", "content": force_prompt},
            ]
            resp = completion(
                model=model, messages=msgs, max_tokens=600,
                tools=PASS2_TOOLS,
                tool_choice={"type": "function",
                             "function": {"name": "write_summary"}},
            )
            if tracker:
                tracker.record("Pass 2 (forced)", resp)
            msg = resp.choices[0].message
            if msg.tool_calls:
                try:
                    args = json.loads(msg.tool_calls[0].function.arguments)
                    dispatch("write_summary", args)
                except (json.JSONDecodeError, TypeError):
                    pass
        await asyncio.to_thread(_force)

    # Fallback if still nothing
    if state["result"] is None:
        state["result"] = {
            "summary": f"[{rel_path} chunk {chunk_index}]",
            "chunk_category": "unknown",
            "search_value": "low",
        }

    summary_text   = state["result"]["summary"]
    chunk_category = state["result"]["chunk_category"]
    search_value   = state["result"]["search_value"]

    # Determine skip_index
    skip_index = chunk_category == "boilerplate" and search_value == "low"

    # Reference resolution pass (for --refine; also catches vague markers)
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
        "kb_queries":     state["kb_hits"],
    }


async def run_pass2(model: str, codebase: Path, module_map: ModuleMap,
                    language: str, max_chunks: Optional[int] = None,
                    summaries_path: Optional[Path] = None,
                    tracker: Optional[TokenTracker] = None,
                    workers: int = 4, rpm: int = 60,
                    max_concurrent: int = 50,
                    quiet: bool = False, verbose: bool = False,
                    codebases: Optional[list[Path]] = None) -> list[dict]:
    """Chunk each file and summarize each chunk using a tool loop.

    The LLM can call search_kb during summarization to look up unfamiliar
    types or concepts. Simple chunks go straight to write_summary; complex
    ones that reference other modules will search first (up to
    MAX_PASS2_SEARCH_ROUNDS times per chunk).
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

    all_codebases = codebases if codebases else [codebase]

    project_desc = f"{module_map.project}: {module_map.description}"

    if cached_context.get("file_summaries"):
        file_summaries = cached_context["file_summaries"]
        print(f"[File summaries] Loaded {len(file_summaries)} from cache")
    else:
        file_summaries = await _generate_file_summaries(
            model, codebase, module_map,
            max_concurrent=max_concurrent, rpm=rpm, tracker=tracker, quiet=quiet,
            codebases=all_codebases,
        )

    # ── Build caller frequency map (single pass, no LLM) ─────────────────
    all_files_for_freq: list[Path] = []
    for mod in module_map.modules:
        for fname in mod.files:
            fpath = _resolve_file(fname, codebase, all_codebases)
            if fpath:
                all_files_for_freq.append(fpath)

    import time as _time_freq
    t0 = _time_freq.monotonic()
    caller_freq = build_caller_frequency_map(
        all_files_for_freq, codebases=all_codebases, codebase=codebase)
    elapsed = _time_freq.monotonic() - t0
    # Look up doc mentions for frequently-called functions
    doc_mentions: dict[str, str] = {}
    if caller_freq:
        print(f"\n[Caller frequency] {len(caller_freq)} frequently-called functions "
              f"detected ({elapsed:.1f}s)")
        # Show top 5 for visibility
        top5 = sorted(caller_freq.items(), key=lambda x: -x[1])[:5]
        for sym, count in top5:
            print(f"  {sym}: called from {count} files")

        # Query R2R for doc mentions (only if RAG is available)
        if rag:
            doc_mentions = _lookup_doc_mentions(list(caller_freq.keys()))
            if doc_mentions:
                print(f"[Caller frequency] Found doc mentions for "
                      f"{len(doc_mentions)} functions")
    else:
        print(f"\n[Caller frequency] No frequently-called functions detected ({elapsed:.1f}s)")

    # ── Collect chunks and detect multi-chunk functions ──────────────────
    work_items: list[dict] = []
    skipped = 0
    # Collect per-file chunk data for function grouping
    per_file_chunks: list[tuple[str, list[str], "ModuleDefinition"]] = []

    print(f"\n[Pass 2] Collecting chunks from {len(module_map.modules)} modules...")

    for mod in module_map.modules:
        for fname in mod.files:
            fpath = _resolve_file(fname, codebase, all_codebases)
            if not fpath:
                skipped += 1
                continue

            chunks = chunk_file(fpath, language=language)
            if not chunks:
                skipped += 1
                continue

            rel_path = _relative_to_any(fpath, all_codebases)
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
    kb_query_count = 0
    error_count = 0
    # Lock protects summaries list + counters against concurrent on_result calls
    _result_lock = asyncio.Lock()

    async def on_result(idx, entry):
        nonlocal completed_count, refined_count, kb_query_count
        async with _result_lock:
            ref_tag = ""
            if entry.pop("refined", False):
                ref_tag = " [refined]"
                refined_count += 1

            kb_hits = entry.pop("kb_queries", 0)
            kb_query_count += kb_hits
            kb_tag = f" [{kb_hits}q]" if kb_hits else ""

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
                      f"{len(entry['summary'])} chars{ref_tag}{kb_tag}{eta}")

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
            file_summary=it.get("file_summary", ""),
            tracker=tracker,
            caller_freq=caller_freq,
            doc_mentions=doc_mentions,
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

    kb_note = f", {kb_query_count} KB queries" if kb_query_count else ""
    print(f"\n[Pass 2] Done: {len(summaries)} summaries total "
          f"({completed_count} new, {total_cached} cached, "
          f"{error_count} errors, {refined_count} refined, "
          f"{overview_count} overviews, {func_card_count} function cards{kb_note})")
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
    parser.add_argument("--codebase", nargs="+", required=True,
                        help="Root directory(ies) of the codebase to analyze. "
                             "Multiple paths supported: --codebase /path/a /path/b")
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
                             "Can specify multiple: --docs /path/a /path/b")
    parser.add_argument("--discover", action="store_true",
                        help="Only run module discovery (Pass 1). "
                             "Produces module_map.json. If module_map.json already has "
                             "review errors, runs a focused refinement round instead. "
                             "Run --summarize next.")
    parser.add_argument("--summarize", action="store_true",
                        help="Only run summarization (Pass 2). Skips module discovery. "
                             "Requires an existing module_map.json from a previous --discover run.")
    parser.add_argument("--improve", action="store_true",
                        help="Re-read existing summaries.json and rewrite weak ones using KB "
                             "context. Cheaper than re-running the full pipeline. "
                             "Requires existing summaries.json.")
    parser.add_argument("--reindex", action="store_true",
                        help="Re-index existing summaries.json into R2R without re-summarizing. "
                             "Useful after editing summaries manually, changing models, or "
                             "after clearing R2R.")
    parser.add_argument("--purge-code", action="store_true",
                        help="Delete only the code chunks (source_type=code) from R2R using "
                             "doc_ids stored in summaries.json. Leaves docs and tickets intact. "
                             "Run before re-running study_agent to avoid duplicates.")
    parser.add_argument("--refine", action="store_true",
                        help="After summarization, re-run LLM on summaries that contain "
                             "vague references ('delegates to', 'defined elsewhere', etc.). "
                             "Much cheaper than --improve: only touches ~5%% of summaries.")
    parser.add_argument("--incremental", action="store_true",
                        help="Only re-summarize files whose content has changed since "
                             "the last run. Uses a hash manifest (file_hashes.json).")
    parser.add_argument("--exclude", nargs="+", default=None,
                        help="Additional directories to skip. Can specify multiple: "
                             "--exclude generated proto_out experimental")
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

    # Resolve multiple --codebase paths
    codebases = [Path(p).resolve() for p in args.codebase]
    for cb in codebases:
        if not cb.is_dir():
            print(f"Error: --codebase '{cb}' is not a directory")
            sys.exit(1)
    # Primary codebase is the first one (used for relative paths and output naming)
    codebase = codebases[0]

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
        detected = detect_language(codebase, codebases=codebases)
        if detected:
            args.language = detected
        else:
            print("Could not auto-detect language. Use --language to specify.")
            sys.exit(1)

    # ── Collect source files from all codebase paths ─────────────────────────
    files: list[Path] = []
    for cb in codebases:
        cb_files = collect_source_files(cb, language=args.language,
                                        include_tests=False)
        files.extend(cb_files)
    skip_note = " (excluding tests)"
    if len(codebases) > 1:
        print(f"Found {len(files)} {args.language} source files across {len(codebases)} codebases{skip_note}")
        for cb in codebases:
            n = sum(1 for f in files if str(f).startswith(str(cb)))
            print(f"  {cb}: {n} files")
    else:
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
    file_summary_tokens = len(files) * 500
    func_prepass_tokens = int(est_chunks * 0.2 * 600)
    func_card_tokens    = int(est_chunks * 0.15 * 1200)
    class_summary_tokens = int(len(files) / 5 * 500)
    pass1_tokens = 50000
    pass2_tokens = (est_prompt_tokens + est_completion_tokens
                    + file_summary_tokens + func_prepass_tokens
                    + func_card_tokens + class_summary_tokens)

    # Only count the passes that will actually run
    if args.discover:
        est_total = pass1_tokens
    elif args.summarize:
        est_total = pass2_tokens
    else:
        est_total = pass1_tokens + pass2_tokens
    # Try to get real pricing from litellm's model database
    rate_per_mtok = None
    _pricing_source = None
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
    est_cost = est_total / 1_000_000 * rate_per_mtok if rate_per_mtok else None

    total_calls = 1 if args.discover else est_chunks   # Pass 1 is one tool-loop, not per-chunk
    est_minutes = total_calls / args.rpm

    if args.dry_run:
        mode = ("Pass 1 only (--discover)" if args.discover
                else "Pass 2 only (--summarize)" if args.summarize
                else "Full pipeline (Pass 1 + Pass 2)")
        print(f"\n{'='*60}")
        print(f"  DRY RUN — Cost Estimate  [{mode}]")
        print(f"{'='*60}")
        print(f"  Source files:      {len(files):,}")
        if not args.summarize:
            print(f"  Est. Pass 1:       ~{pass1_tokens:,} tokens")
        if not args.discover:
            print(f"  Est. chunks:       ~{est_chunks:,} (avg 3/file)")
            print(f"  Est. file summaries: ~{file_summary_tokens:,} tokens ({len(files):,} files × ~500 tok)")
            print(f"  Est. func pre-pass:  ~{func_prepass_tokens:,} tokens (~20% multi-chunk funcs × ~600 tok)")
            print(f"  Est. func cards:     ~{func_card_tokens:,} tokens (~15% multi-chunk funcs × ~1200 tok)")
            print(f"  Est. class summaries: ~{class_summary_tokens:,} tokens (~1 class per 5 files)")
            print(f"  Est. Pass 2:       ~{est_prompt_tokens + est_completion_tokens:,} tokens")
            print(f"                     ({est_chunks:,} chunks × ~2,500 tok/chunk)")
        print(f"  Est. total tokens: ~{est_total:,}")
        if est_cost is not None:
            print(f"  Est. cost:         ~${est_cost:.2f} (at ${rate_per_mtok:.2f}/MTok — {_pricing_source})")
        else:
            _cost_model = args.model_fast or args.model
            print(f"  Est. cost:         unknown — '{_cost_model}' not in litellm pricing db")
            print(f"                     Check provider pricing and multiply by ~{est_total/1_000_000:.1f}M tokens")
        if est_minutes > 60:
            print(f"  Est. wall time:    ~{est_minutes / 60:.1f} hours (at {args.rpm} RPM)")
        else:
            print(f"  Est. wall time:    ~{int(est_minutes)} minutes (at {args.rpm} RPM)")
        print(f"{'='*60}")
        if not args.discover and not args.model_fast:
            print(f"\n  Tip: Use --model-fast openai/gpt-4o-mini to reduce Pass 2 cost by ~90%")
        print(f"\n  Note: Actual cost depends on model, file sizes, and content.")
        print(f"  Run without --dry-run to proceed.")
        return

    # Show cost estimate and confirm (unless --yes)
    if not args.yes:
        if est_minutes > 60:
            time_str = f"~{est_minutes / 60:.1f} hours"
        else:
            time_str = f"~{int(est_minutes)} minutes"
        if est_cost is not None:
            print(f"\nEstimate: ~{est_chunks:,} chunks, ~${est_cost:.2f}, {time_str} at {args.rpm} RPM")
        else:
            _cost_model = args.model_fast or args.model
            print(f"\nEstimate: ~{est_chunks:,} chunks, ~{est_total/1_000_000:.1f}M tokens "
                  f"(cost unknown — '{_cost_model}' not in litellm pricing db), {time_str} at {args.rpm} RPM")
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
        changed_files, new_manifest = filter_changed_files(
            codebase, files, hash_manifest_path, codebases=codebases)
        if not changed_files:
            print("[Incremental] No files changed since last run. Nothing to do.")
            return
        print(f"[Incremental] {len(changed_files)}/{len(files)} files changed — "
              f"only these will be re-summarized")
        # Remove old summaries for changed files so they get regenerated
        if summaries_path.exists():
            try:
                existing = json.loads(summaries_path.read_text())
                changed_rels = {_relative_to_any(f, codebases or [codebase])
                                for f in changed_files}
                kept = [e for e in existing if e.get("source_file") not in changed_rels]
                summaries_path.write_text(json.dumps(kept, indent=2))
                print(f"[Incremental] Purged {len(existing) - len(kept)} stale summaries, "
                      f"kept {len(kept)}")
            except Exception:
                pass

    tracker = TokenTracker()
    _start_time = time.time()

    # ── Index-only mode: just push existing summaries to R2R ─────────────────
    if args.purge_code:
        if not summaries_path.exists():
            print(f"Error: {summaries_path} not found — nothing to purge.")
            sys.exit(1)
        summaries = json.loads(summaries_path.read_text())
        doc_ids = []
        for e in summaries:
            if e.get("doc_id"):
                doc_ids.append(e["doc_id"])
            if e.get("code_doc_id"):
                doc_ids.append(e["code_doc_id"])
        if not doc_ids:
            print("No doc_ids found in summaries.json — nothing to purge.")
            print("(They are stored after a successful --reindex or full run.)")
            return
        print(f"Purging {len(doc_ids)} code chunk document(s) from R2R "
              f"(docs and tickets untouched)...")
        from codebase_shared.r2r_indexer import purge_docs
        purge_docs(doc_ids)
        # Clear stored doc_ids so a subsequent --reindex starts fresh
        for e in summaries:
            e.pop("doc_id", None)
            e.pop("code_doc_id", None)
        summaries_path.write_text(json.dumps(summaries, indent=2))
        print(f"Done. Re-run with --reindex (or full run) to re-populate R2R.")
        return

    if args.reindex:
        if not summaries_path.exists():
            print(f"Error: {summaries_path} not found. Run Pass 2 first.")
            sys.exit(1)
        summaries = json.loads(summaries_path.read_text())
        print(f"[Reindex] Indexing {len(summaries)} summaries from {summaries_path}")
        index_summaries_to_r2r(summaries)
        print("[Reindex] Done.")
        return

    # ── Review-only mode: re-review existing summaries ───────────────────────
    if args.improve:
        if not summaries_path.exists():
            print(f"Error: {summaries_path} not found. Run Pass 2 first.")
            sys.exit(1)
        summaries = json.loads(summaries_path.read_text())
        print(f"[Review-only] Reviewing {len(summaries)} summaries from {summaries_path}")
        summaries, edit_rate = asyncio.run(run_review(
            args.model, summaries, tracker=tracker,
            workers=4, rpm=args.rpm,
            max_concurrent=args.max_concurrent,
            quiet=args.quiet,
        ))
        summaries_path.write_text(json.dumps(summaries, indent=2))
        print(f"[Review-only] Done. Edit rate: {edit_rate:.1%}")
        print(f"  Run with --reindex to push updated summaries to R2R.")
        tracker.print_summary()
        return

    # ── Pass 1 ──────────────────────────────────────────────────────────────────
    if not args.summarize:
        # Check if an existing module_map has review errors → focused re-run
        _focused = False
        if module_map_path.exists():
            try:
                _existing = ModuleMap(**json.loads(module_map_path.read_text()))
                _error_issues = [i for i in _existing.review_issues
                                 if i.get("severity") == "error"]
                if _error_issues:
                    print(f"\n[Pass 1] Existing module_map has {len(_error_issues)} error issue(s) "
                          f"→ running focused refinement instead of full Pass 1")
                    module_map = run_pass1_focused(
                        args.model, _existing, codebase, files, args.language,
                        tracker=tracker, codebases=codebases,
                    )
                    _focused = True
            except Exception as _e:
                print(f"  [warn] Could not load existing module_map ({_e}) — running full Pass 1")

        if not _focused:
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
                                   tracker=tracker, codebases=codebases)

        module_map_path.write_text(json.dumps(module_map.model_dump(), indent=2))
        print(f"\n[Pass 1] Written to {module_map_path}")
    else:
        if not module_map_path.exists():
            print(f"Error: {module_map_path} not found. Run Pass 1 first.")
            sys.exit(1)
        module_map = ModuleMap(**json.loads(module_map_path.read_text()))
        print(f"[Pass 2] Loaded {len(module_map.modules)} modules from {module_map_path}")

    if args.discover:
        print("\nPass 1 complete. Run with --summarize to generate summaries.")
        return

    # ── Show ETA ─────────────────────────────────────────────────────────────
    if est_minutes > 60:
        print(f"\nEstimated time remaining: ~{est_minutes / 60:.1f} hours at {args.rpm} RPM")
    elif est_minutes > 1:
        print(f"\nEstimated time remaining: ~{int(est_minutes)} minutes at {args.rpm} RPM")

    # ── Pass 2: Summarization ───────────────────────────────────────────────────
    summaries = asyncio.run(run_pass2(
        args.model_fast or args.model, codebase, module_map,
        language=args.language,
        max_chunks=None,
        summaries_path=summaries_path,
        tracker=tracker,
        workers=4,
        rpm=args.rpm,
        max_concurrent=args.max_concurrent,
        quiet=args.quiet,
        verbose=args.verbose,
        codebases=codebases,
    ))
    summaries_path.write_text(json.dumps(summaries, indent=2))

    # Save hash manifest so --incremental can detect changes next time
    if new_manifest is not None:
        save_hash_manifest(hash_manifest_path, new_manifest)
        print(f"[Incremental] Saved file hashes to {hash_manifest_path}")

    # ── Targeted refine: re-summarize only vague summaries ──────────────────
    if args.refine and summaries and not _quota_exhausted:
        vague_indices = []
        for i, entry in enumerate(summaries):
            s = entry.get("summary", "")
            marker = _needs_reference_resolution(s)
            if marker and not entry.get("skip_index"):
                vague_indices.append((i, marker))

        if vague_indices:
            print(f"\n[Refine] Found {len(vague_indices)} summaries with vague references — re-reviewing")
            refined_summaries, edit_rate = asyncio.run(run_review(
                args.model,
                [summaries[i] for i, _ in vague_indices],
                tracker=tracker,
                workers=4, rpm=args.rpm,
                max_concurrent=args.max_concurrent,
                quiet=args.quiet,
            ))
            # Write back improved summaries
            for (orig_idx, _), refined in zip(vague_indices, refined_summaries):
                summaries[orig_idx] = refined
            summaries_path.write_text(json.dumps(summaries, indent=2))
            print(f"[Refine] Done. {len(vague_indices)} reviewed, edit rate: {edit_rate:.1%}")
        else:
            print(f"\n[Refine] No vague summaries found — skipping")

    # Always index into R2R
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
    example1 = dim('"How does X work?"')
    example2 = dim('"Where is Y implemented?"')
    print(f"  Try: {example1} or {example2}")
    print()

    if tracker.phases:
        print(dim(tracker.summary()))


if __name__ == "__main__":
    main()
