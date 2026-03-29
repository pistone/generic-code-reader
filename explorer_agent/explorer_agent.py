"""
Explorer Agent — Autonomous codebase exploration with no fixed iteration count.

Unlike the study_agent (fixed pipeline: Pass 1 → Pass 2 → Review), this agent
decides for itself what to read, how deep to go, and when to stop. It maintains
a "knowledge state" and keeps exploring until confidence is high or budget runs out.

It can optionally bootstrap from study_agent's existing KB (summaries.json,
module_map.json) to identify gaps and under-explored areas.

Output is compatible with study_agent: produces summaries.json entries that can
be indexed into R2R with the same schema.

Usage:
  # Fresh exploration (no prior KB)
  python -m explorer_agent.explorer_agent --codebase /path/to/src

  # Bootstrap from existing study_agent results, explore gaps
  python -m explorer_agent.explorer_agent --codebase /path/to/src \
      --existing-kb ../indexer/summaries.json

  # Control budget
  python -m explorer_agent.explorer_agent --codebase /path/to/src \
      --max-llm-calls 200 --max-cost 5.00
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codebase_shared.utils import (
    TokenTracker, llm_call, allm_call, llm_call_multi,
    _extract_content, _is_quota_error,
)
try:
    from codebase_shared.colors import green, yellow, red, bold, dim
except ImportError:
    green = yellow = red = bold = dim = lambda x: x  # type: ignore

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")
R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")

SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules",
    ".venv", "venv", "env", ".env", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".tox",
    "third_party", "thirdparty", "3rdparty", "vendor",
    "external", "deps",
    "cmake-build-debug", "cmake-build-release", "out",
    ".build", "_build",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll",
                 ".egg-info", ".dist-info", ".lock"}

# ── Knowledge State ───────────────────────────────────────────────────────────

class KnowledgeState:
    """Tracks what the agent knows and what it still wants to learn.

    This is the agent's "memory" — persisted to disk for resume support.
    """

    def __init__(self, save_path: Optional[Path] = None):
        self.save_path = save_path

        # What we've explored
        self.explored_dirs: set[str] = set()        # dirs we've listed
        self.read_files: set[str] = set()            # files we've read
        self.summarized_files: set[str] = set()      # files we've summarized

        # What we've learned
        self.architecture: str = ""                   # high-level understanding
        self.modules: dict[str, dict] = {}            # module_name → {description, files, questions, confidence}
        self.summaries: list[dict] = []               # compatible with study_agent
        self.insights: list[str] = []                 # cross-cutting observations
        self.hypotheses: list[dict] = []              # {claim, status, evidence}

        # What we want to learn
        self.frontier: list[dict] = []                # [{action, target, reason, priority}]
        self.open_questions: list[str] = []           # things we still don't understand

        # Budget tracking
        self.llm_calls: int = 0
        self.total_tokens: int = 0

    def confidence_score(self) -> float:
        """Overall confidence in our understanding. 0.0 = nothing, 1.0 = complete."""
        if not self.modules:
            return 0.0

        scores = []
        for mod in self.modules.values():
            scores.append(mod.get("confidence", 0.0))

        # Penalize open questions
        q_penalty = min(0.3, len(self.open_questions) * 0.03)

        return max(0.0, (sum(scores) / len(scores)) - q_penalty) if scores else 0.0

    def coverage(self, total_files: int) -> float:
        """Fraction of files we've at least read."""
        if total_files == 0:
            return 1.0
        return len(self.read_files) / total_files

    def save(self):
        """Persist to disk."""
        if not self.save_path:
            return
        data = {
            "explored_dirs": sorted(self.explored_dirs),
            "read_files": sorted(self.read_files),
            "summarized_files": sorted(self.summarized_files),
            "architecture": self.architecture,
            "modules": self.modules,
            "summaries": self.summaries,
            "insights": self.insights,
            "hypotheses": self.hypotheses,
            "frontier": self.frontier,
            "open_questions": self.open_questions,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "KnowledgeState":
        """Resume from a saved state."""
        ks = cls(save_path=path)
        if not path.exists():
            return ks
        try:
            data = json.loads(path.read_text())
            ks.explored_dirs = set(data.get("explored_dirs", []))
            ks.read_files = set(data.get("read_files", []))
            ks.summarized_files = set(data.get("summarized_files", []))
            ks.architecture = data.get("architecture", "")
            ks.modules = data.get("modules", {})
            ks.summaries = data.get("summaries", [])
            ks.insights = data.get("insights", [])
            ks.hypotheses = data.get("hypotheses", [])
            ks.frontier = data.get("frontier", [])
            ks.open_questions = data.get("open_questions", [])
            ks.llm_calls = data.get("llm_calls", 0)
            ks.total_tokens = data.get("total_tokens", 0)
        except Exception as e:
            print(f"[warn] Could not load state from {path}: {e}")
        return ks


# ── Codebase interface ────────────────────────────────────────────────────────

class CodebaseView:
    """Provides read-only access to the codebase for the agent's tools.

    root: the directory being explored (all_files, list_dir, search scope)
    context_root: optional wider directory for cross-reference reads.
        When set, read_file can access files outside root but inside
        context_root. This lets you explore a subdirectory while still
        following includes/imports that point elsewhere in the repo.
    """

    def __init__(self, root: Path, context_root: Optional[Path] = None):
        self.root = root.resolve()
        self.context_root = context_root.resolve() if context_root else None
        self._file_index: Optional[dict[str, list[Path]]] = None
        self._all_files: Optional[list[Path]] = None
        # Separate index for context_root (lazy-built on first cross-ref read)
        self._context_index: Optional[dict[str, list[Path]]] = None

    def _ensure_index(self):
        if self._file_index is not None:
            return
        self._file_index = {}
        self._all_files = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.suffix in SKIP_SUFFIXES:
                continue
            self._all_files.append(p)
            self._file_index.setdefault(p.name, []).append(p)

    def _ensure_context_index(self):
        """Build filename index over context_root (for cross-ref lookups)."""
        if self._context_index is not None or not self.context_root:
            return
        self._context_index = {}
        for p in self.context_root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.suffix in SKIP_SUFFIXES:
                continue
            self._context_index.setdefault(p.name, []).append(p)

    @property
    def all_files(self) -> list[Path]:
        self._ensure_index()
        return self._all_files  # type: ignore

    def list_dir(self, rel_dir: str = ".") -> dict:
        """List contents of a directory. Returns {files: [...], dirs: [...], file_counts: {...}}."""
        self._ensure_index()
        target = self.root / rel_dir if rel_dir != "." else self.root
        if not target.is_dir():
            return {"error": f"Not a directory: {rel_dir}"}

        files = []
        dirs = set()
        for f in self._all_files:  # type: ignore
            try:
                rel = f.relative_to(target)
            except ValueError:
                continue
            parts = rel.parts
            if len(parts) == 1:
                files.append(str(rel))
            elif len(parts) > 1:
                dirs.add(parts[0])

        # Count files per subdir
        file_counts = {}
        for d in dirs:
            d_path = target / d
            count = sum(1 for f in self._all_files if f.is_relative_to(d_path))  # type: ignore
            file_counts[d] = count

        return {
            "files": sorted(files),
            "dirs": sorted(dirs),
            "file_counts": file_counts,
        }

    def read_file(self, rel_path: str, max_lines: int = 200) -> str:
        """Read a file's content.

        Tries in order:
        1. Exact path under self.root
        2. Filename match in self.root index
        3. Exact path under self.context_root (if set)
        4. Filename match in self.context_root index (if set)
        """
        p = (self.root / rel_path).resolve()
        # Allow reads within root or context_root
        allowed_roots = [str(self.root)]
        if self.context_root:
            allowed_roots.append(str(self.context_root))
        if not any(str(p).startswith(r) for r in allowed_roots):
            return f"[error: path traversal blocked: {rel_path}]"
        if not p.exists():
            # Try fuzzy match by filename in exploration root
            name = Path(rel_path).name
            self._ensure_index()
            matches = self._file_index.get(name, [])  # type: ignore
            if matches:
                p = matches[0]
            elif self.context_root:
                # Try under context_root: exact path first, then filename match
                ctx_p = (self.context_root / rel_path).resolve()
                try:
                    _in_ctx = ctx_p.is_relative_to(self.context_root)
                except AttributeError:
                    _in_ctx = str(ctx_p).startswith(str(self.context_root) + os.sep)
                if ctx_p.exists() and _in_ctx:
                    p = ctx_p
                else:
                    self._ensure_context_index()
                    ctx_matches = self._context_index.get(name, [])  # type: ignore
                    if ctx_matches:
                        p = ctx_matches[0]
                    else:
                        return f"[file not found: {rel_path}]"
            else:
                return f"[file not found: {rel_path}]"
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            result = lines[:max_lines]
            if len(lines) > max_lines:
                result.append(f"\n... ({len(lines) - max_lines} more lines)")
            return "\n".join(result)
        except Exception as e:
            return f"[error reading {rel_path}: {e}]"

    def file_stats(self, rel_path: str) -> dict:
        """Get basic stats about a file without reading it fully."""
        p = self.root / rel_path
        if not p.exists():
            return {"error": "not found"}
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            return {
                "lines": len(lines),
                "size_bytes": p.stat().st_size,
                "extension": p.suffix,
            }
        except Exception as e:
            return {"error": str(e)}

    def search_content(self, pattern: str, max_results: int = 10) -> list[dict]:
        """Search files for a pattern using ripgrep (fast) or grep (fallback).

        Searches root first. If context_root is set and results are sparse,
        also searches the wider context (tagged with [context] prefix).
        """
        import subprocess
        import shutil

        results: list[dict] = []

        def _run_search(search_dir: Path, tag: str = "") -> list[dict]:
            """Run rg or grep and parse results."""
            remaining = max_results - len(results)
            if remaining <= 0:
                return []

            rg = shutil.which("rg")
            if rg:
                cmd = [
                    rg, "--no-heading", "--line-number", "--max-count",
                    str(remaining), "--max-columns", "200",
                    "--case-sensitive", "--iglob", "!.git",
                    pattern, str(search_dir),
                ]
            else:
                cmd = [
                    "grep", "-rn", "--max-count", str(remaining),
                    "-I",  # skip binary files
                    pattern, str(search_dir),
                ]

            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=30, errors="replace",
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fall back to Python search for this directory
                return self._python_search(search_dir, pattern, remaining, tag)

            hits = []
            for line in proc.stdout.splitlines():
                if len(results) + len(hits) >= max_results:
                    break
                # Format: filepath:lineno:text
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    try:
                        fpath = Path(parts[0])
                        rel = str(fpath.relative_to(search_dir))
                        hits.append({
                            "file": f"{tag}{rel}" if tag else rel,
                            "line": int(parts[1]),
                            "text": parts[2].strip()[:200],
                        })
                    except (ValueError, Exception):
                        continue
            return hits

        # Search exploration root first
        results.extend(_run_search(self.root))

        # If few results and context_root available, widen the search
        if len(results) < max_results and self.context_root:
            results.extend(_run_search(self.context_root, tag="[context] "))

        return results

    def _python_search(self, search_dir: Path, pattern: str,
                       max_results: int, tag: str = "") -> list[dict]:
        """Fallback: Python-based search when rg/grep unavailable."""
        import re
        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return [{"error": f"Invalid pattern: {pattern}"}]

        self._ensure_index()
        for f in (self._all_files or []):
            if len(results) >= max_results:
                break
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel = str(f.relative_to(search_dir))
                        results.append({
                            "file": f"{tag}{rel}" if tag else rel,
                            "line": i,
                            "text": line.strip()[:200],
                        })
                        if len(results) >= max_results:
                            break
            except Exception:
                continue
        return results

    def tree(self, max_depth: int = 2) -> str:
        """Directory tree string."""
        self._ensure_index()
        lines = [f"{self.root.name}/"]
        shown_dirs: set[str] = set()

        for f in self._all_files:  # type: ignore
            rel = f.relative_to(self.root)
            parts = rel.parts[:-1]
            for depth in range(min(len(parts), max_depth)):
                dir_path = "/".join(parts[:depth + 1])
                if dir_path not in shown_dirs:
                    shown_dirs.add(dir_path)
                    indent = "  " * (depth + 1)
                    # Count files under this dir
                    dir_full = self.root / dir_path
                    n = sum(1 for ff in self._all_files if ff.is_relative_to(dir_full))  # type: ignore
                    lines.append(f"{indent}{parts[depth]}/  ({n} files)")

        return "\n".join(lines)


# ── The Agent ─────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are an autonomous code exploration agent. Your goal is to deeply understand
a software codebase by reading its source code. You decide what to explore next.

You have a knowledge state that tracks:
- What you've already explored and understood
- Open questions you haven't answered yet
- Hypotheses you're testing
- Your confidence level per module

At each step, you analyze your current knowledge and decide the SINGLE best
next action. You are NOT doing a breadth-first scan — you go DEEP on the
parts that matter and SKIP the parts that don't.

Strategy guidelines:
- Start with the top-level structure to form an architectural hypothesis
- Read entry points, main files, and public APIs first
- Follow dependency chains when you need to understand how things connect
- When you find something complex, go deeper (read more of that module)
- When you find something trivial (config, boilerplate), note it and move on
- Form hypotheses and verify them ("I think X talks to Y via Z — let me check")
- Stop when you can confidently explain the system to a new developer

Output ONLY valid JSON — no markdown fences, no commentary outside JSON."""

PLANNER_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "decide_action",
            "description": "Decide the next exploration action",
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief reasoning about what you know and what you need to learn next",
                    },
                    "action": {
                        "type": "string",
                        "enum": [
                            "list_dir",
                            "read_file",
                            "read_files",
                            "search",
                            "summarize_understanding",
                            "define_module",
                            "form_hypothesis",
                            "verify_hypothesis",
                            "deep_dive",
                            "done",
                        ],
                        "description": "The action to take",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target for the action (path, query, module name, etc.)",
                    },
                    "targets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple targets (for read_files)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this action is the most valuable right now",
                    },
                    "priority_note": {
                        "type": "string",
                        "description": "What you'll do AFTER this action (1-2 words)",
                    },
                },
                "required": ["thinking", "action", "reason"],
            },
        },
    },
]

SUMMARIZER_SYSTEM = """\
You are building a knowledge base about a software codebase. Given source code
and context, write a concise domain-aware summary.

Requirements:
- Use the project's own vocabulary (class names, function names, domain terms)
- State what the code DOES, not just what it IS
- Be specific: mention actual names, data flows, algorithms
- 2-4 sentences for complex code, 1 sentence for trivial code
- Plain prose, no markdown"""

DEEP_DIVE_SYSTEM = """\
You are analyzing a complex code module in depth. Given multiple source files
from the same module, produce a structured analysis:

1. PURPOSE: What this module does in the system (1-2 sentences)
2. KEY ABSTRACTIONS: The main classes/structs/interfaces and what they represent
3. DATA FLOW: How data moves through this module
4. EXTERNAL INTERFACES: How other modules interact with this one
5. COMPLEXITY NOTES: Any non-obvious logic, algorithms, or patterns

Be specific — use actual names from the code. Plain prose, no markdown bullets."""

ARCHITECTURE_SYSTEM = """\
You are synthesizing your understanding of a software codebase's architecture.
Based on everything you've explored so far, write a clear architectural overview.

Include:
- What the system does (1-2 sentences)
- Major components and their responsibilities
- How components communicate / depend on each other
- Key data flows
- Design patterns or architectural styles you've identified

Be specific — use actual names from the code. Write for a developer who needs
to understand this codebase quickly."""


class ExplorerAgent:
    """Autonomous codebase exploration agent.

    The agent loop:
      1. Observe: review current knowledge state
      2. Plan: LLM decides what to do next
      3. Act: execute the chosen action
      4. Update: record findings, update state
      5. Evaluate: check if done or budget exhausted
    """

    def __init__(
        self,
        codebase: CodebaseView,
        model: str = DEFAULT_MODEL,
        model_fast: Optional[str] = None,
        state: Optional[KnowledgeState] = None,
        tracker: Optional[TokenTracker] = None,
        max_llm_calls: int = 300,
        max_cost: float = 10.0,
        confidence_target: float = 0.8,
        verbose: bool = False,
    ):
        self.codebase = codebase
        self.model = model
        self.model_fast = model_fast or model
        self.state = state or KnowledgeState()
        self.tracker = tracker or TokenTracker()
        self.max_llm_calls = max_llm_calls
        self.max_cost = max_cost
        self.confidence_target = confidence_target
        self.verbose = verbose

        self._quota_exhausted = False
        self._start_time = 0.0
        self._step_count = 0

    def _budget_ok(self) -> bool:
        """Check if we have budget remaining."""
        if self._quota_exhausted:
            return False
        if self.state.llm_calls >= self.max_llm_calls:
            print(f"\n{yellow('[Budget]')} Max LLM calls reached ({self.max_llm_calls})")
            return False
        return True

    def _llm(self, system: str, user: str, max_tokens: int = 2048,
             json_mode: bool = False, phase: str = "explore") -> str:
        """Make an LLM call with budget tracking."""
        try:
            result = llm_call(
                self.model, system, user,
                max_tokens=max_tokens,
                json_mode=json_mode,
                tracker=self.tracker,
                phase=phase,
            )
            self.state.llm_calls += 1
            return result
        except Exception as e:
            err_str = str(e).lower()
            if _is_quota_error(err_str):
                self._quota_exhausted = True
                print(f"\n{red('[Quota]')} Exhausted: {e}")
                return ""
            raise

    def _llm_multi(self, system: str, messages: list[dict], max_tokens: int = 2048,
                    json_mode: bool = False, phase: str = "explore") -> str:
        """Multi-turn LLM call with budget tracking."""
        try:
            result = llm_call_multi(
                self.model, system, messages,
                max_tokens=max_tokens,
                json_mode=json_mode,
                tracker=self.tracker,
                phase=phase,
            )
            self.state.llm_calls += 1
            return result
        except Exception as e:
            err_str = str(e).lower()
            if _is_quota_error(err_str):
                self._quota_exhausted = True
                print(f"\n{red('[Quota]')} Exhausted: {e}")
                return ""
            raise

    # ── Actions ───────────────────────────────────────────────────────────

    def _act_list_dir(self, target: str) -> str:
        """List directory contents."""
        info = self.codebase.list_dir(target or ".")
        self.state.explored_dirs.add(target or ".")
        if "error" in info:
            return info["error"]

        parts = []
        if info["files"]:
            parts.append(f"Files: {', '.join(info['files'][:30])}")
            if len(info['files']) > 30:
                parts.append(f"  ... and {len(info['files']) - 30} more")
        if info["dirs"]:
            dir_lines = []
            for d in info["dirs"]:
                count = info["file_counts"].get(d, "?")
                dir_lines.append(f"  {d}/  ({count} files)")
            parts.append("Subdirs:\n" + "\n".join(dir_lines))
        return "\n".join(parts) if parts else "(empty directory)"

    def _act_read_file(self, target: str) -> str:
        """Read a single file."""
        content = self.codebase.read_file(target, max_lines=200)
        self.state.read_files.add(target)
        return content

    def _act_read_files(self, targets: list[str]) -> str:
        """Read multiple files."""
        parts = []
        for t in targets[:6]:  # cap at 6 files per batch
            content = self.codebase.read_file(t, max_lines=120)
            self.state.read_files.add(t)
            parts.append(f"=== {t} ===\n{content}")
        return "\n\n".join(parts)

    def _act_search(self, target: str) -> str:
        """Search for a pattern across files."""
        results = self.codebase.search_content(target, max_results=15)
        if not results:
            return f"No matches for: {target}"
        if results and "error" in results[0]:
            return results[0]["error"]
        lines = []
        for r in results:
            lines.append(f"  {r['file']}:{r['line']}  {r['text']}")
        return "\n".join(lines)

    def _act_summarize_understanding(self, target: str) -> str:
        """Summarize current understanding of a file or area, producing a KB entry."""
        # Read the file if we haven't
        if target not in self.state.read_files:
            content = self.codebase.read_file(target, max_lines=300)
            self.state.read_files.add(target)
        else:
            content = self.codebase.read_file(target, max_lines=300)

        # Build context from what we already know
        context_parts = []
        if self.state.architecture:
            context_parts.append(f"Architecture: {self.state.architecture[:500]}")

        # Find which module this file belongs to
        module_name = "unknown"
        module_desc = ""
        for mname, minfo in self.state.modules.items():
            if target in minfo.get("files", []):
                module_name = mname
                module_desc = minfo.get("description", "")
                break

        context = "\n".join(context_parts)
        prompt = f"""File: {target}
Module: {module_name} — {module_desc}
{f'Context: {context}' if context else ''}

Source code:
```
{content[:6000]}
```

Write a domain-aware summary of this file."""

        summary = self._llm(SUMMARIZER_SYSTEM, prompt, max_tokens=400,
                            phase="summarize")
        if not summary:
            return "(LLM call failed)"

        # Create KB entry compatible with study_agent
        entry = {
            "summary": summary.strip(),
            "raw_code": content[:4000],
            "source_file": target,
            "module": module_name,
            "chunk_type": "agent_exploration",
            "chunk_index": 0,
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:12],
        }
        self.state.summaries.append(entry)
        self.state.summarized_files.add(target)

        return f"Summarized: {summary.strip()[:200]}..."

    def _act_define_module(self, target: str, description: str = "",
                           files: list[str] = None, questions: list[str] = None) -> str:
        """Define or update a module in our knowledge map."""
        self.state.modules[target] = {
            "description": description,
            "files": files or [],
            "questions": questions or [],
            "confidence": 0.3,  # starts low, agent raises as it learns more
        }
        return f"Defined module: {target}"

    def _act_deep_dive(self, target: str) -> str:
        """Deep dive into a module — reads multiple files and synthesizes."""
        # Find files for this module
        mod = self.state.modules.get(target, {})
        mod_files = mod.get("files", [])

        if not mod_files:
            # Try to find files in the directory matching the module name
            info = self.codebase.list_dir(target)
            if "error" not in info:
                mod_files = [f"{target}/{f}" for f in info.get("files", [])][:10]

        if not mod_files:
            return f"No files found for module: {target}"

        # Read key files (up to 5)
        file_contents = []
        for f in mod_files[:5]:
            content = self.codebase.read_file(f, max_lines=150)
            self.state.read_files.add(f)
            file_contents.append(f"=== {f} ===\n{content}")

        all_content = "\n\n".join(file_contents)

        prompt = f"""Module: {target}
Description so far: {mod.get('description', 'unknown')}

Source files:
{all_content[:12000]}

Analyze this module in depth."""

        analysis = self._llm(DEEP_DIVE_SYSTEM, prompt, max_tokens=1500,
                             phase="deep_dive")
        if not analysis:
            return "(LLM call failed)"

        # Update module confidence
        if target in self.state.modules:
            self.state.modules[target]["confidence"] = min(
                0.9, self.state.modules[target].get("confidence", 0) + 0.3
            )
            self.state.modules[target]["deep_analysis"] = analysis.strip()

        # Also add as a KB entry
        entry = {
            "summary": f"[Module deep-dive: {target}] {analysis.strip()}",
            "raw_code": "",
            "source_file": f"module:{target}",
            "module": target,
            "chunk_type": "module_analysis",
            "chunk_index": -1,
        }
        self.state.summaries.append(entry)

        return f"Deep dive complete:\n{analysis.strip()[:400]}..."

    def _act_form_hypothesis(self, target: str) -> str:
        """Record a hypothesis to test."""
        self.state.hypotheses.append({
            "claim": target,
            "status": "untested",
            "evidence": [],
        })
        return f"Hypothesis recorded: {target}"

    def _act_verify_hypothesis(self, target: str) -> str:
        """Find and verify a hypothesis by index or claim text."""
        # Find the hypothesis
        hyp = None
        for h in self.state.hypotheses:
            if h["status"] == "untested" and (target in h["claim"] or h["claim"] in target):
                hyp = h
                break
        if not hyp:
            return f"No untested hypothesis matching: {target}"

        # Ask the LLM to evaluate based on what we know
        evidence_ctx = "\n".join(f"- {e}" for e in hyp["evidence"]) if hyp["evidence"] else "(none collected yet)"
        known_files = ", ".join(sorted(self.state.read_files)[:20])

        prompt = f"""Hypothesis: {hyp['claim']}

Evidence collected so far:
{evidence_ctx}

Files I've read: {known_files}

Based on the above, evaluate this hypothesis. Is it:
1. CONFIRMED — strong evidence supports it
2. REFUTED — evidence contradicts it
3. NEEDS_MORE — need to read specific files to confirm

Respond in JSON: {{"verdict": "confirmed|refuted|needs_more", "reasoning": "...", "next_action": "..."}}"""

        raw = self._llm(PLANNER_SYSTEM, prompt, json_mode=True, max_tokens=300,
                        phase="verify")
        if not raw:
            return "(LLM call failed)"

        try:
            result = json.loads(raw)
            hyp["status"] = result.get("verdict", "needs_more")
            hyp["evidence"].append(result.get("reasoning", ""))
            self.state.insights.append(
                f"Hypothesis '{hyp['claim'][:60]}': {hyp['status']} — "
                f"{result.get('reasoning', '')[:100]}"
            )
            return f"Verdict: {hyp['status']} — {result.get('reasoning', '')[:200]}"
        except json.JSONDecodeError:
            return f"Could not parse verdict: {raw[:200]}"

    # ── Planning ──────────────────────────────────────────────────────────

    def _build_state_summary(self) -> str:
        """Summarize current knowledge state for the planner."""
        total_files = len(self.codebase.all_files)
        parts = [
            f"## Exploration State (step {self._step_count})",
            f"Files: {len(self.state.read_files)} read, "
            f"{len(self.state.summarized_files)} summarized, "
            f"{total_files} total in codebase",
            f"Dirs explored: {len(self.state.explored_dirs)}",
            f"Modules defined: {len(self.state.modules)}",
            f"Confidence: {self.state.confidence_score():.0%}",
            f"Budget: {self.state.llm_calls}/{self.max_llm_calls} LLM calls used",
            f"Open questions: {len(self.state.open_questions)}",
            f"Untested hypotheses: {sum(1 for h in self.state.hypotheses if h['status'] == 'untested')}",
        ]

        if self.state.architecture:
            parts.append(f"\n## Architecture understanding:\n{self.state.architecture[:600]}")

        if self.state.modules:
            parts.append("\n## Known modules:")
            for name, info in self.state.modules.items():
                conf = info.get("confidence", 0)
                n_files = len(info.get("files", []))
                parts.append(f"  - {name} ({conf:.0%} confident, {n_files} files): "
                             f"{info.get('description', '?')[:80]}")

        if self.state.open_questions:
            parts.append("\n## Open questions:")
            for q in self.state.open_questions[:5]:
                parts.append(f"  - {q}")

        if self.state.insights:
            parts.append(f"\n## Recent insights ({len(self.state.insights)} total):")
            for i in self.state.insights[-3:]:
                parts.append(f"  - {i[:120]}")

        return "\n".join(parts)

    def _plan_next_action(self) -> Optional[dict]:
        """Ask the LLM to decide the next action."""
        state_summary = self._build_state_summary()

        # On first step, include the directory tree and context info
        extra = ""
        if self._step_count == 0:
            tree = self.codebase.tree(max_depth=2)
            extra = f"\n\n## Directory tree:\n```\n{tree}\n```"
            if self.codebase.context_root:
                extra += (
                    f"\n\nNote: You are exploring a subdirectory. "
                    f"You can read_file with paths relative to the wider repo root "
                    f"({self.codebase.context_root.name}/) to follow cross-references "
                    f"like #include or import statements that point outside this directory."
                )

        prompt = f"""{state_summary}{extra}

Given the current state, decide your SINGLE next action.

Available actions:
- list_dir: List files/subdirs in a directory (target = dir path)
- read_file: Read a single file in detail (target = file path)
- read_files: Read multiple related files (targets = list of paths)
- search: Search for a pattern/identifier across files (target = regex pattern)
- summarize_understanding: Write a KB summary of a file (target = file path)
- define_module: Define or update a module (target = module name)
- form_hypothesis: Record a hypothesis to test later (target = claim)
- verify_hypothesis: Check a hypothesis against evidence (target = claim)
- deep_dive: In-depth analysis of a module (target = module name)
- done: Stop exploring — you have sufficient understanding

Think carefully: what single action gives you the most new understanding?
Avoid re-reading files you've already read unless needed for verification."""

        from litellm import completion

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            response = completion(
                model=self.model,
                messages=messages,
                max_tokens=500,
                tools=PLANNER_TOOLS_SCHEMA,
                tool_choice={"type": "function", "function": {"name": "decide_action"}},
            )
            self.state.llm_calls += 1
            if self.tracker:
                self.tracker.record("plan", response)

            msg = response.choices[0].message
            if msg.tool_calls:
                args = json.loads(msg.tool_calls[0].function.arguments)
                return args
            return None
        except Exception as e:
            err_str = str(e).lower()
            if _is_quota_error(err_str):
                self._quota_exhausted = True
                print(f"\n{red('[Quota]')} Exhausted during planning: {e}")
            else:
                print(f"\n{red('[Error]')} Planning failed: {e}")
            return None

    # ── Main loop ─────────────────────────────────────────────────────────

    def _execute_action(self, plan: dict) -> str:
        """Execute a planned action and return the observation."""
        action = plan.get("action", "")
        target = plan.get("target", "")
        targets = plan.get("targets", [])

        if action == "list_dir":
            return self._act_list_dir(target)
        elif action == "read_file":
            return self._act_read_file(target)
        elif action == "read_files":
            return self._act_read_files(targets or [target])
        elif action == "search":
            return self._act_search(target)
        elif action == "summarize_understanding":
            return self._act_summarize_understanding(target)
        elif action == "define_module":
            # Extract description from thinking/reason
            return self._act_define_module(
                target,
                description=plan.get("reason", ""),
            )
        elif action == "form_hypothesis":
            return self._act_form_hypothesis(target)
        elif action == "verify_hypothesis":
            return self._act_verify_hypothesis(target)
        elif action == "deep_dive":
            return self._act_deep_dive(target)
        elif action == "done":
            return "DONE"
        else:
            return f"Unknown action: {action}"

    def _synthesize_architecture(self):
        """Generate/update the architectural overview."""
        if not self.state.modules:
            return

        modules_desc = []
        for name, info in self.state.modules.items():
            modules_desc.append(
                f"- {name}: {info.get('description', '?')} "
                f"({len(info.get('files', []))} files, "
                f"{info.get('confidence', 0):.0%} confidence)"
            )
            if info.get("deep_analysis"):
                modules_desc.append(f"  Analysis: {info['deep_analysis'][:200]}")

        insights = "\n".join(f"- {i}" for i in self.state.insights[-10:])

        prompt = f"""Modules identified:
{chr(10).join(modules_desc)}

Insights:
{insights}

Files read: {len(self.state.read_files)} of {len(self.codebase.all_files)}
Summaries written: {len(self.state.summaries)}

Synthesize the overall architecture."""

        result = self._llm(ARCHITECTURE_SYSTEM, prompt, max_tokens=1500,
                           phase="architecture")
        if result:
            self.state.architecture = result.strip()

    def _update_state_from_plan(self, plan: dict, observation: str):
        """Update knowledge state after an action completes.

        Uses the planner LLM to extract structured updates (new modules,
        questions, insights) from the observation.
        """
        action = plan.get("action", "")
        target = plan.get("target", "")

        # For certain actions, ask the LLM to extract updates
        if action in ("list_dir", "read_file", "read_files", "deep_dive") and observation:
            update_prompt = f"""You just performed: {action} on {target}
Your reasoning was: {plan.get('thinking', '')}

Observation:
{observation[:3000]}

Based on this, provide structured updates in JSON:
{{
    "new_modules": [{{"name": "...", "description": "...", "files": ["..."]}}],
    "updated_modules": [{{"name": "...", "additional_files": ["..."], "confidence_delta": 0.1}}],
    "new_questions": ["..."],
    "resolved_questions": ["..."],
    "insights": ["..."]
}}

Only include fields that have actual updates. Empty lists are fine."""

            raw = self._llm(PLANNER_SYSTEM, update_prompt, json_mode=True,
                            max_tokens=800, phase="update")
            if raw:
                try:
                    updates = json.loads(raw)
                    # Apply updates
                    for new_mod in updates.get("new_modules", []):
                        name = new_mod.get("name", "")
                        if name and name not in self.state.modules:
                            self.state.modules[name] = {
                                "description": new_mod.get("description", ""),
                                "files": new_mod.get("files", []),
                                "questions": [],
                                "confidence": 0.2,
                            }

                    for upd in updates.get("updated_modules", []):
                        name = upd.get("name", "")
                        if name in self.state.modules:
                            mod = self.state.modules[name]
                            mod["files"] = list(set(
                                mod.get("files", []) + upd.get("additional_files", [])
                            ))
                            delta = upd.get("confidence_delta", 0.1)
                            mod["confidence"] = min(0.95, mod.get("confidence", 0) + delta)

                    for q in updates.get("new_questions", []):
                        if q and q not in self.state.open_questions:
                            self.state.open_questions.append(q)

                    for q in updates.get("resolved_questions", []):
                        self.state.open_questions = [
                            oq for oq in self.state.open_questions
                            if q.lower() not in oq.lower()
                        ]

                    for insight in updates.get("insights", []):
                        if insight:
                            self.state.insights.append(insight)

                except json.JSONDecodeError:
                    pass  # best effort

    def run(self) -> KnowledgeState:
        """Main exploration loop."""
        self._start_time = time.monotonic()
        total_files = len(self.codebase.all_files)

        # Header
        print(f"\n{'='*60}")
        print(f"  {bold('Explorer Agent')} — Autonomous Codebase Exploration")
        print(f"{'='*60}")
        print(f"  Codebase:    {self.codebase.root}")
        print(f"  Total files: {total_files:,}")
        print(f"  Model:       {self.model}")
        print(f"  Budget:      {self.max_llm_calls} LLM calls")
        print(f"  Confidence target: {self.confidence_target:.0%}")

        if self.state.llm_calls > 0:
            print(f"\n  {green('[Resume]')} Resuming from step {self._step_count} "
                  f"({self.state.llm_calls} calls used, "
                  f"{self.state.confidence_score():.0%} confidence)")

        # Bootstrap from existing KB if available
        if self.state.summaries:
            print(f"  {green('[KB]')} {len(self.state.summaries)} existing summaries loaded")

        print(f"\n{'─'*60}")

        # ── Main loop ──────────────────────────────────────────────────
        while self._budget_ok():
            self._step_count += 1
            elapsed = time.monotonic() - self._start_time

            # Check confidence
            conf = self.state.confidence_score()
            if conf >= self.confidence_target and self._step_count > 5:
                print(f"\n{green('[Done]')} Confidence target reached: {conf:.0%}")
                break

            # Plan
            plan = self._plan_next_action()
            if plan is None:
                print(f"\n{yellow('[Halt]')} Planner returned no action")
                break

            action = plan.get("action", "?")
            target = plan.get("target", "")
            thinking = plan.get("thinking", "")

            # Show step
            elapsed_str = f"{int(elapsed)}s"
            print(f"\n{bold(f'[Step {self._step_count}]')} "
                  f"({self.state.llm_calls}/{self.max_llm_calls} calls, "
                  f"{conf:.0%} conf, {elapsed_str})")

            if self.verbose and thinking:
                print(f"  {dim('Thinking:')} {thinking[:150]}")

            if action == "done":
                reason = plan.get("reason", "sufficient understanding")
                print(f"  {green('→ Done:')} {reason}")
                break

            target_display = target[:60] if target else plan.get("targets", [""])[0][:60]
            print(f"  → {bold(action)} {target_display}")
            if plan.get("reason"):
                print(f"    {dim(plan['reason'][:100])}")

            # Execute
            observation = self._execute_action(plan)

            if self.verbose and observation:
                # Show first few lines of observation
                obs_lines = observation.strip().splitlines()[:5]
                for line in obs_lines:
                    print(f"    {dim(line[:120])}")
                if len(observation.strip().splitlines()) > 5:
                    print(f"    {dim(f'... ({len(observation)} chars total)')}")

            # Update state from observation
            if observation and action != "done":
                self._update_state_from_plan(plan, observation)

            # Periodically synthesize architecture
            if self._step_count % 10 == 0 and self.state.modules:
                print(f"\n  {dim('[Synthesizing architecture...]')}")
                self._synthesize_architecture()

            # Save state periodically
            if self._step_count % 5 == 0:
                self.state.save()

        # ── Final synthesis ─────────────────────────────────────────────
        if self.state.modules and self._budget_ok():
            print(f"\n{dim('[Final architecture synthesis...]')}")
            self._synthesize_architecture()

        # Save final state
        self.state.save()

        # ── Summary ─────────────────────────────────────────────────────
        elapsed = time.monotonic() - self._start_time
        self._print_summary(elapsed)

        return self.state

    def _print_summary(self, elapsed: float):
        """Print a post-run summary card."""
        total_files = len(self.codebase.all_files)
        conf = self.state.confidence_score()

        print(f"\n{'='*60}")
        print(f"  {bold('Explorer Agent — Results')}")
        print(f"{'='*60}")
        print(f"  Steps:           {self._step_count}")
        print(f"  LLM calls:       {self.state.llm_calls}/{self.max_llm_calls}")
        print(f"  Time:            {int(elapsed)}s ({elapsed/60:.1f}m)")
        print(f"  Files read:      {len(self.state.read_files)}/{total_files}")
        print(f"  Files summarized: {len(self.state.summarized_files)}")
        print(f"  Modules:         {len(self.state.modules)}")
        print(f"  KB entries:      {len(self.state.summaries)}")
        print(f"  Insights:        {len(self.state.insights)}")
        print(f"  Confidence:      {conf:.0%}")

        if self.state.open_questions:
            print(f"\n  Open questions ({len(self.state.open_questions)}):")
            for q in self.state.open_questions[:5]:
                print(f"    - {q[:80]}")

        if self.state.modules:
            print(f"\n  Modules:")
            for name, info in self.state.modules.items():
                c = info.get("confidence", 0)
                bar = "█" * int(c * 10) + "░" * (10 - int(c * 10))
                print(f"    [{bar}] {name}: {info.get('description', '?')[:60]}")

        if self._quota_exhausted:
            print(f"\n  {yellow('⚠ Quota exhausted — re-run to continue exploration')}")

        if self.tracker:
            print(f"\n{self.tracker.summary()}")

        print(f"{'='*60}")


# ── KB Gap Analysis (bootstrap from study_agent) ─────────────────────────────

def analyze_existing_kb(summaries_path: Path) -> dict:
    """Analyze an existing KB to find gaps and weak areas.

    Returns a dict with:
    - modules: {name: {file_count, avg_summary_len, weak_files}}
    - uncovered_types: file extensions not in the KB
    - short_summaries: entries with very short summaries
    - suggested_focus: list of areas to explore deeper
    """
    if not summaries_path.exists():
        return {"error": "summaries.json not found"}

    summaries = json.loads(summaries_path.read_text())
    if not summaries:
        return {"error": "empty summaries"}

    modules: dict[str, dict] = {}
    short = []
    files_seen = set()

    for entry in summaries:
        mod = entry.get("module", "unknown")
        src = entry.get("source_file", "")
        summary = entry.get("summary", "")
        files_seen.add(src)

        if mod not in modules:
            modules[mod] = {"file_count": 0, "total_len": 0, "count": 0,
                            "weak_files": []}
        m = modules[mod]
        m["count"] += 1
        m["total_len"] += len(summary)

        if len(summary) < 80:
            short.append({"file": src, "module": mod, "summary_len": len(summary)})
            m["weak_files"].append(src)

    # Compute averages
    for mod, info in modules.items():
        info["avg_summary_len"] = info["total_len"] // max(info["count"], 1)
        info["file_count"] = len(set(
            e.get("source_file", "") for e in summaries if e.get("module") == mod
        ))

    # Suggest focus areas: modules with low avg summary length or many weak files
    focus = []
    for mod, info in sorted(modules.items(),
                             key=lambda x: len(x[1]["weak_files"]), reverse=True):
        if info["weak_files"]:
            focus.append({
                "module": mod,
                "reason": f"{len(info['weak_files'])} weak summaries "
                          f"(avg {info['avg_summary_len']} chars)",
                "files": info["weak_files"][:5],
            })

    return {
        "total_entries": len(summaries),
        "modules": modules,
        "short_summaries": short[:20],
        "suggested_focus": focus[:10],
        "files_covered": len(files_seen),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explorer Agent: autonomous codebase exploration"
    )
    parser.add_argument("--codebase", required=True,
                        help="Root directory of the codebase to explore")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="LLM model string (default: %(default)s)")
    parser.add_argument("--model-fast", default=None,
                        help="Cheaper model for bulk summarization")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write results (default: same dir as script)")
    parser.add_argument("--context-root", default=None,
                        help="Wider codebase root for cross-reference reads. "
                             "When set, the agent explores only --codebase but can "
                             "follow includes/imports into files under --context-root.")
    parser.add_argument("--existing-kb", default=None,
                        help="Path to existing summaries.json from study_agent "
                             "(bootstraps exploration from gaps)")
    parser.add_argument("--max-llm-calls", type=int, default=300,
                        help="Max LLM calls budget (default: 300)")
    parser.add_argument("--max-cost", type=float, default=10.0,
                        help="Max estimated cost in dollars (default: 10.0)")
    parser.add_argument("--confidence-target", type=float, default=0.8,
                        help="Stop when confidence reaches this level (default: 0.8)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed thinking and observations")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing KB gaps, don't explore")
    args = parser.parse_args()

    codebase = Path(args.codebase).resolve()
    if not codebase.is_dir():
        print(f"Error: --codebase '{codebase}' is not a directory")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze existing KB if requested
    if args.existing_kb:
        kb_path = Path(args.existing_kb).resolve()
        print(f"\n[KB Analysis] Analyzing {kb_path}...")
        analysis = analyze_existing_kb(kb_path)

        if "error" in analysis:
            print(f"  {red(analysis['error'])}")
        else:
            print(f"  Total entries: {analysis['total_entries']}")
            print(f"  Files covered: {analysis['files_covered']}")
            print(f"  Modules: {len(analysis['modules'])}")
            if analysis["suggested_focus"]:
                print(f"\n  Suggested focus areas:")
                for area in analysis["suggested_focus"][:5]:
                    print(f"    - {area['module']}: {area['reason']}")

        if args.analyze_only:
            # Write analysis and exit
            (output_dir / "kb_gap_analysis.json").write_text(
                json.dumps(analysis, indent=2)
            )
            print(f"\nAnalysis written to {output_dir / 'kb_gap_analysis.json'}")
            return

    # Set up state (resume support)
    state_path = output_dir / "explorer_state.json"
    state = KnowledgeState.load(state_path)
    state.save_path = state_path

    # Bootstrap from existing KB
    if args.existing_kb and not state.summaries:
        kb_path = Path(args.existing_kb).resolve()
        if kb_path.exists():
            existing = json.loads(kb_path.read_text())
            state.summaries = existing
            for entry in existing:
                state.summarized_files.add(entry.get("source_file", ""))
            print(f"  Bootstrapped {len(existing)} entries from existing KB")

            # Seed open questions from gap analysis
            analysis = analyze_existing_kb(kb_path)
            for area in analysis.get("suggested_focus", [])[:5]:
                state.open_questions.append(
                    f"Why are summaries weak in module '{area['module']}'? "
                    f"({area['reason']})"
                )

    # Create agent
    tracker = TokenTracker()
    context_root = Path(args.context_root).resolve() if args.context_root else None
    if context_root:
        if not context_root.is_dir():
            print(f"Error: --context-root '{context_root}' is not a directory")
            sys.exit(1)
        if not codebase.is_relative_to(context_root):
            print(f"{yellow('[warn]')} --codebase is not inside --context-root. "
                  f"Cross-references may not resolve correctly.")
        else:
            print(f"[Context] Exploration: {codebase.relative_to(context_root)}/  "
                  f"  Cross-refs: {context_root}/")
    view = CodebaseView(codebase, context_root=context_root)

    agent = ExplorerAgent(
        codebase=view,
        model=args.model,
        model_fast=args.model_fast,
        state=state,
        tracker=tracker,
        max_llm_calls=args.max_llm_calls,
        max_cost=args.max_cost,
        confidence_target=args.confidence_target,
        verbose=args.verbose,
    )

    # Run
    final_state = agent.run()

    # Write outputs
    summaries_path = output_dir / "summaries.json"
    summaries_path.write_text(json.dumps(final_state.summaries, indent=2))
    print(f"\nSummaries: {summaries_path} ({len(final_state.summaries)} entries)")

    # Write module map (compatible with study_agent format)
    if final_state.modules:
        module_map = {
            "project": codebase.name,
            "description": final_state.architecture[:200] if final_state.architecture else "",
            "modules": [
                {
                    "name": name,
                    "description": info.get("description", ""),
                    "files": info.get("files", []),
                    "questions": info.get("questions", []),
                }
                for name, info in final_state.modules.items()
            ],
        }
        module_map_path = output_dir / "module_map.json"
        module_map_path.write_text(json.dumps(module_map, indent=2))
        print(f"Module map: {module_map_path} ({len(final_state.modules)} modules)")

    # Write architecture overview
    if final_state.architecture:
        arch_path = output_dir / "architecture.md"
        arch_path.write_text(final_state.architecture)
        print(f"Architecture: {arch_path}")

    # Write cost log
    log_entry = tracker.to_log_entry(args.model, agent="explorer_agent",
                                      codebase=str(codebase))
    cost_log_path = output_dir / "cost_log.jsonl"
    with open(cost_log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


if __name__ == "__main__":
    main()
