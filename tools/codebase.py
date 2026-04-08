"""
tools/codebase.py — Codebase module discovery and summarization tools.

discover_modules: Uses Claude directly to identify modules from directory structure,
                  manifests, READMEs, and index files. Single-shot — much more reliable
                  than the exploration loop.

summarize_codebase: Runs Pass 2 (chunking + summarization) over a module map,
                    indexes results into R2R.

import_modules: Loads module definitions from a hand-crafted JSON file
                (e.g. indexer/analysis_modules.json) and generates questions.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Make project root importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from codebase_shared.utils import llm_call, TokenTracker

# ── Skip patterns ─────────────────────────────────────────────────────────────

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".env",
    "target", "build", "dist", ".idea", ".vscode", "vendor", ".cache",
    "coverage", ".mypy_cache", ".pytest_cache", ".tox", "eggs",
    ".eggs", "htmlcov", ".hypothesis", "buck-out", "_build", ".stack-work",
}

_MANIFEST_FILES = [
    "package.json", "Cargo.toml", "go.mod", "setup.py", "pyproject.toml",
    "setup.cfg", "pom.xml", "build.gradle", "CMakeLists.txt", "pubspec.yaml",
    "composer.json", "Gemfile", "mix.exs", "project.clj",
]

_INDEX_FILES = [
    "__init__.py", "index.ts", "index.js", "lib.rs", "mod.rs",
    "lib.py", "main.py", "index.py",
]

# ── Context gathering ─────────────────────────────────────────────────────────

def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS or part.startswith(".") for part in path.parts)


def _build_dir_tree(root: Path, max_depth: int = 4, _depth: int = 0) -> str:
    """Filtered directory tree, depth-limited."""
    lines = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return ""
    for entry in entries:
        if entry.name in _SKIP_DIRS or entry.name.startswith("."):
            continue
        prefix = "  " * _depth
        if entry.is_dir():
            lines.append(f"{prefix}{entry.name}/")
            if _depth < max_depth:
                sub = _build_dir_tree(entry, max_depth, _depth + 1)
                if sub:
                    lines.append(sub)
        else:
            lines.append(f"{prefix}{entry.name}")
    return "\n".join(lines)


def gather_codebase_context(codebase: Path, language: str = "auto") -> str:
    """Collect high-signal context: tree, manifests, READMEs, index files.

    Returns a single string ready to paste into a Claude prompt.
    """
    parts: list[str] = []

    # 1. Directory tree
    tree = _build_dir_tree(codebase)
    parts.append(f"## Directory Tree\n```\n{tree}\n```")

    # 2. Package manifests at root
    for fname in _MANIFEST_FILES:
        fpath = codebase / fname
        if fpath.exists() and fpath.stat().st_size < 20_000:
            content = fpath.read_text(errors="replace")[:3000]
            parts.append(f"## {fname}\n```\n{content}\n```")

    # 3. README files (root first, then subdirs up to 8 total)
    readmes = sorted(
        [p for p in codebase.rglob("README*") if not _should_skip(p)],
        key=lambda p: len(p.parts),
    )[:8]
    for readme in readmes:
        rel = readme.relative_to(codebase)
        content = readme.read_text(errors="replace")[:2000]
        parts.append(f"## {rel}\n{content}")

    # 4. Key index files at each major directory (first 2 levels)
    for fname in _INDEX_FILES:
        matches = sorted(
            [p for p in codebase.rglob(fname) if not _should_skip(p)],
            key=lambda p: len(p.parts),
        )[:20]
        for fpath in matches:
            rel = fpath.relative_to(codebase)
            # Only include if it has meaningful exports/imports
            content = fpath.read_text(errors="replace")[:1500].strip()
            if len(content) > 50:
                parts.append(f"## {rel}\n```\n{content}\n```")

    return "\n\n---\n\n".join(parts)


# ── Claude-based module discovery ─────────────────────────────────────────────

_DISCOVER_SYSTEM = """\
You are a senior software architect. Given a codebase's directory structure,
manifests, READMEs, and key index files, identify all meaningful modules.

A module is a coherent subsystem with a single clear responsibility.
Err on the side of MORE, SMALLER modules — don't lump unrelated things together.
Each significant directory is likely its own module or part of one.

Output ONLY valid JSON."""

_DISCOVER_PROMPT = """\
Analyse this codebase and identify all meaningful modules.

{context}

Output JSON with this exact structure:
{{
  "project": "<project name>",
  "description": "<1-2 sentence overall architecture summary>",
  "modules": [
    {{
      "name": "<short-lowercase-hyphenated-name>",
      "description": "<1-2 sentences: what this module is responsible for>",
      "dir_paths": ["<relative/dir>", ...]
    }}
  ]
}}

Rules:
- Every significant source directory must appear in exactly one module's dir_paths
- A module with 3+ unrelated directories should be split into separate modules
- Test/fixture directories should be their own module named "tests"
- Don't create a catch-all "misc" or "other" module — assign everything specifically
- Infrastructure, config, scripts can each be their own module if significant
- Root-level files (no subdirectory) can be assigned dir_path "." """

_QUESTIONS_SYSTEM = (
    "You are generating targeted questions to guide codebase summarization. "
    "Output ONLY valid JSON."
)

_QUESTIONS_PROMPT = """\
Generate 6-10 domain-specific investigative questions for this codebase module.
These questions guide an LLM when writing summaries — they should focus attention
on what matters most in this module's code.

Module: {name}
Description: {description}
Directories: {dirs}
Sample files: {sample_files}

Good questions:
- Name specific algorithms, data structures, protocols, or failure modes
- Ask about contracts, invariants, error handling, concurrency, or performance
- Reference domain vocabulary from the description

Bad questions (never produce these):
- "What does X do?" — too generic
- "How is Y used?" — too vague

Output JSON:
{{"questions": ["...", "...", ...]}}"""


def _discover_modules_raw(
    context: str,
    model: str = "anthropic/claude-opus-4-5",
    tracker: Optional[TokenTracker] = None,
) -> dict:
    """Single LLM call → raw module JSON."""
    prompt = _DISCOVER_PROMPT.format(context=context)
    raw = llm_call(model, _DISCOVER_SYSTEM, prompt,
                   max_tokens=4000, json_mode=True,
                   tracker=tracker, phase="discover_modules")
    return json.loads(raw)


def _generate_questions_for_module(
    name: str,
    description: str,
    dir_paths: list[str],
    sample_files: list[str],
    model: str = "anthropic/claude-opus-4-5",
    tracker: Optional[TokenTracker] = None,
) -> list[str]:
    """Single LLM call → list of domain-specific questions for one module."""
    prompt = _QUESTIONS_PROMPT.format(
        name=name,
        description=description,
        dirs=", ".join(dir_paths) or "(root)",
        sample_files=", ".join(sample_files[:20]) or "(unknown)",
    )
    try:
        raw = llm_call(model, _QUESTIONS_SYSTEM, prompt,
                       max_tokens=800, json_mode=True,
                       tracker=tracker, phase="generate_questions")
        return json.loads(raw).get("questions", [])
    except Exception as e:
        print(f"  [questions] Failed for {name}: {e}")
        return [f"What are the key responsibilities of the {name} module?"]


def _resolve_module_files(
    codebase: Path,
    all_files: list[Path],
    dir_paths: list[str],
    assigned: set[str],
) -> list[str]:
    """Resolve dir_paths to actual file paths, excluding already-assigned files."""
    from indexer.study_agent import _normalise_dir_path, _relative_to_any

    result: list[str] = []
    for dp in dir_paths:
        ndp = _normalise_dir_path(dp)
        for f in all_files:
            try:
                rel = str(_relative_to_any(f, [codebase]))
            except Exception:
                continue
            if rel in assigned:
                continue
            if ndp == ".":
                if "/" not in rel:
                    result.append(rel)
                    assigned.add(rel)
            elif rel.startswith(ndp + "/") or rel == ndp:
                result.append(rel)
                assigned.add(rel)
    return result


def _collect_files(codebase: Path, language: str = "auto") -> list[Path]:
    """Collect all source files in codebase."""
    from indexer.study_agent import collect_source_files
    try:
        lang = language if language != "auto" else "python"
        return collect_source_files(codebase, language=lang)
    except Exception:
        # Fallback: glob common extensions
        exts = {
            "python": ["*.py"],
            "javascript": ["*.js", "*.ts", "*.jsx", "*.tsx"],
            "rust": ["*.rs"],
            "go": ["*.go"],
            "java": ["*.java"],
            "cpp": ["*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp"],
        }.get(language, ["*.py", "*.js", "*.ts", "*.rs", "*.go", "*.java", "*.cpp"])
        files = []
        for ext in exts:
            files.extend(f for f in codebase.rglob(ext) if not _should_skip(f))
        return files


# ── Public tool: discover_modules ─────────────────────────────────────────────

def discover_modules(
    codebase: str,
    language: str = "auto",
    model: str = "anthropic/claude-opus-4-5",
    output_path: Optional[str] = None,
    docs_context: Optional[str] = None,
    tracker: Optional[TokenTracker] = None,
) -> dict:
    """Discover codebase modules using Claude's direct analysis.

    Claude sees the full directory tree, manifests, READMEs, and index files
    in one shot — much more reliable than the exploration loop.

    Args:
        codebase:     Path to the codebase root directory.
        language:     Programming language hint (auto-detected if omitted).
        model:        LLM model (should be Claude for best results).
        output_path:  Where to write module_map.json. Defaults to
                      <codebase>/../indexer/module_map.json.
        docs_context: Optional additional context (e.g. architecture docs).
        tracker:      Optional token usage tracker.

    Returns:
        dict with keys: project, description, modules (list), output_path
    """
    codebase_path = Path(codebase).resolve()
    if not codebase_path.exists():
        raise ValueError(f"Codebase path does not exist: {codebase_path}")

    out_path = Path(output_path) if output_path else (
        _ROOT / "indexer" / "module_map.json"
    )

    print(f"[discover_modules] Gathering context from {codebase_path} ...")
    context = gather_codebase_context(codebase_path, language)
    if docs_context:
        context = docs_context + "\n\n---\n\n" + context

    print(f"[discover_modules] Asking Claude to identify modules ...")
    raw = _discover_modules_raw(context, model=model, tracker=tracker)

    project = raw.get("project", codebase_path.name)
    description = raw.get("description", "")
    raw_modules = raw.get("modules", [])
    print(f"[discover_modules] Claude identified {len(raw_modules)} modules")

    # Collect all source files
    all_files = _collect_files(codebase_path, language)
    print(f"[discover_modules] {len(all_files)} source files found")

    # Resolve files per module
    assigned: set[str] = set()
    modules = []
    for mod_raw in raw_modules:
        name = mod_raw.get("name", "unknown")
        desc = mod_raw.get("description", "")
        dir_paths = mod_raw.get("dir_paths", [])
        files = _resolve_module_files(codebase_path, all_files, dir_paths, assigned)

        # Generate questions
        print(f"  [{name}] Generating questions ({len(files)} files) ...")
        questions = _generate_questions_for_module(
            name, desc, dir_paths, files[:20],
            model=model, tracker=tracker,
        )

        modules.append({
            "name": name,
            "description": desc,
            "dir_paths": dir_paths,
            "files": files,
            "questions": questions,
        })

    # Assign orphaned files to nearest module by prefix
    unassigned = [
        str(f.relative_to(codebase_path))
        for f in all_files
        if str(f.relative_to(codebase_path)) not in assigned
    ]
    if unassigned:
        print(f"[discover_modules] {len(unassigned)} orphaned files — assigning by prefix")
        _assign_orphans(unassigned, modules)

    module_map = {
        "project": project,
        "description": description,
        "modules": modules,
        "review_issues": [],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(module_map, indent=2))
    print(f"[discover_modules] Written to {out_path}")

    return {
        "project": project,
        "description": description,
        "module_count": len(modules),
        "file_count": len(all_files),
        "output_path": str(out_path),
        "modules": [{"name": m["name"], "description": m["description"],
                     "files": len(m["files"]), "dirs": m["dir_paths"]}
                    for m in modules],
    }


def _assign_orphans(orphans: list[str], modules: list[dict]) -> None:
    """Assign orphaned files to nearest module by longest common path prefix."""
    dir_to_mod: list[tuple[str, int]] = []
    for mi, m in enumerate(modules):
        for dp in m["dir_paths"]:
            dp_norm = dp.strip().lstrip("./").rstrip("/") or "."
            dir_to_mod.append((dp_norm, mi))

    for rel in orphans:
        best_len, best_mod = 0, -1
        for dp, mi in dir_to_mod:
            if rel.startswith(dp + "/") or rel == dp:
                if len(dp) > best_len:
                    best_len, best_mod = len(dp), mi
            else:
                parts_rel = rel.split("/")
                parts_dp = dp.split("/")
                common = sum(1 for a, b in zip(parts_rel, parts_dp) if a == b)
                if common > best_len:
                    best_len, best_mod = common, mi
        if best_mod >= 0:
            modules[best_mod]["files"].append(rel)


# ── Public tool: import_modules ───────────────────────────────────────────────

def import_modules(
    definition_file: str,
    codebase: str,
    language: str = "auto",
    model: str = "anthropic/claude-opus-4-5",
    output_path: Optional[str] = None,
    tracker: Optional[TokenTracker] = None,
) -> dict:
    """Import module definitions from a hand-crafted JSON file and generate questions.

    The JSON file format (e.g. indexer/analysis_modules.json):
    {
      "modules": [
        {"name": "...", "description": "...", "directories": ["src/core"]},
        ...
      ]
    }

    Args:
        definition_file: Path to the module definition JSON.
        codebase:        Path to the codebase root.
        language:        Programming language hint.
        model:           LLM model for question generation.
        output_path:     Where to write module_map.json.
        tracker:         Optional token tracker.

    Returns:
        dict with module map summary and output_path.
    """
    def_path = Path(definition_file).resolve()
    if not def_path.exists():
        raise ValueError(f"Definition file not found: {def_path}")

    codebase_path = Path(codebase).resolve()
    out_path = Path(output_path) if output_path else (
        _ROOT / "indexer" / "module_map.json"
    )

    raw = json.loads(def_path.read_text())
    raw_modules = raw.get("modules", [])
    print(f"[import_modules] Loaded {len(raw_modules)} module definitions from {def_path}")

    all_files = _collect_files(codebase_path, language)
    print(f"[import_modules] {len(all_files)} source files found")

    assigned: set[str] = set()
    modules = []
    for mod_raw in raw_modules:
        name = mod_raw.get("name", "unknown")
        desc = mod_raw.get("description", "")
        # Support both "dir_paths" and "directories" key names
        dir_paths = mod_raw.get("dir_paths") or mod_raw.get("directories", [])

        files = _resolve_module_files(codebase_path, all_files, dir_paths, assigned)

        print(f"  [{name}] {len(files)} files — generating questions ...")
        questions = _generate_questions_for_module(
            name, desc, dir_paths, files[:20],
            model=model, tracker=tracker,
        )

        modules.append({
            "name": name,
            "description": desc,
            "dir_paths": dir_paths,
            "files": files,
            "questions": questions,
        })

    unassigned = [
        str(f.relative_to(codebase_path))
        for f in all_files
        if str(f.relative_to(codebase_path)) not in assigned
    ]
    if unassigned:
        print(f"[import_modules] {len(unassigned)} orphaned files — assigning by prefix")
        _assign_orphans(unassigned, modules)

    module_map = {
        "project": raw.get("project", codebase_path.name),
        "description": raw.get("description", ""),
        "modules": modules,
        "review_issues": [],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(module_map, indent=2))
    print(f"[import_modules] Written to {out_path}")

    return {
        "project": module_map["project"],
        "module_count": len(modules),
        "file_count": len(all_files),
        "output_path": str(out_path),
        "modules": [{"name": m["name"], "files": len(m["files"])} for m in modules],
    }


# ── Public tool: summarize_codebase ──────────────────────────────────────────

def summarize_codebase(
    codebase: str,
    language: str = "auto",
    model: str = None,
    module_map_path: Optional[str] = None,
    output_path: Optional[str] = None,
    rpm: int = 60,
    max_concurrent: int = 50,
    tracker: Optional[TokenTracker] = None,
) -> dict:
    """Run Pass 2 summarization over an existing module map and index into R2R.

    Args:
        codebase:         Path to codebase root.
        language:         Programming language hint.
        model:            LLM model for summarization.
        module_map_path:  Path to module_map.json (default: indexer/module_map.json).
        output_path:      Where to write summaries.json (default: indexer/summaries.json).
        rpm:              API rate limit in requests/minute.
        max_concurrent:   Max concurrent LLM requests.
        tracker:          Optional token tracker.

    Returns:
        dict with summary count, output_path, indexed count.
    """
    from indexer.study_agent import run_pass2, ModuleMap

    codebase_path = Path(codebase).resolve()
    mm_path = Path(module_map_path) if module_map_path else _ROOT / "indexer" / "module_map.json"
    out_path = Path(output_path) if output_path else _ROOT / "indexer" / "summaries.json"
    _model = model or os.environ.get("LLM_MODEL", "openai/gpt-4o")

    if not mm_path.exists():
        raise ValueError(f"module_map.json not found at {mm_path}. Run discover_modules first.")

    module_map = ModuleMap(**json.loads(mm_path.read_text()))
    print(f"[summarize_codebase] {len(module_map.modules)} modules, model={_model}")

    summaries = asyncio.run(run_pass2(
        _model, codebase_path, module_map,
        language=language,
        summaries_path=out_path,
        tracker=tracker,
        rpm=rpm,
        max_concurrent=max_concurrent,
    ))

    out_path.write_text(json.dumps(summaries, indent=2))
    indexed = sum(1 for s in summaries if s.get("doc_id"))
    print(f"[summarize_codebase] {len(summaries)} summaries, {indexed} indexed to R2R")

    return {
        "summary_count": len(summaries),
        "indexed_count": indexed,
        "output_path": str(out_path),
    }
