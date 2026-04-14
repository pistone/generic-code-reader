"""
Shared helpers for MCP server — formatting, logging, staging queue.
Backend-independent.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR     = Path(__file__).parent
# When LOCAL_KB_DIR is set (local workflow), store data files there
# so they live next to the ChromaDB and not inside the package dir.
_data_dir    = Path(os.getenv("LOCAL_KB_DIR", str(BASE_DIR)))
LOG_FILE     = _data_dir / "query_log.jsonl"
STAGING_FILE = Path(os.getenv("STAGING_FILE", str(_data_dir / "staging_queue.json")))
USER_KB_FILE = Path(os.getenv("USER_KB_FILE", str(_data_dir / "user_contributed.jsonl")))

try:
    SEARCH_LIMIT = int(os.getenv("KB_SEARCH_LIMIT", "5"))
except (ValueError, TypeError):
    SEARCH_LIMIT = 5


def log_query(query: str, num_results: int, approx_tokens: int,
              answer: str = "", result_files: list[str] | None = None) -> None:
    """Append one line to the query log for experiment measurement."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "num_results": num_results,
            "approx_tokens": approx_tokens,
            "result_files": result_files or [],
            "answer": answer,
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def format_results(hits: list) -> str:
    """Format search hits into a readable string for Claude."""
    if not hits:
        return "No results found in the knowledge base."

    parts = []
    for i, hit in enumerate(hits):
        meta        = getattr(hit, "metadata", None) or {}
        text        = getattr(hit, "text", "") or ""
        source      = meta.get("source_file", "unknown")
        module      = meta.get("module", "unknown")
        chunk_type  = meta.get("chunk_type", "")
        source_type = meta.get("source_type", "")
        source_kind = meta.get("source_kind", "")
        score       = getattr(hit, "score", 0)

        type_parts = []
        if source_type:
            type_parts.append(source_type)
        if source_kind:
            type_parts.append(source_kind)
        elif chunk_type:
            type_parts.append(chunk_type)
        type_tag = "/".join(type_parts) if type_parts else "unknown"

        header = f"## Result {i+1}  [{type_tag}]  (score: {score:.3f})"
        file_line = f"**File**: `{source}`  |  **Module**: `{module}`"

        if chunk_type == "raw_code":
            section = [header, file_line, "", "**Source code**:", f"```\n{text}\n```"]
        else:
            section = [header, file_line, "", "**Summary**:", text]

        parts.append("\n".join(section))

    return "\n\n---\n\n".join(parts)


def load_staging() -> list:
    if STAGING_FILE.exists():
        try:
            data = json.loads(STAGING_FILE.read_text())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, Exception):
            pass
    return []


def save_staging(queue: list) -> None:
    """Atomic write to prevent corruption from concurrent calls."""
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=STAGING_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(queue, f, indent=2)
        os.replace(tmp_path, STAGING_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def log_user_entry(entry: dict) -> None:
    """Append to JSONL audit log and staging queue."""
    try:
        with USER_KB_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
    queue = load_staging()
    queue.append(entry)
    save_staging(queue)


def get_module_map_text() -> str:
    """Read module_map.json and return formatted module list."""
    # Check default location (full install) and KB data dir (local workflow)
    candidates = [
        Path(__file__).resolve().parent.parent / "indexer" / "module_map.json",
        _data_dir / "module_map.json",
    ]
    module_map_path = next((p for p in candidates if p.exists()), None)
    if module_map_path is None:
        return "No module map found. Run study_agent first."

    try:
        data = json.loads(module_map_path.read_text())
    except Exception as e:
        return f"Could not read module map: {e}"

    project = data.get("project", "Unknown")
    desc = data.get("description", "")
    modules = data.get("modules", [])

    lines = [f"**{project}**: {desc}", f"{len(modules)} modules:\n"]
    for mod in modules:
        name = mod.get("name", "?")
        mod_desc = mod.get("description", "")
        files = mod.get("files", [])
        questions = mod.get("questions", [])
        lines.append(f"- **{name}** ({len(files)} files): {mod_desc}")
        if questions:
            for q in questions[:2]:
                lines.append(f"  - {q}")
            if len(questions) > 2:
                lines.append(f"  - ...and {len(questions) - 2} more questions")

    return "\n".join(lines)


def user_contribution_count() -> int:
    """Count user-contributed entries."""
    if USER_KB_FILE.exists():
        return sum(1 for line in USER_KB_FILE.read_text().splitlines() if line.strip())
    return 0
