"""
Work directory configuration for generated artifacts.

All generated artifacts (summaries, hashes, tickets, logs, etc.) go to a
work directory to keep the repository clean and avoid committing proprietary
target-codebase data.

Usage:
    from codebase_shared.work_dir import get_work_dir, ensure_work_dirs

    # Get the work directory path
    work = get_work_dir()  # Returns Path object

    # Ensure subdirectories exist
    ensure_work_dirs()

    # Get specific subdirectory
    work / "indexer" / "summaries.json"
    work / "ticket_agent" / "tickets"

Configuration:
    Set WORK_DIR environment variable to customize location.
    Default: ./work/ (relative to current working directory)
"""

import os
from pathlib import Path


def get_work_dir() -> Path:
    """Get the work directory path from environment or default."""
    work_dir = os.environ.get("WORK_DIR", "work")
    return Path(work_dir).resolve()


def ensure_work_dirs() -> Path:
    """Create work directory structure if it doesn't exist.

    Returns the work directory path.
    """
    work = get_work_dir()

    # Create subdirectories
    subdirs = [
        "indexer",
        "doc_agent",
        "ticket_agent/tickets",
        "ticket_agent/lessons",
        "mcp_server",
        "reviewer",
        "auditor",
        "logs",
    ]

    for subdir in subdirs:
        (work / subdir).mkdir(parents=True, exist_ok=True)

    return work


def get_indexer_dir() -> Path:
    """Get indexer output directory."""
    return get_work_dir() / "indexer"


def get_doc_agent_dir() -> Path:
    """Get doc_agent output directory."""
    return get_work_dir() / "doc_agent"


def get_ticket_agent_dir() -> Path:
    """Get ticket_agent output directory."""
    return get_work_dir() / "ticket_agent"


def get_logs_dir() -> Path:
    """Get logs directory."""
    return get_work_dir() / "logs"
