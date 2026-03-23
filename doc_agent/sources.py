"""Document source adapters.

Pluggable sources that yield RawDocument objects from different backends.
v1: LocalFileSource (reads from disk).  Future: SharePoint, Confluence, etc.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Protocol


@dataclass
class RawDocument:
    """A document fetched from a source, before parsing."""
    path: str                # relative path from source root
    content_bytes: bytes     # raw file content
    source_type: str         # "local" | "sharepoint" | "confluence" (future)
    last_modified: datetime  # file mtime or API timestamp


class DocumentSource(Protocol):
    """Protocol for pluggable document sources."""
    def list_documents(self) -> Iterator[RawDocument]: ...


DOC_EXTENSIONS = {".md", ".rst", ".txt", ".html", ".htm", ".pdf"}

SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules",
    ".venv", "venv", "env", ".env", "build", "dist",
    ".tox", ".mypy_cache", ".pytest_cache",
}


class LocalFileSource:
    """Reads document files from a local directory tree."""

    def __init__(self, root: Path, extensions: set[str] | None = None):
        self.root = root.resolve()
        self.extensions = extensions or DOC_EXTENSIONS

    def list_documents(self) -> Iterator[RawDocument]:
        for p in sorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.suffix.lower() not in self.extensions:
                continue
            stat = p.stat()
            yield RawDocument(
                path=str(p.relative_to(self.root)),
                content_bytes=p.read_bytes(),
                source_type="local",
                last_modified=datetime.fromtimestamp(stat.st_mtime),
            )
