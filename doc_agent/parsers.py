"""File-type parsers for document ingestion.

Each parser converts raw bytes into a ParsedDocument with section structure.
Section-aware splitting enables heading-level chunking downstream.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser as _StdlibHTMLParser
from pathlib import Path
from typing import Optional


@dataclass
class Section:
    """A heading-delimited section within a document."""
    heading: str    # e.g. "Architecture Overview", or "" for preamble
    content: str    # text content (without the heading line itself)
    level: int      # 0=preamble, 1=h1, 2=h2, 3=h3, etc.


@dataclass
class ParsedDocument:
    """A parsed document ready for chunking."""
    title: str
    text: str                                       # full plain text
    sections: list[Section] = field(default_factory=list)
    source_file: str = ""
    last_modified: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class MarkdownParser:
    """Parse .md files: strip YAML frontmatter, split into sections."""

    def parse(self, path: str, content: bytes,
              last_modified: datetime) -> ParsedDocument:
        text = content.decode("utf-8", errors="replace")
        text = self._strip_frontmatter(text)
        sections = self._split_sections(text)
        title = self._extract_title(sections, path)
        return ParsedDocument(
            title=title, text=text, sections=sections,
            source_file=path, last_modified=last_modified,
        )

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Remove YAML frontmatter delimited by --- ... ---"""
        if not text.startswith("---"):
            return text
        end = text.find("\n---", 3)
        if end == -1:
            return text
        # skip past the closing --- and any trailing newline
        after = end + 4
        if after < len(text) and text[after] == "\n":
            after += 1
        return text[after:]

    @staticmethod
    def _split_sections(text: str) -> list[Section]:
        """Split at markdown heading lines (^#{1,6} )."""
        parts = re.split(r"\n(?=#{1,6}\s)", text)
        sections: list[Section] = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^(#{1,6})\s+(.*)", part)
            if m:
                level = len(m.group(1))
                heading = m.group(2).strip()
                content = part[m.end():].strip()
            else:
                level = 0
                heading = ""
                content = part
            sections.append(Section(heading=heading, content=content,
                                    level=level))
        return sections

    @staticmethod
    def _extract_title(sections: list[Section], fallback_path: str) -> str:
        """Return the first heading, or filename stem as fallback."""
        for s in sections:
            if s.level >= 1 and s.heading:
                return s.heading
        return Path(fallback_path).stem.replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

class _TextExtractor(_StdlibHTMLParser):
    """Minimal HTML-to-text + heading extractor using stdlib."""

    def __init__(self):
        super().__init__()
        self.sections: list[Section] = []
        self._current_heading: Optional[str] = None
        self._current_level: int = 0
        self._buf: list[str] = []
        self._in_heading: bool = False
        self._heading_buf: list[str] = []
        self._skip_tags = {"script", "style", "noscript"}
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        m = re.match(r"^h([1-6])$", tag)
        if m:
            # flush previous section
            self._flush_section()
            self._in_heading = True
            self._current_level = int(m.group(1))
            self._heading_buf = []

    def handle_endtag(self, tag: str):
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._in_heading and re.match(r"^h[1-6]$", tag):
            self._current_heading = " ".join(self._heading_buf).strip()
            self._heading_buf = []
            self._in_heading = False

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        if self._in_heading:
            self._heading_buf.append(data)
        else:
            self._buf.append(data)

    def _flush_section(self):
        content = " ".join(self._buf).strip()
        if content or self._current_heading:
            self.sections.append(Section(
                heading=self._current_heading or "",
                content=content,
                level=self._current_level,
            ))
        self._buf = []

    def finish(self):
        self._flush_section()


class HTMLParser:
    """Parse .html files: strip tags, extract headings as sections."""

    def parse(self, path: str, content: bytes,
              last_modified: datetime) -> ParsedDocument:
        text = content.decode("utf-8", errors="replace")
        extractor = _TextExtractor()
        extractor.feed(text)
        extractor.finish()

        plain_text = "\n\n".join(
            f"{s.heading}\n{s.content}" if s.heading else s.content
            for s in extractor.sections
        )
        title = self._extract_title(extractor.sections, path)
        return ParsedDocument(
            title=title, text=plain_text, sections=extractor.sections,
            source_file=path, last_modified=last_modified,
        )

    @staticmethod
    def _extract_title(sections: list[Section], fallback_path: str) -> str:
        for s in sections:
            if s.level >= 1 and s.heading:
                return s.heading
        return Path(fallback_path).stem.replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class PDFParser:
    """Parse .pdf files using pymupdf (fitz).  Raises RuntimeError if
    pymupdf is not installed (binary PDF bytes cannot be treated as text)."""

    def parse(self, path: str, content: bytes,
              last_modified: datetime) -> ParsedDocument:
        try:
            import pymupdf  # noqa: F811
        except ImportError:
            raise RuntimeError(
                f"Cannot parse PDF '{path}': pymupdf not installed. "
                "Install with: pip install pymupdf"
            )

        doc = pymupdf.Document(stream=content, filetype="pdf")
        sections: list[Section] = []
        full_text_parts: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if not text:
                continue
            heading = f"Page {page_num + 1}"
            sections.append(Section(heading=heading, content=text,
                                    level=1))
            full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)
        title = (doc.metadata.get("title") or "").strip()
        if not title:
            title = Path(path).stem.replace("_", " ").replace("-", " ").title()

        doc.close()
        return ParsedDocument(
            title=title, text=full_text, sections=sections,
            source_file=path, last_modified=last_modified,
        )


# ---------------------------------------------------------------------------
# Plain text / RST
# ---------------------------------------------------------------------------

class PlainTextParser:
    """Parse .txt and .rst files as a single section."""

    def parse(self, path: str, content: bytes,
              last_modified: datetime) -> ParsedDocument:
        text = content.decode("utf-8", errors="replace")
        title = Path(path).stem.replace("_", " ").replace("-", " ").title()
        sections = [Section(heading="", content=text.strip(), level=0)]
        return ParsedDocument(
            title=title, text=text, sections=sections,
            source_file=path, last_modified=last_modified,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MARKDOWN = MarkdownParser()
_HTML = HTMLParser()
_PDF = PDFParser()
_PLAIN = PlainTextParser()

PARSER_MAP = {
    ".md":   _MARKDOWN,
    ".html": _HTML,
    ".htm":  _HTML,
    ".pdf":  _PDF,
    ".txt":  _PLAIN,
    ".rst":  _PLAIN,
}


def get_parser(suffix: str):
    """Return the appropriate parser for a file extension."""
    return PARSER_MAP.get(suffix.lower(), _PLAIN)
