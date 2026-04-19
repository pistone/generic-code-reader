"""
tools/confluence.py — Confluence page downloader with relevance filtering.

download_confluence: Fetches pages from a Confluence space, saves as Markdown
                     with rich metadata, pre-filters for codebase relevance,
                     and optionally indexes relevant pages into R2R.

Requires:
  CONFLUENCE_URL    Base URL (e.g. https://your-org.atlassian.net)
  CONFLUENCE_EMAIL  Atlassian account email
  CONFLUENCE_TOKEN  Atlassian API token (https://id.atlassian.com/manage/api-tokens)
"""

import base64
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Confluence REST API client ────────────────────────────────────────────────

class ConfluenceClient:
    def __init__(self, base_url: str, email: str, token: str):
        self.base_url = base_url.rstrip("/")
        creds = base64.b64encode(f"{email}:{token}".encode()).decode()
        self._headers = {
            "Authorization": f"Basic {creds}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        import urllib.parse
        import urllib.request

        url = f"{self.base_url}/wiki/rest/api/{path.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    def get_space_pages(self, space_key: str, limit: int = 50) -> list[dict]:
        """List all pages in a space with version info (paginated)."""
        pages = []
        start = 0
        while True:
            resp = self._get(f"space/{space_key}/content/page", {
                "limit": limit,
                "start": start,
                "expand": "title,ancestors,version",
            })
            results = resp.get("results", [])
            pages.extend(results)
            if resp.get("_links", {}).get("next") is None:
                break
            start += limit
        return pages

    def get_page(self, page_id: str) -> dict:
        """Fetch a page with body, metadata, version, and author."""
        return self._get(f"content/{page_id}", {
            "expand": (
                "body.storage,title,ancestors,version,"
                "metadata.labels,history.lastUpdated"
            ),
        })

    def get_space_info(self, space_key: str) -> dict:
        return self._get(f"space/{space_key}")


# ── Storage format → Markdown conversion ──────────────────────────────────────

def _storage_to_markdown(html: str) -> str:
    """Convert Confluence storage format to Markdown."""
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        h.protect_links = True
        return h.handle(html).strip()
    except ImportError:
        md = re.sub(r"<[^>]+>", " ", html)
        for ent, rep in [("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")]:
            md = md.replace(ent, rep)
        return re.sub(r"\s{3,}", "\n\n", md).strip()


def _safe_filename(title: str, page_id: str) -> str:
    """Deterministic filename: <page_id>_<sanitised_title>.md"""
    safe = re.sub(r'[^\w\s-]', '', title).strip()
    safe = re.sub(r'[\s-]+', '_', safe)[:60]
    return f"{page_id}_{safe}.md"


def _build_metadata_header(full: dict, conf_url: str, space: str) -> str:
    """Build a YAML-like metadata block at the top of the Markdown file."""
    page_id  = full.get("id", "")
    title    = full.get("title", "")
    version  = full.get("version", {}).get("number", "?")
    web_url  = f"{conf_url}/wiki/spaces/{space}/pages/{page_id}"

    # Last updated
    last_updated = full.get("history", {}).get("lastUpdated", {})
    modified_by  = last_updated.get("by", {}).get("displayName", "unknown")
    modified_at  = last_updated.get("when", "")
    if modified_at:
        try:
            dt = datetime.fromisoformat(modified_at.replace("Z", "+00:00"))
            modified_at = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass

    # Ancestors (breadcrumb)
    ancestors = full.get("ancestors", [])
    breadcrumb = " > ".join(a.get("title", "") for a in ancestors) + (
        f" > {title}" if ancestors else title
    )

    # Labels
    labels = [
        lbl.get("name", "")
        for lbl in full.get("metadata", {}).get("labels", {}).get("results", [])
    ]
    labels_str = ", ".join(labels) if labels else ""

    lines = [
        "---",
        f"confluence_id: {page_id}",
        f"confluence_url: {web_url}",
        f"space: {space}",
        f"title: {title}",
        f"version: {version}",
        f"last_modified: {modified_at}",
        f"last_modified_by: {modified_by}",
        f"breadcrumb: {breadcrumb}",
    ]
    if labels_str:
        lines.append(f"labels: {labels_str}")
    lines.append(f"downloaded_at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("---")
    return "\n".join(lines)


# ── Manifest (incremental) ────────────────────────────────────────────────────

def _load_manifest(out_dir: Path) -> dict:
    p = out_dir / ".confluence_manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    p = out_dir / ".confluence_manifest.json"
    p.write_text(json.dumps(manifest, indent=2))


# ── Public tool: download_confluence ─────────────────────────────────────────

def download_confluence(
    space: str,
    output_dir: Optional[str] = None,
    page_ids: Optional[list[str]] = None,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    max_pages: int = 500,
    also_index: bool = True,
    model: str = None,
    incremental: bool = True,
) -> dict:
    """Download Confluence pages as Markdown with metadata, optionally index into R2R.

    Each file gets a YAML frontmatter block with: confluence_id, confluence_url,
    space, title, version, last_modified, last_modified_by, breadcrumb, labels,
    downloaded_at. This metadata is preserved when indexed into R2R.

    All pages are downloaded — relevance filtering is left to the caller or agent.
    Confluence spaces tend to be small enough that downloading everything is fine.
    Use include_pattern / exclude_pattern for coarse title-based filtering if needed.

    Args:
        space:           Confluence space key (e.g. "ENG").
        output_dir:      Directory to save .md files
                         (default: doc_agent/confluence/<space>/).
        page_ids:        Specific page IDs to fetch (whole space if omitted).
        include_pattern: Regex: only include pages whose title matches.
        exclude_pattern: Regex: exclude pages whose title matches.
        max_pages:       Maximum pages to download.
        also_index:      Index downloaded pages into R2R immediately.
        model:           LLM model for indexing (if also_index=True).
        incremental:     Skip pages whose Confluence version number hasn't changed
                         since last download.

    Returns:
        dict with page_count, skipped_unchanged, errors, output_dir, files,
        and indexing results if also_index=True.
    """
    conf_url   = os.environ.get("CONFLUENCE_URL", "")
    conf_email = os.environ.get("CONFLUENCE_EMAIL", "")
    conf_token = os.environ.get("CONFLUENCE_TOKEN", "")

    if not all([conf_url, conf_email, conf_token]):
        missing = [k for k, v in {
            "CONFLUENCE_URL": conf_url,
            "CONFLUENCE_EMAIL": conf_email,
            "CONFLUENCE_TOKEN": conf_token,
        }.items() if not v]
        raise EnvironmentError(f"Missing Confluence credentials: {', '.join(missing)}")

    _model = model or os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    client  = ConfluenceClient(conf_url, conf_email, conf_token)

    out_dir = Path(output_dir) if output_dir else (
        _ROOT / "doc_agent" / "confluence" / space.lower()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(out_dir) if incremental else {}
    inc_re = re.compile(include_pattern, re.IGNORECASE) if include_pattern else None
    exc_re = re.compile(exclude_pattern, re.IGNORECASE) if exclude_pattern else None

    if page_ids:
        pages = [{"id": pid, "title": pid, "version": {}} for pid in page_ids]
    else:
        print(f"[confluence] Listing pages in space '{space}' ...")
        pages = client.get_space_pages(space)
        print(f"[confluence] Found {len(pages)} pages")

    pages = pages[:max_pages]

    downloaded: list[str] = []
    skipped_unchanged = 0
    errors = 0

    for page in pages:
        page_id = page["id"]
        title   = page.get("title", page_id)

        if inc_re and not inc_re.search(title):
            continue
        if exc_re and exc_re.search(title):
            continue

        # Incremental: skip if version unchanged
        current_version = page.get("version", {}).get("number")
        if incremental and current_version is not None:
            cached = manifest.get(page_id, {})
            if cached.get("version") == current_version and cached.get("file"):
                if Path(cached["file"]).exists():
                    skipped_unchanged += 1
                    downloaded.append(cached["file"])
                    continue

        try:
            full = client.get_page(page_id)
        except Exception as e:
            print(f"  [{page_id}] Fetch error: {e}")
            errors += 1
            continue

        body_html   = full.get("body", {}).get("storage", {}).get("value", "")
        md_content  = _storage_to_markdown(body_html)
        metadata    = _build_metadata_header(full, conf_url, space)
        version_num = full.get("version", {}).get("number", "?")

        fname   = _safe_filename(title, page_id)
        fpath   = out_dir / fname
        content = f"{metadata}\n\n# {title}\n\n{md_content}"
        fpath.write_text(content, encoding="utf-8")
        downloaded.append(str(fpath))
        manifest[page_id] = {"version": version_num, "file": str(fpath), "title": title}
        _save_manifest(out_dir, manifest)
        print(f"  [{page_id}] {title}")

    print(
        f"\n[confluence] {len(downloaded)} downloaded "
        f"({skipped_unchanged} unchanged/skipped, {errors} errors)"
    )

    result: dict = {
        "page_count": len(downloaded),
        "skipped_unchanged": skipped_unchanged,
        "errors": errors,
        "output_dir": str(out_dir),
        "files": downloaded,
    }

    if also_index and downloaded:
        from tools.docs import index_docs
        idx = index_docs(downloaded, model=_model)
        result["indexing"] = idx

    return result
