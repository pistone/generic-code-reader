"""
tools/confluence.py — Confluence page downloader.

download_confluence: Fetches pages from a Confluence space and saves them
                     as Markdown files. Optionally indexes them into R2R
                     via index_docs.

Requires:
  CONFLUENCE_URL    Base URL of your Confluence instance (e.g. https://your-org.atlassian.net)
  CONFLUENCE_EMAIL  Atlassian account email
  CONFLUENCE_TOKEN  Atlassian API token (https://id.atlassian.com/manage/api-tokens)
"""

import base64
import os
import re
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Confluence REST API client ─────────────────────────────────────────────────

class ConfluenceClient:
    def __init__(self, base_url: str, email: str, token: str):
        import urllib.request
        self.base_url = base_url.rstrip("/")
        creds = base64.b64encode(f"{email}:{token}".encode()).decode()
        self._headers = {
            "Authorization": f"Basic {creds}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._opener = urllib.request.build_opener()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        import json
        import urllib.parse
        import urllib.request

        url = f"{self.base_url}/wiki/rest/api/{path.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())

    def get_space_pages(self, space_key: str, limit: int = 50) -> list[dict]:
        """List all pages in a space (paginated)."""
        pages = []
        start = 0
        while True:
            resp = self._get(f"space/{space_key}/content/page", {
                "limit": limit,
                "start": start,
                "expand": "title,ancestors",
            })
            results = resp.get("results", [])
            pages.extend(results)
            if resp.get("_links", {}).get("next") is None:
                break
            start += limit
        return pages

    def get_page(self, page_id: str) -> dict:
        """Fetch a single page with its storage-format body."""
        return self._get(f"content/{page_id}", {
            "expand": "body.storage,title,ancestors,version",
        })

    def get_child_pages(self, page_id: str) -> list[dict]:
        """Get immediate child pages."""
        resp = self._get(f"content/{page_id}/child/page", {"limit": 50})
        return resp.get("results", [])


# ── Storage format → Markdown conversion ──────────────────────────────────────

def _storage_to_markdown(html: str, title: str = "") -> str:
    """Convert Confluence storage format (HTML-like XML) to Markdown.

    Uses html2text if available, otherwise falls back to basic tag stripping.
    """
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        h.protect_links = True
        md = h.handle(html)
    except ImportError:
        # Fallback: strip tags
        md = re.sub(r"<[^>]+>", " ", html)
        md = re.sub(r"&nbsp;", " ", md)
        md = re.sub(r"&lt;", "<", md)
        md = re.sub(r"&gt;", ">", md)
        md = re.sub(r"&amp;", "&", md)
        md = re.sub(r"\s{3,}", "\n\n", md)

    if title:
        md = f"# {title}\n\n{md.strip()}"
    return md.strip()


def _safe_filename(title: str) -> str:
    """Convert a page title to a safe filename."""
    safe = re.sub(r'[^\w\s-]', '', title).strip()
    safe = re.sub(r'[\s-]+', '_', safe)
    return safe[:80] + ".md"


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
) -> dict:
    """Download Confluence pages as Markdown files, optionally index into R2R.

    Args:
        space:           Confluence space key (e.g. "ENG", "ARCH").
        output_dir:      Directory to save .md files (default: doc_agent/confluence/<space>/).
        page_ids:        Specific page IDs to fetch (fetches whole space if omitted).
        include_pattern: Regex: only include pages whose title matches.
        exclude_pattern: Regex: exclude pages whose title matches.
        max_pages:       Maximum pages to download.
        also_index:      If True, run index_docs on downloaded files after download.
        model:           LLM model for indexing (if also_index=True).

    Returns:
        dict with page_count, output_dir, and indexing results if also_index=True.
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

    client = ConfluenceClient(conf_url, conf_email, conf_token)

    out_dir = Path(output_dir) if output_dir else (
        _ROOT / "doc_agent" / "confluence" / space.lower()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    inc_re = re.compile(include_pattern) if include_pattern else None
    exc_re = re.compile(exclude_pattern) if exclude_pattern else None

    # Collect pages to download
    if page_ids:
        pages = [{"id": pid, "title": pid} for pid in page_ids]
    else:
        print(f"[confluence] Listing pages in space '{space}' ...")
        pages = client.get_space_pages(space)
        print(f"[confluence] Found {len(pages)} pages")

    pages = pages[:max_pages]

    downloaded = []
    skipped = 0
    for page in pages:
        page_id = page["id"]
        title = page.get("title", page_id)

        if inc_re and not inc_re.search(title):
            skipped += 1
            continue
        if exc_re and exc_re.search(title):
            skipped += 1
            continue

        try:
            full = client.get_page(page_id)
            body_html = full.get("body", {}).get("storage", {}).get("value", "")
            md = _storage_to_markdown(body_html, title=title)
            fname = _safe_filename(title)
            fpath = out_dir / fname
            fpath.write_text(md, encoding="utf-8")
            downloaded.append(str(fpath))
            print(f"  [{page_id}] {title} → {fname}")
        except Exception as e:
            print(f"  [{page_id}] Error: {e}")

    print(f"[confluence] Downloaded {len(downloaded)} pages, skipped {skipped}")

    result: dict = {
        "page_count": len(downloaded),
        "skipped": skipped,
        "output_dir": str(out_dir),
        "files": downloaded,
    }

    if also_index and downloaded:
        from tools.docs import index_docs
        idx = index_docs(downloaded, model=model)
        result["indexing"] = idx

    return result
