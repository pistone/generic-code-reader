"""R2R backend for the MCP server."""

import os

R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")


def _client():
    from r2r import R2RClient
    return R2RClient(R2R_URL)


def search(query: str, limit: int) -> list:
    client = _client()
    results = client.retrieval.search(
        query=query,
        search_settings={"limit": limit, "use_hybrid_search": True},
    )
    return results.results.chunk_search_results


def search_by_type(query: str, source_type: str, limit: int = 1) -> list:
    client = _client()
    results = client.retrieval.search(
        query=query,
        search_settings={
            "limit": limit,
            "use_hybrid_search": True,
            "filters": {"source_type": {"$eq": source_type}},
        },
    )
    return results.results.chunk_search_results


def add_entry(text: str, metadata: dict) -> str:
    """Index a document into R2R. Returns doc_id."""
    client = _client()
    resp = client.documents.create(raw_text=text, metadata=metadata)
    return str(resp.results.document_id)


def status() -> list[str]:
    lines = []
    try:
        _client()
        lines.append(f"R2R: healthy ({R2R_URL})")
    except Exception as e:
        lines.append(f"R2R: error ({e})")
        return lines

    client = _client()
    for stype in ["code", "doc", "ticket"]:
        try:
            results = client.retrieval.search(
                query="*",
                search_settings={
                    "limit": 1,
                    "filters": {"source_type": {"$eq": stype}},
                },
            )
            hits = getattr(results, "results", results) if results else []
            chunk_results = getattr(hits, "chunk_search_results", hits) if hits else []
            count = "1+" if chunk_results else "0"
            lines.append(f"  {stype}: {count} entries")
        except Exception:
            lines.append(f"  {stype}: unknown")
    return lines


def error_hint() -> str:
    return f"Check: curl {R2R_URL}/v3/health"
