#!/usr/bin/env python3
"""Cross-reference auditor: detect conflicts between docs and code in R2R.

Compares doc_agent entries against study_agent entries to find
contradictions caused by stale documentation.  Uses timestamps as a
cheap first pass, then LLM comparison only on flagged pairs.

Outputs a conflict report for human review — does NOT auto-fix.

Usage:
    python -m auditor.auditor
    python -m auditor.auditor --threshold-days 60 --model openai/gpt-4o
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, Generator

import litellm
from r2r import R2RClient

R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")
DEFAULT_THRESHOLD_DAYS = 90
MIN_SIMILARITY_SCORE = 0.3


# ---------------------------------------------------------------------------
# TokenTracker (same pattern as study_agent / reviewer_agent)
# ---------------------------------------------------------------------------

class TokenTracker:
    """Accumulates prompt/completion token counts per phase."""

    def __init__(self):
        self.phases: dict[str, dict[str, int]] = {}

    def _ensure_phase(self, phase: str) -> dict[str, int]:
        if phase not in self.phases:
            self.phases[phase] = {"prompt": 0, "completion": 0, "calls": 0}
        return self.phases[phase]

    def record(self, phase: str, response) -> None:
        p = self._ensure_phase(phase)
        usage = getattr(response, "usage", None)
        if usage:
            p["prompt"] += getattr(usage, "prompt_tokens", 0)
            p["completion"] += getattr(usage, "completion_tokens", 0)
        p["calls"] += 1

    def summary(self) -> str:
        lines = []
        total_p = total_c = 0
        for phase, counts in self.phases.items():
            lines.append(
                f"[Tokens] {phase}: {counts['prompt']:,} prompt "
                f"+ {counts['completion']:,} completion  "
                f"({counts['calls']} call(s))"
            )
            total_p += counts["prompt"]
            total_c += counts["completion"]
        lines.append(
            f"[Tokens] Total: {total_p:,} prompt + {total_c:,} completion "
            f"= {total_p + total_c:,} tokens"
        )
        return "\n".join(lines)

    def to_log_entry(self, model: str) -> dict:
        total = sum(p["prompt"] + p["completion"]
                    for p in self.phases.values())
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "agent": "auditor",
            "phases": dict(self.phases),
            "total_tokens": total,
        }


# ---------------------------------------------------------------------------
# LLM call (same pattern as study_agent)
# ---------------------------------------------------------------------------

def llm_call(model: str, system: str, user: str,
             max_tokens: int = 1024,
             json_mode: bool = False,
             tracker: Optional[TokenTracker] = None,
             phase: str = "") -> str:
    """Call LLM via litellm.  Returns response text."""
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = litellm.completion(**kwargs)
    text = response.choices[0].message.content

    if tracker and phase:
        tracker.record(phase, response)

    return text


# ---------------------------------------------------------------------------
# R2R helpers
# ---------------------------------------------------------------------------

def collect_doc_entries(client: R2RClient) -> list[dict]:
    """List all doc_agent entries from R2R.

    Returns list of dicts with keys:
      doc_id, source_file, doc_title, last_modified, text
    """
    docs = []
    try:
        result = client.documents.list(limit=1000)
        for doc in result.results:
            meta = doc.metadata or {}
            if meta.get("source_type") != "doc":
                continue
            if meta.get("chunk_type") != "doc_summary":
                continue
            # Retrieve the document text via search (list doesn't include text)
            try:
                search_resp = client.retrieval.search(
                    query=meta.get("doc_title", meta.get("source_file", "")),
                    search_settings={
                        "limit": 1,
                        "filters": {"document_id": {"$eq": str(doc.id)}},
                    },
                )
                hits = search_resp.results.chunk_search_results
                text = hits[0].text if hits else ""
            except Exception:
                text = ""

            docs.append({
                "doc_id": str(doc.id),
                "source_file": meta.get("source_file", ""),
                "doc_title": meta.get("doc_title", ""),
                "last_modified": meta.get("last_modified", ""),
                "text": text,
            })
    except Exception as e:
        print(f"[error] Failed to list documents: {e}")
    return docs


def find_related_code(client: R2RClient, doc_text: str,
                      limit: int = 3) -> list[dict]:
    """Search R2R for code entries related to a doc excerpt.

    Returns list of dicts with keys:
      source_file, module, summary, score
    """
    if not doc_text:
        return []

    query = doc_text[:200]
    try:
        results = client.retrieval.search(
            query=query,
            search_settings={
                "limit": limit + 5,  # fetch extra, filter below
                "use_hybrid_search": True,
            },
        )
        hits = results.results.chunk_search_results
    except Exception:
        return []

    code_hits = []
    for hit in hits:
        meta = hit.metadata or {}
        # Skip doc entries — we only want code
        if meta.get("source_type") == "doc":
            continue
        if meta.get("chunk_type") == "doc_summary":
            continue
        score = getattr(hit, "score", 0)
        if score < MIN_SIMILARITY_SCORE:
            continue
        code_hits.append({
            "source_file": meta.get("source_file", ""),
            "module": meta.get("module", ""),
            "summary": (hit.text or "")[:500],
            "score": score,
        })
        if len(code_hits) >= limit:
            break

    return code_hits


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

def parse_iso_date(s: str) -> Optional[datetime]:
    """Best-effort parse of an ISO timestamp string."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def check_timestamp_staleness(doc_last_modified: str,
                              threshold_days: int) -> bool:
    """Return True if doc is older than threshold_days from now."""
    doc_dt = parse_iso_date(doc_last_modified)
    if doc_dt is None:
        # No timestamp → can't determine, flag conservatively
        return True
    now = datetime.now(tz=doc_dt.tzinfo if doc_dt.tzinfo else None)
    age_days = (now - doc_dt).days
    return age_days > threshold_days


# ---------------------------------------------------------------------------
# LLM comparison
# ---------------------------------------------------------------------------

COMPARE_SYSTEM = (
    "You compare documentation with code summaries to detect contradictions. "
    "Output ONLY valid JSON — no markdown fences, no commentary."
)


def build_compare_prompt(doc_file: str, doc_text: str,
                         code_file: str, code_summary: str) -> str:
    return f"""Compare this documentation with this code summary.

## Documentation (from {doc_file})
{doc_text[:800]}

## Code Summary (from {code_file})
{code_summary[:800]}

Are they consistent?  Answer with this JSON:
{{
  "consistent": true or false,
  "explanation": "one-sentence explanation if inconsistent, empty string if consistent"
}}"""


def llm_compare(model: str, doc_file: str, doc_text: str,
                code_file: str, code_summary: str,
                tracker: Optional[TokenTracker] = None) -> dict:
    """Use LLM to check if a doc and code summary are consistent."""
    prompt = build_compare_prompt(doc_file, doc_text,
                                  code_file, code_summary)
    raw = llm_call(model, COMPARE_SYSTEM, prompt,
                   max_tokens=256, json_mode=True,
                   tracker=tracker, phase="LLM comparison")
    try:
        result = json.loads(raw)
        return {
            "consistent": bool(result.get("consistent", True)),
            "explanation": str(result.get("explanation", "")),
        }
    except (json.JSONDecodeError, TypeError):
        return {"consistent": True, "explanation": ""}


# ---------------------------------------------------------------------------
# Main audit pipeline
# ---------------------------------------------------------------------------

def run_audit(model: str, threshold_days: int,
              tracker: Optional[TokenTracker] = None) -> dict:
    """Run the full audit pipeline.  Returns the conflict report dict."""
    client = R2RClient(R2R_URL)

    # Step 1: collect doc entries
    print("[Audit] Collecting doc entries from R2R...")
    doc_entries = collect_doc_entries(client)
    print(f"[Audit] Found {len(doc_entries)} doc entries")

    if not doc_entries:
        print("[Audit] No doc entries found. Nothing to audit.")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "total_doc_entries": 0,
            "pairs_checked_by_timestamp": 0,
            "stale_suspects": 0,
            "llm_comparisons": 0,
            "conflicts_found": 0,
            "conflicts": [],
        }

    stale_suspects: list[dict] = []
    total_pairs = 0

    # Step 2-3: for each doc, find related code and check timestamps
    print(f"[Audit] Checking doc↔code pairs (threshold: {threshold_days} days)...")
    for doc in doc_entries:
        related = find_related_code(client, doc["text"])
        if not related:
            continue

        total_pairs += len(related)
        is_stale = check_timestamp_staleness(doc["last_modified"],
                                             threshold_days)
        if is_stale:
            for code_hit in related:
                stale_suspects.append({
                    "doc": doc,
                    "code": code_hit,
                })

        time.sleep(0.1)  # gentle rate limit on R2R

    print(f"[Audit] {total_pairs} doc↔code pairs checked, "
          f"{len(stale_suspects)} stale suspect(s)")

    # Step 4: LLM comparison on stale suspects only
    conflicts: list[dict] = []
    if stale_suspects:
        print(f"[Audit] Running LLM comparison on {len(stale_suspects)} "
              f"suspect pair(s)...")
        for pair in stale_suspects:
            doc = pair["doc"]
            code = pair["code"]

            result = llm_compare(
                model,
                doc_file=doc["source_file"],
                doc_text=doc["text"],
                code_file=code["source_file"],
                code_summary=code["summary"],
                tracker=tracker,
            )

            if not result["consistent"]:
                conflicts.append({
                    "doc_file": doc["source_file"],
                    "doc_title": doc["doc_title"],
                    "doc_excerpt": doc["text"][:300],
                    "doc_last_modified": doc["last_modified"],
                    "code_file": code["source_file"],
                    "code_summary": code["summary"][:300],
                    "similarity_score": code["score"],
                    "explanation": result["explanation"],
                })

            time.sleep(0.3)  # rate limit

    # Step 5: build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "threshold_days": threshold_days,
        "total_doc_entries": len(doc_entries),
        "pairs_checked_by_timestamp": total_pairs,
        "stale_suspects": len(stale_suspects),
        "llm_comparisons": len(stale_suspects),
        "conflicts_found": len(conflicts),
        "conflicts": conflicts,
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Cross-reference auditor: detect doc↔code conflicts in R2R",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="litellm model string (default: %(default)s)")
    parser.add_argument("--threshold-days", type=int,
                        default=DEFAULT_THRESHOLD_DAYS,
                        help="Flag docs older than this many days "
                             "(default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="Path for conflict report JSON "
                             "(default: auditor/conflict_report.json)")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent
    output_path = (Path(args.output) if args.output
                   else output_dir / "conflict_report.json")
    cost_log_path = output_dir / "cost_log.jsonl"

    tracker = TokenTracker()

    # Run audit
    report = run_audit(args.model, args.threshold_days, tracker=tracker)

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Audit complete")
    print(f"  Doc entries scanned:     {report['total_doc_entries']}")
    print(f"  Doc↔code pairs checked:  {report['pairs_checked_by_timestamp']}")
    print(f"  Stale suspects:          {report['stale_suspects']}")
    print(f"  LLM comparisons:         {report['llm_comparisons']}")
    print(f"  Conflicts found:         {report['conflicts_found']}")
    print(f"{'=' * 60}")

    if report["conflicts"]:
        print(f"\nConflicts:")
        for i, c in enumerate(report["conflicts"], 1):
            print(f"\n  {i}. {c['doc_file']} ↔ {c['code_file']}")
            print(f"     {c['explanation']}")

    print(f"\nFull report: {output_path}")

    # Token tracking
    if tracker.phases:
        print(f"\n{tracker.summary()}")
        entry = tracker.to_log_entry(model=args.model)
        with cost_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
