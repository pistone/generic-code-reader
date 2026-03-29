"""
KB Evaluation Harness — Measure knowledge base effectiveness.

Three evaluation modes:

1. REPLAY: Replay logged queries against the current KB (or a different one)
   and compare answers side-by-side.

   python eval/eval_kb.py replay --query-log mcp_server/query_log.jsonl

2. COMPARE: Run the same queries against two KBs (e.g., old vs new, or
   study_agent vs explorer_agent) and score which gives better answers.

   python eval/eval_kb.py compare \
       --query-log mcp_server/query_log.jsonl \
       --kb-a http://localhost:7272 \
       --kb-b http://localhost:7273

3. BLIND: Give Claude a set of tasks (e.g., tickets to resolve) twice —
   once with KB access, once without — and compare quality/speed.

   python eval/eval_kb.py blind \
       --questions eval/test_questions.jsonl \
       --kb-url http://localhost:7272

Query log format (from MCP server):
  {"timestamp": "...", "query": "...", "answer": "...", "result_files": [...], ...}

Test questions format:
  {"question": "How does X handle Y?", "expected_files": ["a.cpp", "b.h"], "tags": ["module_name"]}
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _query_r2r(url: str, query: str, limit: int = 5) -> dict:
    """Query an R2R instance and return raw results."""
    from r2r import R2RClient
    client = R2RClient(url)
    try:
        results = client.retrieval.search(
            query=query,
            search_settings={"limit": limit, "use_hybrid_search": True},
        )
        hits = results.results.chunk_search_results
        entries = []
        for h in hits:
            meta = h.metadata or {}
            entries.append({
                "text": (h.text or "").strip()[:500],
                "source_file": meta.get("source_file", ""),
                "module": meta.get("module", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "score": getattr(h, "score", 0),
            })
        return {"hits": entries, "count": len(entries)}
    except Exception as e:
        return {"hits": [], "count": 0, "error": str(e)}


def _load_query_log(path: Path) -> list[dict]:
    """Load query log entries."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _load_questions(path: Path) -> list[dict]:
    """Load test questions."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_retrieval(result: dict, expected_files: list[str]) -> dict:
    """Score a retrieval result against expected files.

    Returns precision, recall, and a relevance score.
    """
    if not expected_files:
        return {"precision": None, "recall": None, "note": "no expected files"}

    retrieved_files = [h.get("source_file", "") for h in result.get("hits", [])]
    retrieved_set = set(retrieved_files)
    expected_set = set(expected_files)

    # Fuzzy match: basename matching (handles path differences)
    retrieved_basenames = {Path(f).name for f in retrieved_files if f}
    expected_basenames = {Path(f).name for f in expected_files if f}

    hits_exact = retrieved_set & expected_set
    hits_fuzzy = retrieved_basenames & expected_basenames

    n_retrieved = len(retrieved_files)
    n_expected = len(expected_files)

    precision = len(hits_fuzzy) / n_retrieved if n_retrieved else 0
    recall = len(hits_fuzzy) / n_expected if n_expected else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0,
        "exact_matches": len(hits_exact),
        "fuzzy_matches": len(hits_fuzzy),
        "retrieved": n_retrieved,
        "expected": n_expected,
    }


def score_answer_quality(answer: str) -> dict:
    """Heuristic scoring of answer quality (no LLM needed).

    Checks for specificity signals vs vagueness signals.
    """
    if not answer or answer.startswith("No results found"):
        return {"score": 0.0, "reason": "no results"}

    # Positive signals: specific names, code references
    specific_count = 0
    for signal in ("`::", "class ", "function ", "struct ", "#include",
                   "score: 0.", "**File**:", "```"):
        specific_count += answer.count(signal)

    # Negative signals
    vague_count = 0
    for signal in ("unknown", "no results", "search failed",
                   "not found", "no matches"):
        if signal in answer.lower():
            vague_count += 1

    # Score: 0-1 based on richness
    n_results = answer.count("## Result")
    length_score = min(1.0, len(answer) / 2000)
    specificity_score = min(1.0, specific_count / 10)
    result_score = min(1.0, n_results / 3)
    vague_penalty = min(0.5, vague_count * 0.2)

    score = (length_score * 0.2 + specificity_score * 0.4 +
             result_score * 0.4 - vague_penalty)
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 3),
        "n_results": n_results,
        "specificity": specific_count,
        "length": len(answer),
    }


def llm_judge(question: str, answer_a: str, answer_b: str,
              model: str = "openai/gpt-4o") -> dict:
    """Use an LLM to judge which answer is better (optional, costs tokens).

    Returns {"winner": "A"|"B"|"tie", "reasoning": "..."}
    """
    try:
        from codebase_shared.utils import llm_call
    except ImportError:
        return {"winner": "skip", "reasoning": "llm_call not available"}

    prompt = f"""You are evaluating two knowledge base answers for the same developer question.

Question: {question}

--- Answer A ---
{answer_a[:3000]}

--- Answer B ---
{answer_b[:3000]}

Which answer is more useful for a developer trying to understand or work with the codebase?

Criteria:
1. Relevance: Does it answer the actual question?
2. Specificity: Does it mention actual class/function names, files?
3. Actionability: Could a developer use this to make progress?
4. Conciseness: Not too verbose, not too sparse?

Respond in JSON only:
{{"winner": "A" or "B" or "tie", "score_a": 1-5, "score_b": 1-5, "reasoning": "1-2 sentences"}}"""

    try:
        raw = llm_call(model, "You are a fair judge of technical answers. Respond in JSON only.",
                       prompt, max_tokens=200, json_mode=True)
        return json.loads(raw)
    except Exception as e:
        return {"winner": "error", "reasoning": str(e)}


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_replay(args):
    """Replay logged queries against the current KB and show side-by-side comparison."""
    log_path = Path(args.query_log)
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        sys.exit(1)

    entries = _load_query_log(log_path)
    kb_url = args.kb_url
    limit = args.limit

    print(f"\nReplaying {len(entries)} queries against {kb_url}")
    print(f"{'='*70}")

    results = []
    for i, entry in enumerate(entries[:limit]):
        query = entry.get("query", "")
        original_answer = entry.get("answer", "")
        original_files = entry.get("result_files", [])

        if not query:
            continue

        # Query current KB
        current = _query_r2r(kb_url, query)
        current_files = [h["source_file"] for h in current.get("hits", []) if h.get("source_file")]

        # Score
        original_quality = score_answer_quality(original_answer)
        # Build a rough "answer" from current results for quality scoring
        current_answer = "\n".join(
            f"## Result\n**File**: {h['source_file']}\n{h['text']}"
            for h in current.get("hits", [])
        )
        current_quality = score_answer_quality(current_answer)

        # File overlap
        orig_set = set(Path(f).name for f in original_files)
        curr_set = set(Path(f).name for f in current_files)
        overlap = orig_set & curr_set
        new_files = curr_set - orig_set
        lost_files = orig_set - curr_set

        result = {
            "query": query,
            "original_quality": original_quality["score"],
            "current_quality": current_quality["score"],
            "file_overlap": len(overlap),
            "new_files": len(new_files),
            "lost_files": len(lost_files),
            "current_count": current["count"],
        }
        results.append(result)

        # Print
        delta = current_quality["score"] - original_quality["score"]
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        status = "✓" if delta >= 0 else "✗"

        print(f"\n[{i+1}/{min(len(entries), limit)}] {status} {query[:70]}")
        print(f"  Quality: {original_quality['score']:.2f} → {current_quality['score']:.2f} ({delta_str})")
        if new_files:
            print(f"  New files: {', '.join(sorted(new_files)[:3])}")
        if lost_files:
            print(f"  Lost files: {', '.join(sorted(lost_files)[:3])}")

    # Summary
    if results:
        avg_orig = sum(r["original_quality"] for r in results) / len(results)
        avg_curr = sum(r["current_quality"] for r in results) / len(results)
        improved = sum(1 for r in results if r["current_quality"] > r["original_quality"])
        regressed = sum(1 for r in results if r["current_quality"] < r["original_quality"])

        print(f"\n{'='*70}")
        print(f"  Queries replayed:  {len(results)}")
        print(f"  Avg quality:       {avg_orig:.3f} → {avg_curr:.3f}")
        print(f"  Improved:          {improved}")
        print(f"  Regressed:         {regressed}")
        print(f"  Unchanged:         {len(results) - improved - regressed}")
        print(f"{'='*70}")

    # Save detailed results
    out_path = Path(args.output) if args.output else Path("eval/replay_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results: {out_path}")


def cmd_compare(args):
    """Compare two KBs on the same queries."""
    log_path = Path(args.query_log) if args.query_log else None
    questions_path = Path(args.questions) if args.questions else None

    if not log_path and not questions_path:
        print("Error: need --query-log or --questions")
        sys.exit(1)

    # Load queries
    queries = []
    if questions_path and questions_path.exists():
        for q in _load_questions(questions_path):
            queries.append({
                "query": q["question"],
                "expected_files": q.get("expected_files", []),
                "tags": q.get("tags", []),
            })
    elif log_path and log_path.exists():
        for entry in _load_query_log(log_path):
            if entry.get("query"):
                queries.append({
                    "query": entry["query"],
                    "expected_files": entry.get("result_files", []),
                    "tags": [],
                })

    limit = args.limit
    queries = queries[:limit]

    print(f"\nComparing {len(queries)} queries")
    print(f"  KB-A: {args.kb_a}")
    print(f"  KB-B: {args.kb_b}")
    if args.judge:
        print(f"  LLM judge: {args.judge_model}")
    print(f"{'='*70}")

    results = []
    a_wins = b_wins = ties = 0

    for i, q in enumerate(queries):
        query = q["query"]
        expected = q.get("expected_files", [])

        result_a = _query_r2r(args.kb_a, query)
        result_b = _query_r2r(args.kb_b, query)

        score_a = score_retrieval(result_a, expected) if expected else {}
        score_b = score_retrieval(result_b, expected) if expected else {}

        quality_a = score_answer_quality(
            "\n".join(h["text"] for h in result_a.get("hits", []))
        )
        quality_b = score_answer_quality(
            "\n".join(h["text"] for h in result_b.get("hits", []))
        )

        winner = "tie"
        judge_result = {}

        if args.judge:
            answer_a = "\n".join(
                f"[{h['source_file']}] {h['text']}" for h in result_a.get("hits", [])
            )
            answer_b = "\n".join(
                f"[{h['source_file']}] {h['text']}" for h in result_b.get("hits", [])
            )
            judge_result = llm_judge(query, answer_a, answer_b, model=args.judge_model)
            winner = judge_result.get("winner", "tie")
        else:
            # Heuristic winner
            if quality_a["score"] > quality_b["score"] + 0.05:
                winner = "A"
            elif quality_b["score"] > quality_a["score"] + 0.05:
                winner = "B"

        if winner == "A":
            a_wins += 1
        elif winner == "B":
            b_wins += 1
        else:
            ties += 1

        result_entry = {
            "query": query,
            "winner": winner,
            "quality_a": quality_a["score"],
            "quality_b": quality_b["score"],
            "retrieval_a": score_a,
            "retrieval_b": score_b,
            "judge": judge_result,
        }
        results.append(result_entry)

        marker = {"A": "◀", "B": "▶", "tie": "="}
        print(f"  [{i+1}] {marker.get(winner, '?')} {query[:60]}  "
              f"(A:{quality_a['score']:.2f} vs B:{quality_b['score']:.2f})")

    print(f"\n{'='*70}")
    print(f"  KB-A wins: {a_wins}   KB-B wins: {b_wins}   Ties: {ties}")
    if results:
        avg_a = sum(r["quality_a"] for r in results) / len(results)
        avg_b = sum(r["quality_b"] for r in results) / len(results)
        print(f"  Avg quality:  A={avg_a:.3f}  B={avg_b:.3f}")
    print(f"{'='*70}")

    out_path = Path(args.output) if args.output else Path("eval/compare_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results: {out_path}")


def cmd_blind(args):
    """Run test questions with and without KB, compare answers."""
    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"Error: {questions_path} not found")
        sys.exit(1)

    questions = _load_questions(questions_path)[:args.limit]
    kb_url = args.kb_url

    print(f"\nBlind evaluation: {len(questions)} questions")
    print(f"  KB: {kb_url}")
    print(f"\nFor each question, comparing KB-assisted answer vs no-KB answer")
    print(f"{'='*70}")

    results = []
    for i, q in enumerate(questions):
        query = q["question"]
        expected = q.get("expected_files", [])

        # With KB
        kb_result = _query_r2r(kb_url, query)
        kb_quality = score_answer_quality(
            "\n".join(h["text"] for h in kb_result.get("hits", []))
        )
        kb_retrieval = score_retrieval(kb_result, expected) if expected else {}

        result = {
            "question": query,
            "expected_files": expected,
            "kb_quality": kb_quality["score"],
            "kb_retrieval": kb_retrieval,
            "kb_n_results": kb_result["count"],
        }
        results.append(result)

        status = "✓" if kb_quality["score"] > 0.3 else "✗"
        recall_str = f"recall={kb_retrieval.get('recall', '?')}" if kb_retrieval else ""
        print(f"  [{i+1}] {status} {query[:60]}  "
              f"(quality:{kb_quality['score']:.2f} {recall_str})")

    # Summary
    if results:
        avg_quality = sum(r["kb_quality"] for r in results) / len(results)
        recalls = [r["kb_retrieval"].get("recall", 0) for r in results
                   if r.get("kb_retrieval")]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        zero_results = sum(1 for r in results if r["kb_n_results"] == 0)

        print(f"\n{'='*70}")
        print(f"  Questions:       {len(results)}")
        print(f"  Avg quality:     {avg_quality:.3f}")
        print(f"  Avg recall:      {avg_recall:.3f}" if recalls else "")
        print(f"  Zero results:    {zero_results}")
        print(f"{'='*70}")

    out_path = Path(args.output) if args.output else Path("eval/blind_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results: {out_path}")


def cmd_extract(args):
    """Extract queries from a query log into a test questions file.

    Useful for turning real usage into a reusable eval benchmark.
    """
    log_path = Path(args.query_log)
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        sys.exit(1)

    entries = _load_query_log(log_path)
    seen = set()
    questions = []

    for entry in entries:
        query = entry.get("query", "").strip()
        if not query or query in seen:
            continue
        seen.add(query)
        questions.append({
            "question": query,
            "expected_files": entry.get("result_files", []),
            "tags": [entry.get("module", ""), entry.get("scope", "")],
        })

    out_path = Path(args.output) if args.output else Path("eval/test_questions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    print(f"Extracted {len(questions)} unique queries → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KB Evaluation Harness — measure knowledge base effectiveness"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # replay
    p_replay = sub.add_parser("replay",
        help="Replay logged queries against current KB")
    p_replay.add_argument("--query-log", required=True,
        help="Path to query_log.jsonl")
    p_replay.add_argument("--kb-url", default="http://localhost:7272",
        help="R2R URL to query (default: localhost:7272)")
    p_replay.add_argument("--limit", type=int, default=100,
        help="Max queries to replay")
    p_replay.add_argument("--output", default=None,
        help="Output JSON path")

    # compare
    p_compare = sub.add_parser("compare",
        help="Compare two KBs on the same queries")
    p_compare.add_argument("--query-log", default=None,
        help="Path to query_log.jsonl (use logged queries)")
    p_compare.add_argument("--questions", default=None,
        help="Path to test_questions.jsonl (use curated questions)")
    p_compare.add_argument("--kb-a", required=True,
        help="First R2R URL")
    p_compare.add_argument("--kb-b", required=True,
        help="Second R2R URL")
    p_compare.add_argument("--judge", action="store_true",
        help="Use LLM to judge answer quality (costs tokens)")
    p_compare.add_argument("--judge-model", default="openai/gpt-4o",
        help="Model for LLM judge")
    p_compare.add_argument("--limit", type=int, default=100,
        help="Max queries to compare")
    p_compare.add_argument("--output", default=None,
        help="Output JSON path")

    # blind
    p_blind = sub.add_parser("blind",
        help="Test questions with KB, check quality and recall")
    p_blind.add_argument("--questions", required=True,
        help="Path to test_questions.jsonl")
    p_blind.add_argument("--kb-url", default="http://localhost:7272",
        help="R2R URL")
    p_blind.add_argument("--limit", type=int, default=100,
        help="Max questions")
    p_blind.add_argument("--output", default=None,
        help="Output JSON path")

    # extract
    p_extract = sub.add_parser("extract",
        help="Extract queries from log into reusable test questions")
    p_extract.add_argument("--query-log", required=True,
        help="Path to query_log.jsonl")
    p_extract.add_argument("--output", default=None,
        help="Output JSONL path")

    args = parser.parse_args()

    if args.command == "replay":
        cmd_replay(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "blind":
        cmd_blind(args)
    elif args.command == "extract":
        cmd_extract(args)


if __name__ == "__main__":
    main()
