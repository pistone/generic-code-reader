"""
Smoke tests — validate the full stack before applying to a real codebase.

Prerequisites:
  1. R2R is running:  docker compose -f r2r/compose.yaml up -d
  2. API keys set:    source .env
  3. venv active:     source .venv/bin/activate

Usage:
  python tests/smoke_test.py                    # uses this project as test codebase
  python tests/smoke_test.py --codebase /path   # use a specific directory
  python tests/smoke_test.py --model ollama/llama3.1  # different LLM
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────

R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")

# Tag for test documents so we can clean them up
TEST_ORIGIN = "smoke_test"

# ── Test runner ───────────────────────────────────────────────────────────────

class SmokeTestRunner:
    def __init__(self, codebase: Path, model: str):
        self.codebase = codebase
        self.model = model
        self.results: list[tuple[str, bool, str]] = []  # (name, passed, detail)
        self.r2r_available = False
        self.llm_available = False
        self._test_doc_ids: list[str] = []  # track for cleanup

    def run_test(self, num: int, total: int, name: str, fn):
        label = f"[{num:>2}/{total}] {name}"
        try:
            fn()
            self.results.append((name, True, ""))
            print(f"  {label} {'.' * (50 - len(name))} PASS")
        except SkipTest as e:
            self.results.append((name, True, f"SKIP: {e}"))
            print(f"  {label} {'.' * (50 - len(name))} SKIP ({e})")
        except Exception as e:
            detail = f"{e}\n{traceback.format_exc()}"
            self.results.append((name, False, detail))
            print(f"  {label} {'.' * (50 - len(name))} FAIL")
            print(f"         {e}")

    def cleanup(self):
        """Delete all test documents from R2R."""
        if not self._test_doc_ids:
            return
        try:
            from r2r import R2RClient
            client = R2RClient(R2R_URL)
            for doc_id in self._test_doc_ids:
                try:
                    client.documents.delete(doc_id)
                except Exception:
                    pass
            print(f"\n  [Cleanup] Deleted {len(self._test_doc_ids)} test document(s) from R2R")
        except Exception as e:
            print(f"\n  [Cleanup] Warning: could not clean up test docs: {e}")

    def summary(self):
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        failed_tests = [(n, d) for n, ok, d in self.results if not ok]

        print(f"\n{'='*60}")
        print(f"  Results: {passed}/{total} passed")
        if failed_tests:
            print(f"\n  Failed tests:")
            for name, detail in failed_tests:
                print(f"    - {name}")
                for line in detail.strip().split("\n")[:3]:
                    print(f"      {line}")
        print(f"{'='*60}")
        return len(failed_tests) == 0


class SkipTest(Exception):
    """Raised to skip a test (e.g. when a prerequisite is unavailable)."""
    pass


# ── Individual tests ──────────────────────────────────────────────────────────

def test_r2r_health(runner: SmokeTestRunner):
    """GET /v3/health returns 200."""
    url = f"{R2R_URL}/v3/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                raise AssertionError(f"R2R health returned {resp.status}")
        runner.r2r_available = True
    except Exception as e:
        runner.r2r_available = False
        raise AssertionError(
            f"R2R is not reachable at {R2R_URL}. "
            f"Start it with: docker compose -f r2r/compose.yaml up -d\n"
            f"Error: {e}"
        )


def test_llm_connectivity(runner: SmokeTestRunner):
    """litellm.completion() succeeds with a trivial prompt."""
    try:
        from litellm import completion
    except ImportError:
        raise AssertionError("litellm not installed. Run: pip install -r requirements.txt")

    try:
        resp = completion(
            model=runner.model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content or ""
        if not text.strip():
            raise AssertionError("LLM returned empty response")
        runner.llm_available = True
    except Exception as e:
        runner.llm_available = False
        raise AssertionError(
            f"LLM call failed with model '{runner.model}'. "
            f"Check your API key and model string.\n"
            f"Error: {e}"
        )


def test_embedding_roundtrip(runner: SmokeTestRunner):
    """Index a tiny doc into R2R and delete it — proves embeddings work."""
    if not runner.r2r_available:
        raise SkipTest("R2R not available")

    from r2r import R2RClient
    client = R2RClient(R2R_URL)

    test_text = "Smoke test: embedding connectivity check"
    try:
        resp = client.documents.create(
            raw_text=test_text,
            metadata={"origin": TEST_ORIGIN, "chunk_type": "smoke_test"},
        )
        doc_id = str(resp.results.document_id)
    except Exception as e:
        raise AssertionError(
            f"Failed to index a test document. "
            f"Check VOYAGE_API_KEY is set and R2R embedding config is correct.\n"
            f"Error: {e}"
        )

    # Clean up immediately
    try:
        client.documents.delete(doc_id)
    except Exception:
        runner._test_doc_ids.append(doc_id)


def test_chunking(runner: SmokeTestRunner):
    """chunk_file() on a known Python file produces non-empty chunks."""
    from indexer.study_agent import chunk_file

    # Use this smoke test file itself as input
    test_file = Path(__file__).resolve()
    chunks = chunk_file(test_file, language="python")

    if not chunks:
        raise AssertionError(f"chunk_file() returned no chunks for {test_file.name}")
    if len(chunks[0].strip()) < 10:
        raise AssertionError(f"First chunk is too short ({len(chunks[0])} chars)")


def test_index_search_roundtrip(runner: SmokeTestRunner):
    """Index test_entries.json, search for a known term, verify hits, clean up."""
    if not runner.r2r_available:
        raise SkipTest("R2R not available")

    from r2r import R2RClient
    client = R2RClient(R2R_URL)

    # Load test entries
    test_entries_path = PROJECT_ROOT / "indexer" / "test_entries.json"
    if not test_entries_path.exists():
        raise AssertionError(f"test_entries.json not found at {test_entries_path}")

    entries = json.loads(test_entries_path.read_text())

    # Index all entries
    indexed_ids = []
    for entry in entries:
        try:
            resp = client.documents.create(
                raw_text=entry["summary"],
                metadata={
                    "source_file": entry.get("source_file", ""),
                    "module": entry.get("module", ""),
                    "chunk_type": entry.get("chunk_type", "function_summary"),
                    "origin": TEST_ORIGIN,
                },
            )
            indexed_ids.append(str(resp.results.document_id))
        except Exception as e:
            raise AssertionError(f"Failed to index test entry: {e}")

    runner._test_doc_ids.extend(indexed_ids)

    # Wait for indexing to settle
    time.sleep(2)

    # Search for a term we know is in the test data
    results = client.retrieval.search(
        query="null dereference checker pointer",
        search_settings={"limit": 5},
    )
    hits = results.results.chunk_search_results

    if not hits:
        raise AssertionError("Search returned no hits for 'null dereference checker pointer'")

    # Verify at least one hit mentions null/pointer
    texts = " ".join((h.text or "") for h in hits).lower()
    if "null" not in texts and "pointer" not in texts:
        raise AssertionError(f"Search hits don't mention 'null' or 'pointer': {texts[:200]}")


def test_study_pass1(runner: SmokeTestRunner):
    """run_pass1() on a small codebase produces a valid ModuleMap."""
    if not runner.llm_available:
        raise SkipTest("LLM not available")

    from indexer.study_agent import collect_source_files, run_pass1

    files = collect_source_files(runner.codebase, language="python")
    if not files:
        raise SkipTest(f"No Python files found in {runner.codebase}")

    # Limit to 5 files for speed
    files = files[:5]

    module_map = run_pass1(runner.model, runner.codebase, files, "python")

    if not module_map.modules:
        raise AssertionError("Pass 1 produced zero modules")
    if not module_map.project:
        raise AssertionError("Pass 1 produced empty project name")

    # Verify structure
    for mod in module_map.modules:
        if not mod.name:
            raise AssertionError("Module has empty name")
        if not mod.questions:
            raise AssertionError(f"Module '{mod.name}' has no questions")

    # Save for pass 2 to use
    runner._module_map = module_map


def test_study_pass2(runner: SmokeTestRunner):
    """run_pass2() on the module map produces summaries."""
    if not runner.llm_available:
        raise SkipTest("LLM not available")

    module_map = getattr(runner, "_module_map", None)
    if module_map is None:
        raise SkipTest("Pass 1 did not run")

    from indexer.study_agent import run_pass2

    with tempfile.TemporaryDirectory() as tmpdir:
        summaries_path = Path(tmpdir) / "summaries.json"

        summaries = run_pass2(
            runner.model, runner.codebase, module_map,
            language="python",
            max_chunks=3,  # only 3 chunks to minimize cost
            rag=False,
            summaries_path=summaries_path,
        )

        if not summaries:
            raise AssertionError("Pass 2 produced zero summaries")
        if not summaries_path.exists():
            raise AssertionError("summaries.json was not written")

        # Verify structure
        for s in summaries:
            if not s.get("summary"):
                raise AssertionError(f"Summary is empty for {s.get('source_file', '?')}")
            if not s.get("raw_code"):
                raise AssertionError(f"raw_code is empty for {s.get('source_file', '?')}")

        runner._test_summaries = summaries


def test_index_summaries(runner: SmokeTestRunner):
    """Index the summaries from Pass 2 into R2R and verify search works."""
    if not runner.r2r_available:
        raise SkipTest("R2R not available")

    summaries = getattr(runner, "_test_summaries", None)
    if summaries is None:
        raise SkipTest("Pass 2 did not run")

    from r2r import R2RClient
    client = R2RClient(R2R_URL)

    # Index each summary
    for entry in summaries:
        try:
            resp = client.documents.create(
                raw_text=entry["summary"],
                metadata={
                    "source_file": entry.get("source_file", ""),
                    "module": entry.get("module", ""),
                    "chunk_type": entry.get("chunk_type", "function_summary"),
                    "origin": TEST_ORIGIN,
                },
            )
            runner._test_doc_ids.append(str(resp.results.document_id))
        except Exception as e:
            raise AssertionError(f"Failed to index summary: {e}")

    time.sleep(2)

    # Search for something from the summaries
    first_summary = summaries[0]["summary"]
    # Use first few words as query
    query = " ".join(first_summary.split()[:6])

    results = client.retrieval.search(
        query=query,
        search_settings={"limit": 3},
    )
    hits = results.results.chunk_search_results
    if not hits:
        raise AssertionError(f"Search returned no hits after indexing summaries (query: '{query}')")


def test_mcp_search(runner: SmokeTestRunner):
    """Import the MCP server module and call search_codebase() directly."""
    if not runner.r2r_available:
        raise SkipTest("R2R not available")

    from mcp_server.server import search_codebase

    # There should be docs indexed from previous tests
    result = search_codebase("function summary code analysis")

    if not isinstance(result, str):
        raise AssertionError(f"search_codebase returned {type(result)}, expected str")

    # It's OK if no results (KB might be empty if earlier tests were skipped),
    # but the function must not crash
    if result is None:
        raise AssertionError("search_codebase returned None")


def test_suggest_review_loop(runner: SmokeTestRunner):
    """suggest_index_item() queues an entry, process_queue() reviews it."""
    if not runner.r2r_available or not runner.llm_available:
        raise SkipTest("R2R or LLM not available")

    from mcp_server.server import suggest_index_item, _load_staging, _save_staging, STAGING_FILE

    # Save existing staging queue so we can restore it
    original_queue = _load_staging()

    try:
        # Clear staging for a clean test
        _save_staging([])

        # Submit a suggestion
        result = suggest_index_item(
            topic="smoke test — list comprehension performance",
            summary=(
                "Python list comprehensions are compiled to a tight loop by CPython, "
                "making them ~20% faster than equivalent for-loop-with-append patterns "
                "for simple transformations."
            ),
            source_files=["tests/smoke_test.py"],
            reasoning="Smoke test: validating the suggest→review pipeline works end-to-end.",
            raw_code="result = [x * 2 for x in range(100)]",
            module="smoke_test",
        )

        if "queued" not in result.lower() and "suggestion" not in result.lower():
            raise AssertionError(f"suggest_index_item returned unexpected: {result}")

        # Verify it's in the staging queue
        queue = _load_staging()
        if not queue:
            raise AssertionError("Staging queue is empty after suggest_index_item")

        pending = [e for e in queue if e.get("status") == "pending"]
        if not pending:
            raise AssertionError("No pending entries in staging queue")

        # Run the reviewer on the staging queue
        from reviewer.reviewer_agent import process_queue

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=str(PROJECT_ROOT / "tests")
        ) as tmp:
            tmp_staging = Path(tmp.name)
            tmp.write(json.dumps(queue, indent=2))

        try:
            processed = process_queue(
                model=runner.model,
                codebase=runner.codebase,
                staging_path=tmp_staging,
            )
        finally:
            tmp_staging.unlink(missing_ok=True)

        if processed == 0:
            raise AssertionError("Reviewer processed 0 entries")

        # The reviewer should have made a decision (approve, edit, or reject)
        # Any decision is valid for a smoke test — we just want no crash

    finally:
        # Restore original staging queue
        _save_staging(original_queue)


def test_doc_agent_parsing(runner: SmokeTestRunner):
    """doc_agent parsers can parse markdown and produce chunks."""
    from doc_agent.parsers import get_parser, ParsedDocument
    from doc_agent.doc_agent import chunk_document

    # Parse a markdown string
    md_parser = get_parser(".md")
    content = b"# Test Doc\n\nFirst section content.\n\n## Second\n\nMore content here."
    parsed = md_parser.parse("test.md", content, None)

    if not parsed.text:
        raise AssertionError("Parsed document has empty text")
    if not parsed.title:
        raise AssertionError("Parsed document has no title")

    chunks = chunk_document(parsed)
    if not chunks:
        raise AssertionError("chunk_document produced no chunks")
    if not chunks[0].get("text"):
        raise AssertionError("First chunk has no text")


def test_ticket_agent_filter(runner: SmokeTestRunner):
    """ticket_agent structural filter correctly filters tickets."""
    from ticket_agent.ticket_agent import structural_filter, build_ticket_context

    tickets = [
        # Should pass: resolved with comments
        {"key": "TEST-1", "title": "Bug fix", "status": "Done",
         "resolution": "Fixed", "comments": [{"author": "a", "body": "found it"},
                                               {"author": "b", "body": "fixed"}]},
        # Should fail: not resolved
        {"key": "TEST-2", "title": "Open bug", "status": "Open",
         "comments": [{"author": "a", "body": "x"}, {"author": "b", "body": "y"}]},
        # Should fail: too few comments
        {"key": "TEST-3", "title": "Quick fix", "status": "Done",
         "comments": [{"author": "a", "body": "done"}]},
        # Should fail: won't fix
        {"key": "TEST-4", "title": "Rejected", "status": "Closed",
         "resolution": "Won't Fix",
         "comments": [{"author": "a", "body": "x"}, {"author": "b", "body": "y"}]},
    ]

    passed = structural_filter(tickets)
    if len(passed) != 1:
        raise AssertionError(f"Expected 1 ticket to pass filter, got {len(passed)}")
    if passed[0]["key"] != "TEST-1":
        raise AssertionError(f"Wrong ticket passed: {passed[0]['key']}")

    # Test context building
    ctx = build_ticket_context(passed[0])
    if "TEST-1" not in ctx or "Bug fix" not in ctx:
        raise AssertionError(f"Context missing key fields: {ctx[:100]}")


def test_auditor_import(runner: SmokeTestRunner):
    """auditor module imports and run_audit function exists."""
    try:
        from auditor.auditor import run_audit
    except ImportError as e:
        raise AssertionError(f"Cannot import auditor: {e}")

    import inspect
    sig = inspect.signature(run_audit)
    if "model" not in sig.parameters:
        raise AssertionError("run_audit missing 'model' parameter")


# ── Main ──────────────────────────────────────────────────────────────────────

TESTS = [
    ("R2R Health", test_r2r_health),
    ("LLM Connectivity", test_llm_connectivity),
    ("Embedding Round-trip", test_embedding_roundtrip),
    ("Code Chunking", test_chunking),
    ("Index + Search Round-trip", test_index_search_roundtrip),
    ("Study Agent Pass 1", test_study_pass1),
    ("Study Agent Pass 2", test_study_pass2),
    ("Index Summaries", test_index_summaries),
    ("Doc Agent Parsing", test_doc_agent_parsing),
    ("Ticket Agent Filter", test_ticket_agent_filter),
    ("Auditor Import", test_auditor_import),
    ("MCP Server Search", test_mcp_search),
    ("Suggest + Review Loop", test_suggest_review_loop),
]


def main():
    parser = argparse.ArgumentParser(
        description="Smoke tests: validate the full stack before applying to a real codebase"
    )
    parser.add_argument(
        "--codebase", default=str(PROJECT_ROOT),
        help="Directory with source files to use for study agent tests (default: this project)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"litellm model string (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    codebase = Path(args.codebase).resolve()
    if not codebase.is_dir():
        print(f"Error: --codebase '{codebase}' is not a directory")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Generic Code Reader — Smoke Tests")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  R2R:      {R2R_URL}")
    print(f"  Model:    {args.model}")
    print(f"  Codebase: {codebase}")
    print(f"{'='*60}\n")

    runner = SmokeTestRunner(codebase=codebase, model=args.model)

    total = len(TESTS)
    for i, (name, fn) in enumerate(TESTS, 1):
        runner.run_test(i, total, name, lambda f=fn: f(runner))

    runner.cleanup()
    all_passed = runner.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
