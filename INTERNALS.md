# Internals — Implementation Details

This document covers how the system works under the hood. Read this
when debugging issues or modifying the pipeline.

## How Pass 1 (module discovery) works

Pass 1 is an **LLM tool-calling loop**, not a static analysis. The LLM
gets a depth-limited directory tree and 5 tools: `expand_dirs`,
`list_files`, `read_files`, `search_kb`, and `define_modules`.

It explores iteratively for up to 20 rounds. When `define_modules` is
called, the result is **validated before acceptance**:

1. **Minimum module count**: `max(3, top_dirs // 2)` non-test modules
   required for codebases with 500+ files.
2. **No catch-all modules**: Any single module claiming >40% of
   top-level directories is rejected.
3. **Retry with feedback**: On rejection, the LLM gets an error message
   explaining what's wrong and is told to explore more. After 2
   rejections, the result is accepted as-is to avoid infinite loops.

If the LLM never calls `define_modules` or hits max rounds, it's forced
with `tool_choice`, then falls back to one-module-per-top-level-dir.

**The initial tree depth scales with codebase size**:
- <500 files: depth 3
- 500-5000 files: depth 4
- 5000+ files: depth 5

**`expand_dirs` accumulates**: Each call adds to the set of expanded
directories. The tree is rebuilt with all previously expanded dirs +
new ones, so the LLM sees a progressively richer view.

**Top-level directory summary**: The initial prompt includes a compact
list of all top-level directories with file counts, so the LLM knows
what areas exist before it starts exploring.

### Test module filtering

The `define_modules` schema includes an `is_test` field. The LLM flags
test modules (unit tests, fixtures, mocks, benchmarks). A fallback
detector (`_is_test_module`) also catches test modules by:
- Name keywords: "test", "mock", "fake", "fixture", "harness", "benchmark"
- Description keywords: "unit test", "test suite", "test harness"
- Directory path prefixes: "test*", "mock*", "fake*", "stub*"

Test modules are excluded from Pass 2 — their files are marked as
assigned (so they don't become orphans in an "other" module) but not
summarized or indexed.

### Test file filtering

`collect_source_files()` excludes test dirs by exact match
(SKIP_TEST_DIRS: "test", "tests", "testing", etc.) AND regex match
via `_TEST_DIR_RE`. The regex uses an **explicit prefix allowlist**
(`_XTEST_PREFIXES`) for `{prefix}test` patterns (jtest, cstest, gtest,
unittest, etc.) to avoid false positives on English words like "latest",
"fastest", "attest", "contest". Add project-specific prefixes to
`_XTEST_PREFIXES` if needed.

Also excludes test files by name pattern (test_*.py, *_test.cpp,
*_spec.ts, etc.). Use `--include-tests` to override.

## How Pass 2 (summarization) works

Pass 2 is **async concurrent** — all chunks are summarized in parallel
via `AsyncRateLimiter`. The concurrency is controlled by
`--max-concurrent` (default 50) and `--rpm` (default 500).

The pipeline per chunk:
1. File-level summary (1 sentence) — shared across all chunks in the file
2. Class-level summary — shared across chunks in the same class
3. Function pre-pass — for multi-chunk functions, scans the full function
   to generate targeted questions
4. Chunk summary — the main LLM call, receives all the above as context
5. Call graph inversion — post-pass, no LLM, extracts "calls/called by"
6. Function card synthesis — for multi-chunk functions, one more LLM call

### Resume

`summaries.json` is saved incrementally (every 10 chunks). Each entry
has a `content_hash` (SHA256 of the chunk text). On resume, the hash is
verified — if the file was edited between runs, stale summaries are
discarded and the file is re-processed from scratch.

### Deduplication

Chunks are deduped by exact content hash + length + line count.
Duplicates are logged and skipped.

### Chunk classification

Each chunk summary includes a classification tag
`[category:X search_value:Y]` that the LLM generates alongside the
summary. The tag is parsed and removed from the summary text.

Categories:
- `algorithm` — core logic, state machines → "how does X work?"
- `contract` — interfaces, APIs, protocols → "how do I use X?"
- `glue` — wiring, delegation, config → "how are X and Y connected?"
- `error_handling` — error paths, recovery → "what happens when X fails?"
- `data_model` — types, schemas, structures → "what does X look like?"
- `boilerplate` — getters, imports, logging → rarely searched

Search values: `high`, `medium`, `low`. Chunks classified as
`boilerplate` + `low` are saved in summaries.json but skip R2R
indexing (reducing noise and embedding cost).

Classification is stored as R2R metadata (`chunk_category`,
`search_value`) for filtered search. The `contract` category maps to
`source_kind: specification` and `error_handling` maps to
`source_kind: operational` for scope-based search.

Pass 2 prints a classification breakdown at the end.

### Targeted refine

`--refine` scans summaries for vague markers ("delegates to", "defined
elsewhere", "uses a helper", etc.) and only re-runs LLM on those
chunks. Typically touches ~5% of summaries at ~5% of the cost of a
full `--review-only` pass.

## How the doc agent works

The doc agent is **async concurrent**. Chunks across all files are
summarized in parallel, then indexed per-file.

If a chunk fails summarization due to transient errors (timeout, network),
it falls back to raw text (first 300 chars) rather than skipping the
entire file. Only quota exhaustion causes a full halt.

The manifest (`doc_hashes.json`) saves after each file is indexed.
On resume, files with matching hash + doc_ids in the manifest are
skipped. Files whose previous run was killed mid-flight (no doc_ids)
are re-processed from scratch.

## AsyncRateLimiter

The shared rate limiter (`codebase_shared/utils.py`) provides:
- Semaphore-based concurrency control (`max_concurrent`)
- Token-bucket rate limiting (`calls_per_minute`)
- Quota detection: on hard budget errors, sets `_halted=True` and all
  pending tasks in `run_many()` skip immediately
- `quota_exhausted` flag for callers to check after `run_many()` returns
- `completed_count` and `skipped_count` for progress reporting

## TokenTracker

Thread-safe via `threading.Lock` on all mutation methods. Multiple
async tasks sharing a tracker can record concurrently without races.

Records prompt/completion token counts per phase. At the end of each
run, prints a summary table and appends to `cost_log.jsonl`.

## Quota detection

`_is_quota_error(err_str)` checks for multi-word phrases to avoid false
positives. The signals include: "quota", "insufficient_quota", "budget
exceeded", "spending limit", "credits exhausted", "exceeded your
current quota", etc.

Transient 429 rate limits are NOT treated as quota errors — they trigger
retry with backoff. Only hard budget/billing errors trigger halt.

## max_tokens vs max_completion_tokens

Newer models (OpenAI o1, o3) only accept `max_completion_tokens` instead
of `max_tokens`. All LLM call functions use `_apply_max_tokens()` which
detects the model name (checking for "o1", "o1-mini", "o1-preview",
"o3", "o3-mini" prefixes after stripping the provider prefix) and sets
the right parameter. This is transparent to callers.

## File structure conventions

- `*_hashes.json` — incremental change manifests (file path → content hash + doc_ids)
- `cost_log.jsonl` — per-run token usage logs (one JSON line per run)
- `summaries.json` — main study agent output (list of chunk summaries)
- `module_map.json` — Pass 1 output (project name, description, modules list)
- `context_cache.json` — cached file/class/function summaries for resume
- `call_graph.json` — extracted calls/called_by relationships
- `staging_queue.json` — MCP suggestions awaiting review
- `query_log.jsonl` — MCP search queries with full answers (for eval)
