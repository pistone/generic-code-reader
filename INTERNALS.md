# Internals — Implementation Details

This document covers how the system works under the hood. Read this when debugging or modifying the pipeline.

## Module discovery: discover_modules vs study_agent --discover

There are two module discovery paths:

**`tools/codebase.py: discover_modules()`** (primary, used by `agent.py`)
- Claude reads the full directory tree, manifests (`package.json`, `Cargo.toml`, etc.), READMEs, and key index files (`__init__.py`, `lib.rs`, etc.) in a **single LLM call**.
- Outputs `module_map.json`.
- More reliable than the exploration loop — Claude sees the whole picture at once.

**`indexer/study_agent.py --discover`** (secondary CLI)
- An **LLM tool-calling loop** where the LLM explores the directory tree iteratively using `expand_dirs`, `list_files`, `read_files`, `search_kb`, and `define_modules`.
- Runs up to 30 rounds before forcing a `define_modules` call.
- Two validation gates (coverage gate, structure validation) with up to 2 rejections and focused refinement on the next `--discover` run.
- Still available for fine-grained CLI control but no longer the primary path.

### study_agent --discover validation gates

**Gate 1 — Coverage gate** (fires at most once):
- Tracks which top-level directories have been touched by any exploration call.
- Major directories (>= 30 source files) that are unexplored when `define_modules` is called cause a rejection.
- Fires at most once — subsequent calls skip it.

**Gate 2 — Structure validation** (up to 2 rejections):
1. Minimum module count: `max(3, top_dirs // 2)` non-test modules for codebases with 500+ files.
2. No catch-all modules: any single module claiming >40% of top-level directories is rejected.
3. After 2 rejections, the result is accepted as-is.

**After the loop — advisory review**: `review_module_map()` makes a single LLM call. Issues classified as `error` (split needed, unrelated dirs merged) or `warn` are saved to `module_map.json` under `review_issues`. Re-run `--discover` to trigger focused refinement on flagged modules.

**Initial tree depth scales with codebase size**: <500 files → depth 3; 500-5000 → depth 4; 5000+ → depth 5.

### Test module filtering

The `define_modules` schema has an `is_test` field. A fallback detector (`_is_test_module`) catches tests by name keywords ("test", "mock", "fake", "fixture", "harness", "benchmark") and directory path prefixes ("test*", "mock*", "stub*").

Test modules are excluded from Pass 2 — their files are marked assigned (no orphans) but not summarized.

### Test file filtering

`collect_source_files()` excludes test dirs by exact match (`SKIP_TEST_DIRS`) and regex via `_TEST_DIR_RE`. The regex uses an explicit prefix allowlist (`_XTEST_PREFIXES`) for `{prefix}test` patterns (jtest, cstest, gtest, unittest) to avoid false positives on words like "latest", "fastest", "contest". Also excludes test files by name pattern (`test_*.py`, `*_test.cpp`, `*_spec.ts`).

---

## How Pass 2 (summarization) works

Pass 2 is **async concurrent** — chunks summarized in parallel via `AsyncRateLimiter`. Controlled by `--max-concurrent` (default 50) and `--rpm` (default 500).

Pipeline per chunk:
1. File-level summary (1 sentence) — shared across all chunks in the file
2. Class-level summary — shared across chunks in the same class
3. Function pre-pass — for multi-chunk functions, scans the full function to generate targeted questions
4. Chunk summarization — a **tool loop**:
   - LLM receives the chunk + module context
   - Can call `search_kb(query)` up to `MAX_PASS2_SEARCH_ROUNDS = 3` to look up unfamiliar types, domain concepts, or cross-module functions
   - Calls `write_summary(summary, category, search_value)` to commit
   - Simple chunks go straight to `write_summary`; complex chunks issue 1-2 searches first
5. Call graph inversion — post-pass, no LLM, extracts "calls/called by"
6. Function card synthesis — one more LLM call for multi-chunk functions

Progress output shows `[Nq]` next to chunks that issued KB queries.

### Resume

`summaries.json` saved incrementally every 10 chunks. Each entry has a `content_hash` (SHA256). On resume, the hash is verified — if the file was edited, stale summaries are discarded and the file is re-processed from scratch.

### Deduplication

Chunks deduped by exact content hash + length + line count.

### Chunk classification

Each chunk summary includes a classification tag `[category:X search_value:Y]` parsed and removed from the summary text.

Categories:
- `algorithm` — core logic, state machines → "how does X work?"
- `contract` — interfaces, APIs, protocols → "how do I use X?"
- `glue` — wiring, delegation, config → "how are X and Y connected?"
- `error_handling` — error paths, recovery → "what happens when X fails?"
- `data_model` — types, schemas, structures → "what does X look like?"
- `boilerplate` — getters, imports, logging → rarely searched

Search values: `high`, `medium`, `low`. Chunks classified as `boilerplate` + `low` are saved in `summaries.json` but skip R2R indexing. Classification stored as R2R metadata (`chunk_category`, `search_value`). `contract` maps to `source_kind: specification`; `error_handling` maps to `source_kind: operational`.

### Targeted refine

`--refine` scans summaries for vague markers ("delegates to", "defined elsewhere", "uses a helper") and only re-runs LLM on those chunks. Typically ~5% of summaries at ~5% of full `--improve` cost.

---

## How the doc agent works

Async concurrent — chunks summarized in parallel, then indexed per-file.

On transient errors (timeout, network), falls back to raw text (first 300 chars) rather than skipping the file. Only quota exhaustion causes a full halt.

Manifest (`doc_hashes.json`) saved after each file. On resume, files with matching hash + doc_ids are skipped. Files killed mid-flight (no doc_ids) are re-processed from scratch.

---

## AsyncRateLimiter

`codebase_shared/utils.py`:
- Semaphore-based concurrency control (`max_concurrent`)
- Token-bucket rate limiting (`calls_per_minute`)
- Quota detection: on hard budget errors, sets `_halted=True` — pending tasks in `run_many()` skip immediately
- `quota_exhausted` flag, `completed_count`, `skipped_count` for reporting

## TokenTracker

Thread-safe via `threading.Lock`. Multiple async tasks can record concurrently without races. Records prompt/completion tokens per phase. Prints summary table at run end, appends to `cost_log.jsonl`.

## Quota detection

`_is_quota_error(err_str)` checks for multi-word phrases: "quota", "insufficient_quota", "budget exceeded", "spending limit", "credits exhausted", "exceeded your current quota".

Transient 429 rate limits are NOT quota errors — they trigger retry with backoff. Only hard budget/billing errors trigger halt.

## max_tokens vs max_completion_tokens

Newer models (OpenAI o1, o3) only accept `max_completion_tokens`. All LLM call functions use `_apply_max_tokens()` which detects the model name (checking for "o1", "o1-mini", "o1-preview", "o3", "o3-mini" after stripping provider prefix) and sets the right parameter.

---

## File structure conventions

- `*_hashes.json` — incremental change manifests (file path → content hash + doc_ids)
- `cost_log.jsonl` — per-run token usage (one JSON line per run)
- `summaries.json` — study agent output (list of chunk summaries)
- `module_map.json` — module discovery output (project, description, modules list)
- `context_cache.json` — cached file/class/function summaries for resume
- `call_graph.json` — extracted calls/called_by relationships
- `staging_queue.json` — MCP suggestions awaiting review
- `query_log.jsonl` — MCP search queries with full answers (for eval)
