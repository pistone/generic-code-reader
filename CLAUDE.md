# Generic Code Reader — Architecture & Design

## What This Is

A self-improving domain knowledge base for codebases. The study agent
analyzes source code and generates domain-aware summaries stored in a
vector database (R2R). An MCP server exposes these to Claude Code.
When Claude can't find an answer, it researches manually and suggests
new entries — a reviewer agent verifies and promotes them automatically.

The system is generic: given any codebase and optional design docs, it
produces a knowledge base tailored to that domain's vocabulary.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  STUDY PHASE (runs once, or incrementally on changes)   │
│                                                         │
│  Pass 1: Module Discovery                               │
│    - Reads directory tree + file samples                │
│    - Optionally consults design docs (--docs)           │
│    - Produces: module_map.json (modules + questions)    │
│                                                         │
│  Pass 2: Summarization                                  │
│    2a. File summaries — 1-sentence per file             │
│    2b. Function pre-pass — for multi-chunk functions,   │
│        scans signature + structure → generates targeted  │
│        per-chunk questions ("Focus on: ...")             │
│    2c. Class summaries — 1-sentence per class           │
│    2d. Chunk summarization — domain-aware summaries     │
│        informed by file/class/function context           │
│        + pre-pass questions + optional RAG context       │
│    2e. Call graph inversion — zero LLM cost, extracts   │
│        calls/called-by from summaries + code, enriches  │
│        summaries with "Called by: ..." annotations       │
│    2f. Function cards — for multi-chunk functions,       │
│        synthesizes contract/phases/decisions/complexity  │
│    - Produces: summaries.json, call_graph.json          │
│                                                         │
│  Pass 3+: Review (when --passes > 1)                    │
│    - LLM reviews each summary for accuracy              │
│    - Rewrites weak summaries using KB vocabulary        │
│    - Stops when edit rate < 5% (convergence)            │
│                                                         │
│  Indexer                                                │
│    - Dual-indexes: summary doc + raw code doc           │
│    - Summary → semantic search; raw code → exact match  │
│    - Auto-indexed when --passes > 1                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  RUNTIME (always on, per developer machine)             │
│                                                         │
│  MCP Server (FastMCP, ~180 lines Python)                │
│    search_codebase(query, module="", source_type="")    │
│      → hybrid search (vector + full-text), returns      │
│        formatted results with scores                    │
│      → source_type: "code" | "doc" | "ticket" | ""     │
│    suggest_index_item(topic, summary, source_files,     │
│                       reasoning, raw_code, module)      │
│      → writes to staging_queue.json                     │
│                                                         │
│  Claude Code ←→ MCP Server ←→ R2R (vector DB)          │
└─────────────────────────────────────────────────────────┘
                          │
          suggest_index_item() called
                          ▼
┌─────────────────────────────────────────────────────────┐
│  REVIEWER AGENT (runs on demand or in watch mode)       │
│                                                         │
│    - Reads cited source files to verify accuracy        │
│    - Searches KB for duplicates                         │
│    - Calls LLM (via litellm) to decide:                │
│      approve / edit / reject                            │
│    - On approve: dual-indexes into R2R                  │
│    - On reject: logs to rejected_queue.json             │
│    - Prunes staging queue after processing              │
└─────────────────────────────────────────────────────────┘
                          │
          after all agents have indexed
                          ▼
┌─────────────────────────────────────────────────────────┐
│  AUDITOR (runs after doc_agent + study_agent)           │
│                                                         │
│    - Compares doc entries vs code entries in R2R        │
│    - Timestamp-first: flags docs older than threshold   │
│    - LLM comparison on stale suspects only              │
│    - Outputs conflict_report.json for human review      │
│    - Does NOT auto-fix — report only                    │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. LLM-generated summaries, not raw chunks

Raw code chunks embed poorly — semantic meaning is implicit in logic,
not surface text. LLM summaries using domain vocabulary embed and
retrieve much better for natural language queries.

**Hybrid storage**: index the summary (for search quality), but also
index raw source code as a separate document (chunk_type: "raw_code")
so searches for exact identifiers match the source directly.

### 2. Multi-pass iterative summarization

- **Pass 1**: Module discovery — LLM groups files into modules and
  generates domain-specific question lists
- **Pass 2**: Summarization with multi-level context:
  - Function pre-pass generates targeted questions per chunk
  - Chunk summaries guided by file/class/function context + questions
  - Call graph inversion adds "Called by" annotations (no LLM cost)
  - Function cards synthesize contract/phases/decisions for complex functions
- **Pass 3+**: Review — LLM reviews each summary against KB context,
  rewrites weak ones. Converges when edit rate < 5%

### 3. Doc agent for document ingestion

The doc agent (`doc_agent/`) ingests design docs, runbooks, wiki pages,
and other prose documents into R2R.  It runs **before** the study agent
so code summarization can pull doc knowledge via RAG.

```
python -m doc_agent.doc_agent --docs /path/to/docs --model openai/gpt-4o-mini
python -m doc_agent.doc_agent --docs /path/to/docs --no-summarize  # raw text only
```

Architecture: pluggable sources (LocalFileSource, future: SharePoint,
Confluence) → file-type parsers (Markdown, HTML, PDF, plain text) →
section-aware chunking → LLM summarization with `source_kind`
classification → R2R indexing.

Each chunk is classified by `source_kind`:
- `specification` — API contracts, formats, protocols
- `rationale` — design decision explanations
- `tutorial` — step-by-step instructions
- `operational` — runbooks, deployment, troubleshooting
- `reference` — API reference, parameter lists
- `overview` — high-level architecture descriptions

Use `--no-summarize` to skip LLM and index raw text (faster, cheaper).

The study agent's `--bootstrap-docs` flag is deprecated in favor of
doc_agent, which provides section-aware chunking, heading extraction,
HTML parsing, incremental mode, and proper metadata.

### 4. R2R as the backend

R2R runs as Docker containers (postgres + R2R server) on port 7272.
Uses Voyage embeddings via litellm. Config in `r2r/r2r.toml`:

```toml
[embedding]
provider = "litellm"
base_model = "voyage/voyage-code-2"
base_dimension = 1536

[database]
provider = "postgres"
enable_fts = true      # full-text search for hybrid retrieval
```

### 5. MCP server

Registered via `.mcp.json`:

```json
{
  "mcpServers": {
    "domain-kb": {
      "command": ".venv/bin/python",
      "args": ["mcp_server/server.py"],
      "env": {
        "R2R_URL": "http://localhost:7272",
        "KB_SEARCH_LIMIT": "5"
      }
    }
  }
}
```

Two tools:

`search_codebase(query, module, source_type, scope)` — hybrid search with:
- `source_type` filter: "code"/"doc"/"ticket"
- `scope` parameter for intent-based prioritization:
  - `"implementation"` — boosts code results
  - `"rationale"` — boosts docs/tickets with design rationale
  - `"howto"` — boosts tutorials and operational guides
  - `"troubleshooting"` — boosts tickets with workarounds/root causes
  - `"architecture"` — boosts file/class overviews
- Cross-source linking: when results come from one source type,
  automatically checks for related knowledge in other sources

`suggest_index_item(topic, summary, source_files, reasoning, raw_code, module)`.
Logs every query to `mcp_server/query_log.jsonl`.

### 6. Crash safety and resumability

- Summaries written incrementally every 5 chunks
- Chunk keys (`source_file::chunk_index`) detect already-done work
- `--incremental` flag uses SHA256 file hash manifest to skip
  unchanged files between runs
- Rate-limit retry with immediate write after success

### 7. Token tracking

All agents (study, reviewer, auditor, ticket) track prompt/completion
token counts per phase via shared `TokenTracker`. Printed at the end
of each run and appended to `cost_log.jsonl` for historical tracking.

---

## File Structure

```
generic-code-reader/
├── CLAUDE.md                  ← this file
├── README.md                  ← setup and usage guide
├── preflight.py               ← prerequisite checker (python preflight.py)
├── .mcp.json                  ← MCP server registration
├── .env.example               ← environment variable template
├── .gitignore
├── requirements.txt           ← Python dependencies
├── auditor/
│   └── auditor.py             ← cross-reference auditor (doc↔code conflict detection)
├── ticket_agent/
│   └── ticket_agent.py        ← knowledge extraction from Jira/PR ticket exports
├── doc_agent/
│   ├── doc_agent.py           ← document ingestion pipeline
│   ├── sources.py             ← pluggable source adapters (local, future: SharePoint)
│   └── parsers.py             ← file-type parsers (Markdown, HTML, PDF, plain text)
├── indexer/
│   ├── study_agent.py         ← multi-pass codebase analysis
│   ├── indexer.py             ← feeds summaries to R2R
│   └── test_entries.json      ← sample KB entries for testing
├── mcp_server/
│   └── server.py              ← MCP server (search + suggest tools)
├── reviewer/
│   └── reviewer_agent.py      ← verifies and promotes suggestions
├── codebase_shared/
│   └── utils.py               ← shared utilities (TokenTracker, llm_call, llm_tool_loop, RateLimitedExecutor, manifest helpers)
├── r2r/
│   ├── r2r.toml               ← R2R config (embedding, FTS, etc.)
│   └── compose.yaml           ← Docker compose for R2R + Postgres
└── tests/
    └── smoke_test.py          ← pre-deployment integration tests
```

Runtime artifacts (gitignored):
- `indexer/module_map.json` — Pass 1 output
- `indexer/summaries.json` — Pass 2 output (chunks, overviews, function cards)
- `indexer/call_graph.json` — call graph: calls, called_by, stats
- `indexer/context_cache.json` — cached file/class/function summaries for resume
- `indexer/file_hashes.json` — incremental change manifest
- `indexer/cost_log.jsonl` — token usage log
- `ticket_agent/ticket_summaries.json` — ticket agent output
- `doc_agent/doc_hashes.json` — doc incremental change manifest
- `doc_agent/cost_log.jsonl` — doc agent token usage log
- `auditor/conflict_report.json` — doc↔code conflict report
- `auditor/cost_log.jsonl` — auditor token usage log
- `ticket_agent/ticket_hashes.json` — ticket incremental manifest
- `ticket_agent/cost_log.jsonl` — ticket agent token usage log
- `mcp_server/staging_queue.json` — pending suggestions
- `mcp_server/query_log.jsonl` — search query audit log
- `reviewer/rejected_queue.json` — rejected suggestions
- `reviewer/cost_log.jsonl` — reviewer token usage log

---

## Explorer Agent (Autonomous)

An alternative to the study agent's fixed pipeline. The explorer agent
decides what to read, how deep to go, and when to stop.

```bash
python -m explorer_agent.explorer_agent \
    --codebase /path/to/src \
    --context-root /path/to/repo   # optional: follow refs outside subtree
```

It maintains a knowledge state with modules, hypotheses, and confidence
scores. Output is compatible with the study agent's `summaries.json`
schema and can be indexed into R2R the same way.

Use this when you want deep exploration of a specific area rather than
broad coverage of the whole codebase.

---

## Evaluating KB Effectiveness

The MCP server logs every `search_codebase` call — including the full
question, answer, and result files — to `mcp_server/query_log.jsonl`.

### Workflow

1. **Use the KB naturally.** Have Claude resolve tickets or answer
   questions while the MCP server is running. Queries accumulate.

2. **Extract a reusable benchmark** from real usage:
   ```bash
   python eval/eval_kb.py extract --query-log mcp_server/query_log.jsonl
   # → eval/test_questions.jsonl (deduped, with expected files)
   ```

3. **Evaluate** with one of three modes:

   **Replay** — re-run logged queries against the current (or rebuilt) KB:
   ```bash
   python eval/eval_kb.py replay --query-log mcp_server/query_log.jsonl
   ```

   **Compare** — same queries against two KBs (e.g. study_agent vs
   explorer_agent, or old vs new). Run two R2R instances on different ports:
   ```bash
   python eval/eval_kb.py compare \
       --questions eval/test_questions.jsonl \
       --kb-a http://localhost:7272 \
       --kb-b http://localhost:7273 \
       --judge    # optional: LLM judges which answer is better
   ```

   **Blind** — score KB quality and recall against expected files:
   ```bash
   python eval/eval_kb.py blind \
       --questions eval/test_questions.jsonl
   ```

Each mode produces a JSON results file with per-query scores (quality,
precision, recall, F1) and an aggregate summary.

### Artifacts

- `mcp_server/query_log.jsonl` — full question+answer log (auto-generated)
- `eval/test_questions.jsonl` — curated benchmark (extracted or hand-written)
- `eval/replay_results.json` — replay eval output
- `eval/compare_results.json` — A/B comparison output
- `eval/blind_results.json` — blind eval output

---

## Dependencies

All LLM calls go through `litellm` — works with OpenAI, Anthropic,
Ollama, Groq, or any OpenAI-compatible endpoint (e.g. internal
company servers). Set `LLM_MODEL` to the litellm model string.

Embeddings go through R2R's litellm integration. For on-premise
deployment, change `base_model` in `r2r.toml` to point at your
internal embedding server.
