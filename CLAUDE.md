# Generic Code Reader — Architecture & Design

## What This Is

A self-improving domain knowledge base for codebases. The study agent
analyzes source code and generates domain-aware summaries stored in a
vector database (R2R). An MCP server exposes these to Claude Code.
When Claude can't find an answer, it researches manually and suggests
new entries — a reviewer agent verifies and promotes them automatically.

The system is generic: given any codebase and optional design docs, it
produces a knowledge base tailored to that domain's vocabulary.

**See also**: `README.md` for setup/operations, `INTERNALS.md` for
implementation details.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  STUDY PHASE (runs once, or incrementally on changes)   │
│                                                         │
│  Pass 1: Module Discovery                               │
│    - LLM tool-calling loop explores directory tree      │
│    - Produces: module_map.json (modules + questions)    │
│                                                         │
│  Pass 2: Summarization                                  │
│    - File/class/function context → chunk summaries      │
│    - Each chunk classified: category + search value     │
│    - Call graph inversion + function cards (no LLM)     │
│    - Produces: summaries.json, call_graph.json          │
│                                                         │
│  Indexer                                                │
│    - Dual-indexes: summary doc + raw code doc           │
│    - Low-value chunks (boilerplate) skip indexing       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  RUNTIME (always on, per developer machine)             │
│                                                         │
│  MCP Server (FastMCP)                                   │
│    search_codebase(query)                               │
│    suggest_index_item(topic, summary, ...)              │
│    kb_status() / list_modules()                         │
│                                                         │
│  Claude Code ←→ MCP Server ←→ R2R (vector DB)          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  REVIEWER → verifies suggestions, indexes or rejects    │
│  AUDITOR  → detects doc↔code conflicts (report only)    │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites — What Each Tool Needs Before Running

Use this section as a checklist before invoking any tool.

### Always required (every tool)
| Requirement | How to check | Fix |
|---|---|---|
| Python venv active | `which python` → `.venv/bin/python` | `source .venv/bin/activate` |
| `.env` sourced | `echo $OPENAI_API_KEY` (or your provider) | `source .env` |
| LLM API key set | `echo $LLM_MODEL` (default: `openai/gpt-4o`) | set key in `.env` |

### R2R vector database (required by study_agent, doc_agent, ticket_agent, mcp_server)
```bash
curl -s http://localhost:7272/v3/health   # must return {"results":{"response":"ok"}}
# If not running:
cd r2r && docker compose up -d
```

### `indexer/study_agent.py` — codebase analysis
- R2R running (indexes results at the end of Pass 2)
- `--codebase /path/to/src` pointing to a readable directory
- Recommended: GPT-4o or Claude Sonnet or better (weaker models rush Pass 1)

### `doc_agent/doc_agent.py` — document ingestion
- R2R running
- `--docs /path/to/docs` pointing to a directory with `.md`, `.pdf`, `.txt`, etc.

### `ticket_agent/fetch_tickets.py` — Jira ticket fetch
- No R2R needed (writes plain JSON files only)
- No LLM needed
- Required env vars:
  - `JIRA_URL` — e.g. `https://yourco.atlassian.net`
  - `JIRA_EMAIL` — Atlassian account email that **owns** the API token (must match — wrong email = silent empty results)
  - `JIRA_TOKEN` — API token from https://id.atlassian.com/manage-profile/security/api-tokens
- Verify identity before a long run: `python -m ticket_agent.fetch_tickets --project PROJ --debug` prints the authenticated user

### `ticket_agent/ticket_agent.py` — knowledge extraction from tickets
- R2R running — required for indexing. If R2R is unreachable the agent warns
  and asks to confirm before proceeding; extraction results are always saved to
  `ticket_summaries.json` so `--reindex` can recover once R2R is up.
- LLM API key set
- Ticket JSON files on disk: run `fetch_tickets.py` first
- MR/PR token — **required** (MR diffs are the primary source of solution context):
  - `GITHUB_TOKEN` — `read:repo` scope, for GitHub PRs
  - `GITLAB_TOKEN` + `GITLAB_URL` — for GitLab MRs (default URL: https://gitlab.com)
  - At least one must be set or the agent exits with an error
  - Use `--no-mr` only if tokens are genuinely unavailable or tickets have no linked MRs
- Resumable: if interrupted, re-run the same command — already-processed tickets are skipped via hash manifest

### `mcp_server/server.py` — MCP search server
- R2R running with indexed content (run study_agent + doc_agent first)
- No LLM needed at runtime (search is embedding-based)
- Registered automatically via `.mcp.json` when opening Claude Code in this directory

### `reviewer/reviewer_agent.py` — suggestion reviewer
- R2R running
- LLM API key set
- Runs continuously; triggered by `suggest_index_item()` calls from the MCP server

### Quick environment sanity check
```bash
python preflight.py    # checks R2R health, LLM connectivity, embeddings in one shot
```

---

## Key Design Decisions

### 1. LLM-generated summaries, not raw chunks

Raw code chunks embed poorly. LLM summaries using domain vocabulary
embed and retrieve much better for natural language queries.

**Hybrid storage**: index the summary (for search quality) + raw source
code as a separate document (for exact identifier matches).

### 2. Chunk classification

Each chunk is tagged by the LLM with:
- **Category**: `algorithm`, `contract`, `glue`, `error_handling`,
  `data_model`, `boilerplate`
- **Search value**: `high`, `medium`, `low`

`boilerplate` + `low` chunks skip R2R indexing. Classification is stored
as R2R metadata for filtered search.

### 3. Doc agent for document ingestion

Runs **before** study agent so code summarization can pull doc knowledge
via RAG. Each doc chunk classified by `source_kind`: specification,
rationale, tutorial, operational, reference, overview.

### 4. MCP server

Two main tools + two introspection tools:
- `search_codebase(query)` — hybrid semantic search across all sources
  (code, docs, tickets). Cross-source linking hints when related
  knowledge exists in other source types.
- `suggest_index_item(...)` — queues new entries for reviewer
- `kb_status()` — R2R health + KB statistics
- `list_modules()` — indexed module names and descriptions

### 5. Crash safety and resumability

- Summaries saved incrementally every 10 chunks
- Content hash verification prevents stale cache reuse on resume
- Quota exhaustion halts all agents gracefully, saves progress
- `--incremental` uses SHA256 file manifest to skip unchanged files

### 6. Token tracking

All agents track prompt/completion tokens via shared `TokenTracker`.
Costs logged to `cost_log.jsonl`. Dashboard (`dashboard.py`) shows
token savings from KB vs raw file reads.

---

## File Structure

```
generic-code-reader/
├── CLAUDE.md              ← this file (architecture)
├── README.md              ← setup, operations, troubleshooting
├── INTERNALS.md           ← implementation details
├── run.py                 ← one-command orchestrator
├── status.py              ← KB health dashboard
├── dashboard.py           ← token savings dashboard
├── preflight.py           ← prerequisite checker
├── .mcp.json              ← MCP server registration
├── indexer/
│   └── study_agent.py     ← multi-pass codebase analysis
├── doc_agent/
│   ├── doc_agent.py       ← document ingestion pipeline
│   ├── sources.py         ← pluggable source adapters
│   └── parsers.py         ← file-type parsers
├── explorer_agent/
│   └── explorer_agent.py  ← autonomous goal-driven exploration
├── mcp_server/
│   └── server.py          ← MCP server (search + suggest)
├── reviewer/
│   └── reviewer_agent.py  ← verifies and promotes suggestions
├── auditor/
│   └── auditor.py         ← doc↔code conflict detection
├── ticket_agent/
│   ├── fetch_tickets.py   ← fetch from Jira API → tickets/*.json
│   ├── ticket_agent.py    ← knowledge extraction + MR diff fetching
│   ├── tickets/           ← persistent ticket store (gitignored, one file per ticket)
│   └── lessons/           ← generated lesson .md files (committed, indexable by doc_agent)
├── eval/
│   └── eval_kb.py         ← KB effectiveness evaluation
├── codebase_shared/
│   ├── utils.py           ← shared LLM calls, rate limiter, token tracker
│   ├── r2r_indexer.py     ← standalone concurrent R2R indexer
│   └── colors.py          ← terminal color helpers
└── r2r/
    ├── r2r.toml           ← R2R config (embedding, FTS)
    └── compose.yaml       ← Docker compose for R2R + Postgres
```

Runtime artifacts (gitignored):
- `indexer/module_map.json` — Pass 1 output
- `indexer/summaries.json` — Pass 2 output
- `indexer/context_cache.json` — cached context summaries
- `indexer/call_graph.json` — calls/called_by graph
- `indexer/file_hashes.json` — incremental change manifest
- `*/cost_log.jsonl` — token usage logs
- `mcp_server/query_log.jsonl` — search query + answer log
- `ticket_agent/ticket_hashes.json` — incremental manifest
- `ticket_agent/ticket_summaries.json` — saved extraction results

---

## Dependencies

All LLM calls go through `litellm` — works with OpenAI, Anthropic,
Ollama, Groq, or any OpenAI-compatible endpoint. Set `LLM_MODEL` to
the litellm model string.

R2R runs as Docker containers (postgres + R2R server) on port 7272.
Config in `r2r/r2r.toml`.
