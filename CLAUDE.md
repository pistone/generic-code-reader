# Generic Code Reader — Agent Guide

## What This Is

A self-improving knowledge base for codebases. Source code is analyzed by an LLM, summaries are stored in R2R (vector DB), and an MCP server exposes them to Claude Code. When Claude can't find an answer, it suggests new entries — a reviewer agent verifies and promotes them.

**See also**: `README.md` for setup/operations, `INTERNALS.md` for implementation details.

---

## KB is empty until you populate it

R2R starts empty. Before any `search_codebase` queries return useful results, run the population pipeline.

---

## Primary workflow: agent.py

`agent.py` is a Claude agent that orchestrates the full pipeline. Use it unless you need step-by-step control.

```bash
# Start R2R first
docker compose -f r2r/compose.yaml up -d

# Index a codebase
python agent.py "Index the knowledge base for /path/to/src"

# With docs and tickets
python agent.py "Index docs at /path/to/docs, index codebase at /path/to/src, fetch Jira tickets for project PROJ"

# Use a hand-crafted module definition (skips auto-discovery)
python agent.py "Import modules from indexer/analysis_modules.json and summarize /path/to/src"

# Interactive mode
python agent.py --interactive
```

`agent.py` always requires `ANTHROPIC_API_KEY` (for the orchestrating agent itself). Summarization uses `LLM_MODEL` (default: `openai/gpt-4o`).

### Tools available to agent.py

| Tool | What it does |
|------|-------------|
| `discover_modules` | Claude reads the dir tree, manifests, and READMEs in one shot → `module_map.json` |
| `import_modules` | Loads a hand-crafted JSON (e.g. `indexer/analysis_modules.json`) → `module_map.json` |
| `summarize_codebase` | Pass 2: chunk + summarize files, index into R2R. Requires `module_map.json`. |
| `index_docs` | Parse + summarize + index `.md`/`.html`/`.txt` docs into R2R |
| `download_confluence` | Download Confluence space pages, optionally index |
| `fetch_tickets` | Fetch resolved Jira tickets → `ticket_agent/tickets/*.json` |
| `process_tickets` | Extract lessons from tickets + MR diffs, index into R2R |
| `search_kb` | Verify what's indexed or answer questions about the codebase |

**`discover_modules` vs `import_modules`**: use `discover_modules` by default — Claude analyzes the codebase in a single shot. Use `import_modules` when the user has already provided a `analysis_modules.json` with hand-crafted module definitions.

### Recommended tool order

1. `index_docs` / `download_confluence` — doc context helps code summarization
2. `fetch_tickets` + `process_tickets` — engineering lessons from tickets
3. `discover_modules` or `import_modules` — identify module structure
4. `summarize_codebase` — most expensive, run last

---

## Fallback: CLI scripts

Use the CLI scripts when you need fine-grained phase control or debugging:

```bash
# Prerequisites check
python preflight.py   # R2R health + LLM connectivity

# Module discovery (Pass 1)
python indexer/study_agent.py --codebase /path/to/src --discover
python indexer/study_agent.py --codebase /path/to/src --review   # re-review existing map

# Summarization (Pass 2)
python indexer/study_agent.py --codebase /path/to/src --summarize
python indexer/study_agent.py --codebase /path/to/src --refine   # fix vague summaries (~5% cost)
python indexer/study_agent.py --codebase /path/to/src --improve  # rewrite weak summaries
python indexer/study_agent.py --codebase /path/to/src --reindex  # re-index without re-summarizing

# Docs
python -m doc_agent.doc_agent --docs /path/to/docs
python -m doc_agent.doc_agent --docs /path/to/docs --incremental

# Ticket pipeline (3 mandatory steps in order)
python -m ticket_agent.fetch_tickets --project PROJ
python -m ticket_agent.ticket_agent --tickets ticket_agent/tickets/
python -m doc_agent.doc_agent --docs ticket_agent/lessons         # index lessons into R2R
```

---

## Prerequisites checklist

### Always required
| Requirement | Check | Fix |
|---|---|---|
| Python venv active | `which python` → `.venv/bin/python` | `source .venv/bin/activate` |
| `.env` sourced | `echo $OPENAI_API_KEY` | `source .env` |
| LLM key set | `echo $LLM_MODEL` (default: `openai/gpt-4o`) | set key in `.env` |

### R2R (required by all indexing tools and mcp_server)
```bash
curl -s http://localhost:7272/v3/health   # must return {"results":{"response":"ok"}}
# If not running:
docker compose -f r2r/compose.yaml up -d
```

### Ticket pipeline
- `JIRA_URL`, `JIRA_EMAIL`, `JIRA_TOKEN` — required for `fetch_tickets`
- `GITHUB_TOKEN` or `GITLAB_TOKEN` — required for MR diff fetching in `process_tickets`
- Verify identity before a long run: `python -m ticket_agent.fetch_tickets --project PROJ --debug`

### MCP server
- R2R must be running with indexed content
- Registered automatically via `.mcp.json`
- For a **shared team KB**: set `R2R_URL=http://your-server:7272` in `.mcp.json` env — all teammates search the same R2R instance. See README "Team Setup" for full instructions.
- `STAGING_FILE` env var overrides where `add_to_kb()` logs entries — set to a shared path for team audit trail.
- **Target codebase setup**: copy `templates/CLAUDE.md` and `.mcp.json` into the target codebase directory.

---

## Architecture diagram

```
┌──────────────────────────────────────────────────────────┐
│  POPULATION (agent.py or CLI)                            │
│                                                          │
│  discover_modules / import_modules → module_map.json     │
│  summarize_codebase → summaries.json → R2R               │
│  index_docs / process_tickets → R2R                      │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  RUNTIME (always on)                                     │
│                                                          │
│  MCP Server: search_codebase(), add_to_kb()               │
│  Claude Code ←→ MCP Server ←→ R2R or local ChromaDB      │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  REVIEWER → verifies suggestions, indexes or rejects     │
│  AUDITOR  → detects doc↔code conflicts (report only)     │
└──────────────────────────────────────────────────────────┘
```

---

## Key design decisions

**LLM-generated summaries, not raw chunks** — raw code embeds poorly; domain-vocabulary summaries retrieve much better. Hybrid storage: summary doc (for search quality) + raw source (for exact identifier matches).

**Chunk classification** — each chunk tagged with category (`algorithm`, `contract`, `glue`, `error_handling`, `data_model`, `boilerplate`) and search value (`high`, `medium`, `low`). `boilerplate` + `low` chunks skip R2R indexing.

**`discover_modules` is single-shot** — Claude reads the full dir tree, manifests, and READMEs in one call and outputs `module_map.json`. The old exploration loop (`study_agent.py --discover`) is still available but secondary.

**Crash safety** — summaries saved every 10 chunks; content hashes prevent stale cache reuse; `--incremental` uses SHA256 manifest.

---

## File structure

```
generic-code-reader/
├── agent.py               ← PRIMARY: orchestrating agent
├── tools/
│   ├── codebase.py        ← discover_modules, import_modules, summarize_codebase
│   ├── docs.py            ← index_docs
│   ├── tickets.py         ← fetch_tickets, process_tickets
│   └── confluence.py      ← download_confluence
├── indexer/
│   ├── study_agent.py     ← CLI: Pass 1 + Pass 2
│   └── analysis_modules.json  ← example hand-crafted module definition
├── doc_agent/
│   ├── doc_agent.py       ← CLI: document ingestion
│   ├── sources.py         ← pluggable source adapters
│   └── parsers.py         ← file-type parsers
├── mcp_server/server.py   ← MCP: search_codebase, add_to_kb, kb_status, list_modules (R2R or local)
├── load_kb.py             ← Load JSON files into local ChromaDB (no Docker alternative)
├── reviewer/reviewer_agent.py  ← verify + promote suggestions
├── auditor/auditor.py     ← doc↔code conflict detection
├── ticket_agent/
│   ├── fetch_tickets.py   ← fetch from Jira → tickets/*.json
│   ├── ticket_agent.py    ← knowledge extraction + MR diff fetching
│   ├── tickets/           ← persistent ticket store (gitignored)
│   └── lessons/           ← generated lesson .md files (committed)
├── explorer_agent/explorer_agent.py  ← targeted deep exploration
├── eval/eval_kb.py        ← KB effectiveness evaluation
├── codebase_shared/
│   ├── utils.py           ← LLM calls, rate limiter, token tracker
│   ├── r2r_indexer.py     ← concurrent R2R indexer
│   └── colors.py
├── run.py                 ← legacy one-command orchestrator
├── status.py / dashboard.py / preflight.py
└── r2r/
    ├── r2r.toml           ← embedding config
    └── compose.yaml
```

Runtime artifacts (gitignored):
- `indexer/module_map.json` — module discovery output
- `indexer/summaries.json` — summarization output
- `indexer/context_cache.json`, `call_graph.json`, `file_hashes.json`
- `*/cost_log.jsonl` — token usage logs
- `mcp_server/query_log.jsonl` — search query log
- `ticket_agent/ticket_hashes.json`, `ticket_summaries.json`

---

## Dependencies

All LLM calls go through `litellm` — works with OpenAI, Anthropic, Ollama, Groq, or any OpenAI-compatible endpoint. Set `LLM_MODEL` to the litellm model string (e.g. `openai/gpt-4o`, `anthropic/claude-opus-4-5`, `ollama/llama3.1`).

R2R runs as Docker containers on port 7272. Config in `r2r/r2r.toml`.
