# Generic Code Reader

A self-improving domain knowledge base for codebases. Indexes LLM-generated summaries into a vector database (R2R) and exposes them as an MCP server for Claude Code.

> **What does this do?** Pre-analyzes your codebase with an LLM, creating a searchable knowledge base. Claude Code can then instantly answer questions about architecture, patterns, and implementation details — no more hours of manual code reading.

## Quick Start

```bash
git clone <repo-url> && cd generic-code-reader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # set OPENAI_API_KEY (or your LLM provider key)
source .env
docker compose -f r2r/compose.yaml up -d
python preflight.py               # verify everything works
python run.py --codebase /path/to/src --dry-run   # estimate cost
python run.py --codebase /path/to/src             # run the full pipeline
# Done! Open Claude Code in this directory — search_codebase tool is ready.
```

## Architecture

```
doc_agent/doc_agent.py          → Ingests design docs, runbooks, wiki pages into R2R
indexer/study_agent.py          → Analyzes codebase: module discovery + summarization
ticket_agent/fetch_tickets.py   → Fetches Jira tickets → persistent per-ticket JSON files
ticket_agent/ticket_agent.py    → Extracts knowledge + lessons from tickets (fetches MR diffs)
auditor/auditor.py              → Detects doc↔code conflicts (staleness, contradictions)
mcp_server/server.py            → MCP server: search_codebase + suggest_index_item
reviewer/reviewer_agent.py      → Verifies runtime suggestions before promoting to the KB
codebase_shared/r2r_indexer.py  → Standalone concurrent R2R indexer (used by all agents)
codebase_shared/utils.py        → Shared utilities (TokenTracker, llm_call, AsyncRateLimiter)
```

The self-improving loop: when Claude can't find an answer in the KB, it researches manually and calls `suggest_index_item()`. The reviewer agent verifies the suggestion and promotes it into R2R, so the next developer gets an instant answer.

## Prerequisites

- **Python 3.10+**
- **Docker** (for R2R vector database)
- **LLM API key** — at least one of: OpenAI, Anthropic, Groq, or a local [Ollama](https://ollama.ai) install

The tool reads environment variables from your shell — if they're already set (e.g. from company tooling or `.bashrc`), you do NOT need a `.env` file.

| Variable | Required? | Default |
|----------|-----------|---------|
| `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GROQ_API_KEY`) | At least one | — |
| `OPENAI_API_BASE` | Only if using a company LLM proxy | — |
| `LLM_MODEL` | No | `openai/gpt-4o` |
| `R2R_URL` | No | `http://localhost:7272` |
| `VOYAGE_API_KEY` | No | — (for R2R embeddings) |
| `KB_SEARCH_LIMIT` | No | `5` (MCP server results per search) |
| `GITHUB_TOKEN` | ticket_agent* | — (fetch PR diffs, needs `read:repo` scope) |
| `GITLAB_TOKEN` | ticket_agent* | — (fetch MR diffs) |
| `GITLAB_URL` | No | `https://gitlab.com` (ticket agent: self-hosted GitLab) |
| `JIRA_URL` | No | — (fetch_tickets: e.g. `https://yourco.atlassian.net`) |
| `JIRA_EMAIL` | No | — (fetch_tickets: Atlassian account email) |
| `JIRA_TOKEN` | No | — (fetch_tickets: API token from id.atlassian.com) |

\* `ticket_agent`: at least one of `GITHUB_TOKEN` / `GITLAB_TOKEN` is required — MR diffs are the primary source of solution context. Use `--no-mr` only if tokens are unavailable.

## Setup

### 1. Clone and install

```bash
git clone <repo-url> && cd generic-code-reader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY (or another provider key)
source .env
```

**Embedding dimension note**: If you change the embedding model in `r2r/r2r.toml`, ensure `base_dimension` matches: `text-embedding-3-small` → 512, `voyage-code-2` → 1536. All R2R embedding configs must use the same dimension or you'll see a `Both embedding configurations must use the same dimensions` error.

### 3. Start R2R

```bash
docker compose -f r2r/compose.yaml up -d
# Wait ~30s for it to be healthy, then verify:
python preflight.py
```

### 4. Run the pipeline

```bash
# Full pipeline: docs → code analysis → ticket knowledge
python run.py \
    --codebase /path/to/src \
    --docs /path/to/design-docs \      # optional
    --tickets /path/to/jira-export \   # optional
    --model-fast openai/gpt-4o-mini    # optional: cheaper model for bulk summarization

# Multiple codebases:
python run.py --codebase /path/to/src/core /path/to/src/plugins
```

`run.py` runs the full pipeline in order: doc_agent (if `--docs`) → Pass 1 module discovery → Pass 2 summarization → index to R2R → ticket_agent (if `--tickets`).

### 5. Open Claude Code

The `.mcp.json` is already configured. Open Claude Code in this directory — the `search_codebase` tool is immediately available. It searches across code summaries, docs, and tickets automatically.

## Running Each Step Separately

For large codebases, run phases independently to inspect output before proceeding:

```bash
# (Optional) Index design docs first — Pass 2 will query them on-demand during summarization
python -m doc_agent.doc_agent --docs /path/to/docs
python -m doc_agent.doc_agent --docs /path/to/docs --incremental   # changed docs only

# Phase 1: Module discovery → indexer/module_map.json
python indexer/study_agent.py --codebase /path/to/src --discover
# After discovery, an LLM reviews the module map and flags quality issues.
# If errors are found, re-run --discover to trigger a focused refinement round:
python indexer/study_agent.py --codebase /path/to/src --discover   # fixes flagged modules

# Re-run only the reviewer on an existing module_map.json (no Pass 1):
python indexer/study_agent.py --codebase /path/to/src --review
# If --review flags errors, run --discover to fix them.

# Point at a subdirectory to fill in missing modules — new ones are merged in automatically:
python indexer/study_agent.py --codebase /path/to/src/some/subdir --discover

# Phase 2: Summarization → indexer/summaries.json → auto-indexed to R2R
# The LLM can call search_kb during summarization to look up unfamiliar types and concepts.
python indexer/study_agent.py --codebase /path/to/src --summarize
# Fix vague summaries (~5% of chunks, much cheaper than full re-run):
python indexer/study_agent.py --codebase /path/to/src --refine

# Re-review and rewrite weak summaries (expensive — use --refine first):
python indexer/study_agent.py --codebase /path/to/src --improve

# Re-index without re-summarizing (after edits, model change, or R2R restart):
python indexer/study_agent.py --codebase /path/to/src --reindex
python -m doc_agent.doc_agent --docs /path/to/docs --reindex
python -m ticket_agent.ticket_agent --reindex

# Ticket pipeline — run all 3 steps in order:

# Step 1: Fetch tickets from Jira → ticket_agent/tickets/*.json
python -m ticket_agent.fetch_tickets --project PROJ              # resolved tickets, past year
python -m ticket_agent.fetch_tickets --project PROJ --since -90d # last 90 days
python -m ticket_agent.fetch_tickets --project ABC DEF           # multiple projects / teams
python -m ticket_agent.fetch_tickets --project PROJ --incremental # only new/changed
python -m ticket_agent.fetch_tickets \
  --jql "project = PROJ AND status = Done AND updated >= -90d"   # custom JQL

# Step 2: Extract knowledge + lessons → indexes ticket summaries into R2R
#         Also writes how-to recipes to ticket_agent/lessons/*.md
python -m ticket_agent.ticket_agent --tickets ticket_agent/tickets/ --key-pattern '^ABC-'
python -m ticket_agent.ticket_agent --tickets ticket_agent/tickets/   # all tickets

# Step 3: REQUIRED — index lessons into R2R as rich searchable how-to guides
#         Skipping this leaves lessons as files but not fully searchable in the KB.
python -m doc_agent.doc_agent --docs ticket_agent/lessons
```

## Incremental Updates

When source code changes, re-run with `--incremental` — only files with changed content hashes are re-processed:

```bash
python run.py --codebase /path/to/src --incremental
```

## Tuning

These flags are on `study_agent.py` directly (not `run.py`, which uses sensible defaults):

**`--max-concurrent`** (default: 50) — parallel LLM calls:
- Company shared proxy: `5-10`
- Company dedicated quota: `15-25`
- OpenAI direct: `20-50`
- Local Ollama/vLLM: `2-4`

**`--rpm`** (default: 60) — requests per minute. Auto-detected from proxy headers when possible. OpenAI Tier 1 = 500 RPM.

**`--exclude`** — skip directories that don't add KB value:
```bash
python indexer/study_agent.py --codebase /path/to/src \
    --exclude generated proto_out third_party
```

## Cost and Quota Management

**Estimate before running**: `--dry-run` shows token and cost estimates without any LLM calls. The running agent also prompts "Proceed? [Y/n]" — skip with `--yes`.

**Quota exhaustion**: If the LLM hits a budget error mid-run, all agents halt, save progress, and print a resume message. Re-run the same command to continue. Files already in the manifest (`*_hashes.json`) are skipped — you pay zero tokens for completed work.

## Post-Run

```bash
python status.py                                # KB health and stats
python -m auditor.auditor                       # check doc↔code conflicts
python dashboard.py --period 7d --by-module     # token savings by module
```

## Clean Slate

To re-run the full pipeline from scratch (e.g. after changing prompts or the model):

```bash
# Delete artifacts
rm -f indexer/module_map.json indexer/summaries.json
rm -f indexer/context_cache.json indexer/call_graph.json indexer/file_hashes.json
rm -f doc_agent/doc_hashes.json
rm -f ticket_agent/ticket_hashes.json ticket_agent/ticket_summaries.json

# Reset R2R
docker compose -f r2r/compose.yaml down -v && docker compose -f r2r/compose.yaml up -d

# Re-run
python run.py --codebase /path/to/src ...
```

> **Note**: If you don't clear R2R, old and new summaries coexist. The indexer replaces entries by content hash, but changed prompts produce different text that won't match — always reset R2R for a true clean slate.

## Evaluating KB Effectiveness

The MCP server logs every `search_codebase` call to `mcp_server/query_log.jsonl`.

```bash
python eval/eval_kb.py extract --query-log mcp_server/query_log.jsonl  # build benchmark from usage
python eval/eval_kb.py replay  --query-log mcp_server/query_log.jsonl  # replay against current KB
python eval/eval_kb.py compare \
    --questions eval/test_questions.jsonl \
    --kb-a http://localhost:7272 --kb-b http://localhost:7273            # A/B comparison
python eval/eval_kb.py blind   --questions eval/test_questions.jsonl    # score against expected files
```

## Explorer Agent

For targeted deep exploration of a specific area rather than broad coverage:

```bash
python -m explorer_agent.explorer_agent \
    --codebase /path/to/src/core/engine \
    --context-root /path/to/src
```

Output is compatible with study agent's `summaries.json` schema.

## Reviewer Agent

When Claude Code calls `suggest_index_item()`, suggestions land in `mcp_server/staging_queue.json`. Run the reviewer to verify and promote them:

```bash
python reviewer/reviewer_agent.py --codebase /path/to/your/src   # process once
python reviewer/reviewer_agent.py --codebase /path/to/your/src --watch  # continuous
```

## Testing

```bash
python tests/smoke_test.py                        # uses this project as test codebase
python tests/smoke_test.py --codebase /path/to/src
python tests/smoke_test.py --model ollama/llama3.1
```

Smoke tests check R2R health, LLM connectivity, embeddings, chunking, indexing, Pass 1, Pass 2, MCP search, and the suggest+review loop. Total cost < $0.01.

## Troubleshooting

**R2R won't start / embedding errors**
```bash
curl http://localhost:7272/v3/health
docker compose -f r2r/compose.yaml logs r2r 2>&1 | tail -30
docker compose -f r2r/compose.yaml down && docker compose -f r2r/compose.yaml up -d
```
If you see `Both embedding configurations must use the same dimensions`: check `base_dimension` in `r2r/r2r.toml` matches your model (`voyage-code-2` → 1536, `text-embedding-3-small` → 512).

**LLM calls failing**
```bash
python -c "from litellm import completion; print(completion(model='openai/gpt-4o', messages=[{'role':'user','content':'hi'}], max_tokens=5).choices[0].message.content)"
```
- Wrong model string: use litellm format — `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`, `ollama/llama3.1`
- Missing key: set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` etc.) and `source .env`
- Company proxy: set `OPENAI_API_BASE` in `.env`

**Pass 1 finds too few modules** (e.g. 2 modules for a 10K-file repo): the LLM rushed. Use at least GPT-4o or Claude Sonnet for Pass 1. Re-run with `--discover` — if the auto-review flagged errors in `module_map.json`, the re-run automatically does a focused refinement round instead of starting from scratch.

**Study agent killed mid-run**: just re-run the same command. Content hashes ensure only incomplete files are re-processed.

**Stale `shared/` import errors**: the module was renamed to `codebase_shared/`. If you see `ModuleNotFoundError: No module named 'shared.utils'`, all imports should use `from codebase_shared.utils import ...`.

## Supported Languages

`python`, `javascript`, `typescript`, `cpp`, `java`, `go`, `rust`

The study agent auto-detects language from file extensions. Override with `--language <lang>` if needed.
