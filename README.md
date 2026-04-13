# Generic Code Reader

A self-improving knowledge base for codebases. Pre-analyzes source code with an LLM, stores summaries in a vector DB (R2R or local ChromaDB), and exposes them as an MCP server for Claude Code.

## Quick Start

```bash
git clone <repo-url> && cd generic-code-reader
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY and OPENAI_API_KEY (or your provider)
source .env && docker compose -f r2r/compose.yaml up -d && python preflight.py
python agent.py "Index the knowledge base for /path/to/src"
# Done — open Claude Code in this directory, search_codebase tool is ready.
```

## Architecture

```
agent.py                        → Primary: orchestrating Claude agent (calls tools/ below)
tools/codebase.py               → discover_modules(), import_modules(), summarize_codebase()
tools/docs.py                   → index_docs()
tools/tickets.py                → fetch_tickets(), process_tickets()
tools/confluence.py             → download_confluence()
indexer/study_agent.py          → CLI: codebase analysis (Pass 1 + Pass 2)
doc_agent/doc_agent.py          → CLI: ingest design docs, runbooks, wiki pages
ticket_agent/fetch_tickets.py   → CLI: fetch Jira tickets → JSON files
ticket_agent/ticket_agent.py    → CLI: extract knowledge from tickets
auditor/auditor.py              → detect doc↔code conflicts
mcp_server/server.py            → MCP server: search_codebase + add_to_kb (R2R or local ChromaDB)
reviewer/reviewer_agent.py      → verify runtime suggestions (optional review gate)
codebase_shared/r2r_indexer.py  → concurrent R2R indexer (shared by all agents)
codebase_shared/local_kb.py     → ChromaDB local backend (no Docker needed)
load_kb.py                      → Load JSON files into local ChromaDB
codebase_shared/utils.py        → TokenTracker, llm_call, AsyncRateLimiter
```

When Claude can't find an answer in the KB, it researches manually and calls `add_to_kb()` — the entry is indexed immediately and available to the whole team.

## Prerequisites

- **Python 3.10+**
- **Docker** (for R2R)
- **`ANTHROPIC_API_KEY`** — always required (for `agent.py`)
- **At least one summarization key**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, or a local [Ollama](https://ollama.ai) install

The tool reads env vars from your shell — if they're already set (e.g. from `.bashrc`), you do NOT need a `.env` file.

| Variable | Required? | Default |
|----------|-----------|---------|
| `ANTHROPIC_API_KEY` | Yes (agent.py) | — |
| `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GROQ_API_KEY`) | At least one for summarization | — |
| `OPENAI_API_BASE` | Only if using a company LLM proxy | — |
| `LLM_MODEL` | No | `openai/gpt-4o` |
| `R2R_URL` | No | `http://localhost:7272` |
| `VOYAGE_API_KEY` | No | — (for R2R embeddings) |
| `KB_BACKEND` | No | `r2r` (`r2r` or `local` for ChromaDB) |
| `LOCAL_KB_DIR` | No | `chroma_db` (path to ChromaDB directory) |
| `KB_SEARCH_LIMIT` | No | `5` (MCP server results per search) |
| `GITHUB_TOKEN` | ticket pipeline* | — (`read:repo` scope) |
| `GITLAB_TOKEN` | ticket pipeline* | — |
| `GITLAB_URL` | No | `https://gitlab.com` |
| `JIRA_URL` | ticket pipeline | — (e.g. `https://yourco.atlassian.net`) |
| `JIRA_EMAIL` | ticket pipeline | — |
| `JIRA_TOKEN` | ticket pipeline | — (from id.atlassian.com) |

\* At least one of `GITHUB_TOKEN` / `GITLAB_TOKEN` is required for MR diffs. Use `--no-mr` only if tokens are unavailable.

## Setup

### 1. Install

```bash
git clone <repo-url> && cd generic-code-reader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and at least one summarization key
source .env
```

**Embedding dimension note**: If you change the embedding model in `r2r/r2r.toml`, ensure `base_dimension` matches: `text-embedding-3-small` → 512, `voyage-code-2` → 1536.

### 3. Start R2R

```bash
docker compose -f r2r/compose.yaml up -d
python preflight.py   # verify R2R + LLM connectivity
```

### 4. Build the knowledge base

```bash
# Simplest: let the agent figure out what to do
python agent.py "Index the knowledge base for /path/to/src"

# With docs and tickets:
python agent.py "Index docs at /path/to/docs, then index codebase at /path/to/src, then fetch Jira tickets for project PROJ"

# Interactive mode:
python agent.py --interactive

# Use a hand-crafted module definition instead of auto-discovery:
#   (agent will call import_modules with the file automatically if you mention it)
python agent.py "Import modules from indexer/analysis_modules.json and summarize /path/to/src"
```

`agent.py` is a Claude agent that orchestrates the pipeline using these tools in order:
1. `index_docs` — docs give context for later code summarization
2. `fetch_tickets` + `process_tickets` — extracts engineering lessons from Jira
3. `discover_modules` (Claude-based, single-shot) or `import_modules` (hand-crafted JSON)
4. `summarize_codebase` — most expensive; runs last

### 5. Open Claude Code

The `.mcp.json` is already configured. Open Claude Code in this directory — `search_codebase` is immediately available. It searches across code summaries, docs, and tickets.

## CLI Fallback (Advanced)

The individual CLI scripts are still available for fine-grained control:

```bash
# Module discovery
python indexer/study_agent.py --codebase /path/to/src --discover
python indexer/study_agent.py --codebase /path/to/src --review   # re-review existing map

# Summarization
python indexer/study_agent.py --codebase /path/to/src --summarize
python indexer/study_agent.py --codebase /path/to/src --refine   # fix vague summaries (~5% cost)
python indexer/study_agent.py --codebase /path/to/src --improve  # rewrite weak summaries (expensive)

# Re-index without re-summarizing (after model change or R2R restart)
python indexer/study_agent.py --codebase /path/to/src --reindex
python -m doc_agent.doc_agent --docs /path/to/docs --reindex
python -m ticket_agent.ticket_agent --reindex

# Docs
python -m doc_agent.doc_agent --docs /path/to/docs
python -m doc_agent.doc_agent --docs /path/to/docs --incremental

# Ticket pipeline (3 steps in order)
python -m ticket_agent.fetch_tickets --project PROJ
python -m ticket_agent.ticket_agent --tickets ticket_agent/tickets/
python -m doc_agent.doc_agent --docs ticket_agent/lessons   # index lessons into R2R

# Ticket fetch options
python -m ticket_agent.fetch_tickets --project PROJ --since -90d
python -m ticket_agent.fetch_tickets --project ABC DEF     # multiple projects
python -m ticket_agent.fetch_tickets --project PROJ --incremental
```

## Tuning

These flags apply to `study_agent.py`:

**`--max-concurrent`** (default: 50) — parallel LLM calls:
- Company shared proxy: `5-10`
- Company dedicated quota: `15-25`
- OpenAI direct: `20-50`
- Local Ollama/vLLM: `2-4`

**`--rpm`** (default: 60) — requests per minute.

**`--exclude`** — skip directories:
```bash
python indexer/study_agent.py --codebase /path/to/src --exclude generated proto_out third_party
```

## Cost and Quota Management

**Estimate before running**: `--dry-run` shows token/cost estimates without LLM calls. The agent also prompts "Proceed? [Y/n]" — skip with `--yes`.

**Quota exhaustion**: Agents halt, save progress, and print a resume message. Re-run the same command to continue. Files in the manifest (`*_hashes.json`) are skipped — zero tokens for completed work.

**Incremental updates**: Re-run with `--incremental` — only changed files are re-processed.

## Post-Run

```bash
python status.py                                # KB health and stats
python -m auditor.auditor                       # check doc↔code conflicts
python dashboard.py --period 7d --by-module     # token savings by module
```

## Clean Slate

```bash
rm -f indexer/module_map.json indexer/summaries.json
rm -f indexer/context_cache.json indexer/call_graph.json indexer/file_hashes.json
rm -f doc_agent/doc_hashes.json
rm -f ticket_agent/ticket_hashes.json ticket_agent/ticket_summaries.json
docker compose -f r2r/compose.yaml down -v && docker compose -f r2r/compose.yaml up -d
python agent.py "Index the knowledge base for /path/to/src"
```

> Changed prompts or models → always reset R2R. Old and new summaries coexist otherwise (the indexer replaces by content hash, but changed prompts produce different text that won't match).

## Evaluating KB Effectiveness

The MCP server logs every `search_codebase` call to `mcp_server/query_log.jsonl`.

```bash
python eval/eval_kb.py extract --query-log mcp_server/query_log.jsonl
python eval/eval_kb.py replay  --query-log mcp_server/query_log.jsonl
python eval/eval_kb.py compare \
    --questions eval/test_questions.jsonl \
    --kb-a http://localhost:7272 --kb-b http://localhost:7273
python eval/eval_kb.py blind --questions eval/test_questions.jsonl
```

## Reviewer Agent (Optional)

`add_to_kb()` indexes entries immediately. The reviewer agent is an optional quality gate for teams that want LLM-based verification of user contributions:

```bash
python reviewer/reviewer_agent.py --codebase /path/to/your/src
python reviewer/reviewer_agent.py --codebase /path/to/your/src --watch   # poll every 30s
```

## Team Setup

Share the KB across your team — one R2R instance, many users.

### 1. Run R2R on a shared server

```bash
# On the server (e.g. a team VM or dev box)
docker compose -f r2r/compose.yaml up -d
```

R2R listens on port 7272. Make sure it's reachable from teammates' machines (internal network or VPN).

### 2. Build the KB once

```bash
# On the server, or from any machine with access
R2R_URL=http://your-server:7272 python agent.py "Index the KB for /path/to/src"
```

### 3. Each teammate: clone repo + configure

```bash
git clone <repo-url> && cd generic-code-reader
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Edit `.mcp.json` to point at the shared server:

```json
{
  "mcpServers": {
    "domain-kb": {
      "command": ".venv/bin/python",
      "args": ["mcp_server/server.py"],
      "env": {
        "R2R_URL": "http://your-server:7272",
        "KB_SEARCH_LIMIT": "5"
      }
    }
  }
}
```

Open Claude Code in the repo directory — `search_codebase` is immediately available, hitting the shared KB.

### 4. Shared staging queue (optional)

`suggest_index_item()` writes to `mcp_server/staging_queue.json` locally by default. For a team, point everyone at a shared path so suggestions from any teammate are reviewed centrally:

```json
"env": {
  "R2R_URL": "http://your-server:7272",
  "STAGING_FILE": "/shared/path/staging_queue.json"
}
```

Run the reviewer on the server (or any machine with access to the shared path):

```bash
STAGING_FILE=/shared/path/staging_queue.json \
python reviewer/reviewer_agent.py --codebase /path/to/src --watch
```

Approved suggestions are promoted into R2R and immediately available to all teammates.

### Security note

R2R has no authentication by default. On an internal network this is usually fine. For external access, put R2R behind a reverse proxy (nginx/Caddy) with basic auth or restrict to VPN only.

## Local Backend (No Docker)

If Docker/R2R is impractical, use the local ChromaDB backend instead. No server needed — just `pip install chromadb`.

### 1. Build the KB as usual (produces JSON files)

```bash
python indexer/study_agent.py --codebase /path/to/src --discover
python indexer/study_agent.py --codebase /path/to/src --summarize
# summaries.json is written but R2R indexing can be skipped
```

### 2. Load into local ChromaDB

```bash
python load_kb.py                           # reads from indexer/, doc_agent/, ticket_agent/
python load_kb.py --kb-dir /path/to/kb      # or from a shared directory/repo
python load_kb.py --clean                   # full reload (clear first)
```

### 3. Configure MCP server for local backend

In the target codebase's `.mcp.json`:

```json
{
  "mcpServers": {
    "domain-kb": {
      "command": "/path/to/generic-code-reader/.venv/bin/python",
      "args": ["/path/to/generic-code-reader/mcp_server/server.py"],
      "env": {
        "KB_BACKEND": "local",
        "LOCAL_KB_DIR": "/path/to/chroma_db"
      }
    }
  }
}
```

### Team sharing via git

Commit the JSON files to a shared repo:

```
kb/
├── summaries.json
├── doc_summaries.json
├── ticket_summaries.json
├── user_contributed.jsonl    ← append-only, from add_to_kb
└── module_map.json
```

Each teammate: `git pull && python load_kb.py --kb-dir kb/ --clean`

User contributions from `add_to_kb()` are logged to `user_contributed.jsonl` (one JSON object per line, git-friendly). Commit and push periodically so teammates pick them up.

## Testing

```bash
python tests/smoke_test.py                        # uses this project as test codebase
python tests/smoke_test.py --codebase /path/to/src
python tests/smoke_test.py --model ollama/llama3.1
```

Smoke tests cover R2R health, LLM connectivity, embeddings, chunking, indexing, Pass 1, Pass 2, MCP search, and the suggest+review loop. Total cost < $0.01.

## Troubleshooting

**R2R won't start / embedding errors**
```bash
curl http://localhost:7272/v3/health
docker compose -f r2r/compose.yaml logs r2r 2>&1 | tail -30
docker compose -f r2r/compose.yaml down && docker compose -f r2r/compose.yaml up -d
```
`Both embedding configurations must use the same dimensions`: check `base_dimension` in `r2r/r2r.toml` matches your model.

**LLM calls failing**
```bash
python -c "from litellm import completion; print(completion(model='openai/gpt-4o', messages=[{'role':'user','content':'hi'}], max_tokens=5).choices[0].message.content)"
```
- Wrong model string: use litellm format — `openai/gpt-4o`, `anthropic/claude-opus-4-5`, `ollama/llama3.1`
- Missing key: set the appropriate `*_API_KEY` and `source .env`
- Company proxy: set `OPENAI_API_BASE` in `.env`

**Discovery finds too few modules**: use at least GPT-4o or Claude Sonnet. With `agent.py`, `discover_modules` is Claude-based and single-shot — more reliable than the old exploration loop.

**Agent killed mid-run**: re-run the same command. Content hashes ensure only incomplete files are re-processed.

**`ModuleNotFoundError: No module named 'shared.utils'`**: the module was renamed to `codebase_shared/`. Use `from codebase_shared.utils import ...`.

## Supported Languages

`python`, `javascript`, `typescript`, `cpp`, `java`, `go`, `rust`

Auto-detected from file extensions. Override with `--language <lang>` if needed.
