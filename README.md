# Generic Code Reader

A self-improving domain knowledge base for codebases. Indexes LLM-generated summaries into a vector database (R2R) and exposes them as an MCP server for Claude Code.

## Architecture

```
doc_agent/doc_agent.py       → Ingests design docs, runbooks, wiki pages into R2R
indexer/study_agent.py       → Analyzes codebase, generates summaries (multi-pass with RAG)
indexer/indexer.py           → Feeds summaries into R2R vector DB
ticket_agent/ticket_agent.py → Extracts knowledge from Jira/PR ticket exports
auditor/auditor.py           → Detects doc↔code conflicts (staleness, contradictions)
mcp_server/server.py         → MCP server: search_codebase + suggest_index_item
reviewer/reviewer_agent.py   → Verifies runtime suggestions before promoting to the KB
```

The self-improving loop: when Claude can't find an answer in the KB, it researches manually and calls `suggest_index_item()`. The reviewer agent verifies the suggestion and promotes it into R2R, so the next developer gets an instant answer.

## Setup

### 1. Clone and create virtualenv

```bash
git clone <repo-url> && cd generic-code-reader
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   OPENAI_API_KEY   (for the LLM — or use Ollama for fully local)
#   VOYAGE_API_KEY   (for R2R embeddings)
```

### 3. Start R2R (vector database)

```bash
source .env
docker compose -f r2r/compose.yaml up -d
# Verify: curl http://localhost:7272/v3/health
```

### 4. (Optional) Index design documents

If your project has design docs, architecture docs, runbooks, or wiki exports:

```bash
python -m doc_agent.doc_agent --docs /path/to/docs

# Re-run only on changed docs
python -m doc_agent.doc_agent --docs /path/to/docs --incremental
```

Run this **before** the study agent so RAG-augmented summarization can
reference your documentation.

### 5. Study the codebase

```bash
# Basic: two-pass analysis (module discovery → summarization)
python indexer/study_agent.py --codebase /path/to/your/src --language python

# With RAG augmentation (recommended after indexing docs):
python indexer/study_agent.py --codebase /path/to/your/src --rag --passes 3

# Quick test with 20 chunks
python indexer/study_agent.py --codebase /path/to/your/src --max-chunks 20

# Re-run only on changed files (uses sha256 hash manifest)
python indexer/study_agent.py --codebase /path/to/your/src --incremental
```

Token usage is printed at the end of each run and logged to `indexer/cost_log.jsonl`.

### 6. Index summaries into R2R

```bash
# If you ran with --passes 1 (default), index manually:
python indexer/indexer.py --index indexer/summaries.json

# If you ran with --passes > 1, summaries are already indexed.

# Verify:
python indexer/indexer.py --search "your query here"
```

### 7. (Optional) Index ticket knowledge

If you have exported Jira tickets, PR discussions, or similar:

```bash
# Export tickets with your existing tools, then:
python -m ticket_agent.ticket_agent --tickets /path/to/exported/tickets

# Incremental (skip already-processed tickets)
python -m ticket_agent.ticket_agent --tickets /path/to/tickets --incremental
```

The ticket agent filters aggressively (only resolved tickets with comments),
uses LLM to extract root causes, workarounds, and design decisions, then
deduplicates against the existing KB before indexing.

### 8. (Optional) Audit doc↔code consistency

After indexing both docs and code, check for contradictions:

```bash
python -m auditor.auditor

# Custom staleness threshold (default: 90 days)
python -m auditor.auditor --threshold-days 60

# Results in auditor/conflict_report.json
```

The auditor compares doc entries against code entries using timestamps
first, then LLM comparison on flagged pairs. Review the conflict report
to identify stale documentation.

### 9. Use with Claude Code

The `.mcp.json` is already configured. Open Claude Code in this directory and the `search_codebase` tool will be available.

### 10. (Optional) Run the reviewer agent

```bash
# Process pending suggestions once
python reviewer/reviewer_agent.py --codebase /path/to/your/src

# Or keep it running in watch mode
python reviewer/reviewer_agent.py --codebase /path/to/your/src --watch
```

## Configuration

All scripts respect these environment variables:

| Variable | Default | Used by |
|----------|---------|---------|
| `R2R_URL` | `http://localhost:7272` | All scripts |
| `LLM_MODEL` | `openai/gpt-4o` | study_agent, reviewer_agent |
| `OPENAI_API_KEY` | — | LLM calls (if using OpenAI) |
| `VOYAGE_API_KEY` | — | R2R embeddings |
| `KB_SEARCH_LIMIT` | `5` | MCP server (max results per search) |

For fully local operation (no API keys): use Ollama + swap R2R embeddings to a local model.

```bash
# Ollama example
ollama pull llama3.1
python indexer/study_agent.py --codebase /path/to/src --model ollama/llama3.1
```

## Testing

Run the smoke tests after setup to validate everything works before applying to your codebase:

```bash
# Uses this project itself as the test codebase
python tests/smoke_test.py

# Use a specific directory
python tests/smoke_test.py --codebase /path/to/your/src

# Different LLM provider
python tests/smoke_test.py --model ollama/llama3.1
```

The smoke tests check, in order:

1. R2R health (Docker is up)
2. LLM connectivity (API key works)
3. Embedding round-trip (Voyage embeddings work)
4. Code chunking (tree-sitter AST splitting)
5. Index + search round-trip (R2R stores and retrieves)
6. Study agent Pass 1 (module discovery)
7. Study agent Pass 2 (summarization)
8. Index summaries (end-to-end indexing)
9. MCP server search (tool callable)
10. Suggest + review loop (self-improving pipeline)

Tests clean up after themselves. Total LLM cost is under $0.01.

## Supported Languages

`python`, `javascript`, `typescript`, `cpp`, `java`, `go`, `rust`

Pass `--language <lang>` to the study agent. Defaults to `python`.
