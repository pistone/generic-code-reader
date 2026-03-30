# Deployment Guide — Running on a Large Codebase

## Prerequisites

You need: Python 3.11+, Docker (for R2R), and at least one LLM API key.

The tool reads environment variables — if they're already in your shell
(e.g. from company tooling, a secrets manager, or `.bashrc`), you do NOT
need a `.env` file. The variables it looks for:

| Variable | Required? | Default |
|----------|-----------|---------|
| `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GROQ_API_KEY`) | At least one | — |
| `OPENAI_API_BASE` | Only if using a company LLM proxy | — |
| `LLM_MODEL` | No | `openai/gpt-4o` |
| `R2R_URL` | No | `http://localhost:7272` |

## Quick start

```bash
# 1. Clone and install
git clone <repo> && cd generic-code-reader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start R2R (vector database)
cd r2r && docker compose up -d && cd ..
# Wait ~30s for it to be healthy. Verify:
curl http://localhost:7272/v3/health

# 3. Run preflight check (verifies env vars, R2R, model access)
python preflight.py

# 4. Run everything in one command
python run.py \
    --codebase /path/to/src \
    --docs /path/to/design-docs \
    --tickets /path/to/jira-export \
    --model openai/gpt-4o \
    --model-fast openai/gpt-4o-mini \
    --max-concurrent 10 \
    --rpm 100
```

With no special flags, this runs the full pipeline:
doc_agent (if --docs) → Pass 1 (module discovery) → Pass 2
(summarization) → Index to R2R → ticket_agent (if --tickets).

Multiple codebases can be analyzed together:
```bash
python run.py --codebase /path/to/src/core /path/to/src/plugins
```

## Starting from a clean slate

To re-run the full pipeline from scratch (e.g., after changing the
model, adjusting prompts, or wanting a fresh KB):

```bash
# 1. Delete all study agent artifacts
rm -f indexer/module_map.json indexer/summaries.json
rm -f indexer/context_cache.json indexer/call_graph.json
rm -f indexer/file_hashes.json

# 2. Delete doc agent artifacts (if using --docs)
rm -f doc_agent/doc_hashes.json

# 3. Delete ticket agent artifacts (if using --tickets)
rm -f ticket_agent/ticket_hashes.json ticket_agent/ticket_summaries.json

# 4. Clear R2R (drop all indexed data)
cd r2r && docker compose down -v && docker compose up -d && cd ..

# 5. Re-run
python run.py --codebase /path/to/src ...
```

If you only want to re-run a specific phase (e.g., Pass 1 gave bad
results but Pass 2 was fine), delete only that phase's artifact:
- Bad module map? Delete `module_map.json`, re-run with `--pass1-only`
- Bad summaries? Delete `summaries.json` and `context_cache.json`,
  re-run with `--pass2-only`
- Just want to re-index? `--index-only` (no deletion needed)

**Important**: If you don't clear R2R before a full re-run, old
summaries stay alongside new ones (duplicates). The indexer replaces
entries by content hash, but changed prompts produce different summaries
that won't match. Always reset R2R for a true clean slate.

## Running each phase separately (debugging)

For large codebases, it helps to run each phase independently so
you can inspect output before proceeding:

```bash
# Phase 1: Module discovery only
python indexer/study_agent.py --codebase /path/to/src --pass1-only
# → Inspect indexer/module_map.json. Happy? Continue:

# Phase 2: Summarization only (reads module_map.json)
python indexer/study_agent.py --codebase /path/to/src --pass2-only
# → Inspect indexer/summaries.json. Want to improve vague ones:

# Phase 3 (optional): Targeted refine — only re-summarizes chunks
# with vague references ("delegates to", "defined elsewhere", etc.)
python indexer/study_agent.py --codebase /path/to/src --refine
# → Much cheaper than full review: touches ~5% of summaries

# Phase 4 (optional): Full review — re-reads every summary
python indexer/study_agent.py --codebase /path/to/src --review-only

# Phase 5: Index into R2R
python indexer/study_agent.py --codebase /path/to/src --index-only
```

## Tuning for your environment

**`--max-concurrent`**: How many parallel LLM calls.
- Company shared proxy: `5-10` (don't hog shared quota)
- Company dedicated quota: `15-25`
- OpenAI direct: `20-50`
- Local Ollama/vLLM: `2-4` (GPU-bottlenecked)

**`--rpm`**: Requests per minute cap. Start with `100` for company
proxies, increase if no 429 errors. OpenAI Tier 1 = 500 RPM.

**`--exclude`**: Skip directories that shouldn't be indexed:
```bash
python run.py --codebase /path/to/src \
    --exclude generated proto_out third_party test
```

**`--codebase` can point to a subdirectory**: If you only want to index
`src/core/engine/`, point `--codebase` there. Cross-file references
outside that subtree won't be resolved (e.g., `#include "networking/socket.h"`
from a sibling directory). C++ includes depend on build system `-I` flags
and cannot be reliably resolved statically.

## Quota and cost management

**Cost estimation**: Before processing, the tool shows an estimated cost
and prompts "Proceed? [Y/n]". Use `--yes` to skip the prompt.

**Dry run**: `--dry-run` shows what would happen without making LLM calls.
Cost estimates use litellm's pricing database. If your model isn't in
litellm's database (e.g., a custom company endpoint), the estimate will
show "unknown" — the tool still works, it just can't predict cost.

**Quota exhaustion**: If the LLM returns a quota/budget error mid-run,
all agents halt gracefully, save progress, and print a resume message.
Re-run the same command to continue from where it stopped. The manifest
files (`*_hashes.json`) track what's done.

**Resume behavior**: Files already in the manifest with matching content
hash are skipped. Only incomplete/new files are re-processed. You pay
zero tokens for already-completed files.

## Common issues

**Pass 1 finds too few modules (e.g., 2 modules for a 10K-file repo)**:
This means the LLM didn't explore enough before concluding. The current
defaults (20 exploration rounds, depth-scaled tree) should prevent this.
If it still happens:
- Check the model — smaller/cheaper models are more likely to rush.
  Use at least GPT-4o or Claude Sonnet for Pass 1.
- Check the tree output in the logs — if the initial tree is too shallow,
  the LLM doesn't see the structure it needs to explore.
- You can re-run Pass 1 only with `--pass1-only` to iterate quickly.

**Doc agent is slow**: The doc agent uses async concurrent summarization
(default 20 concurrent). If it's still slow:
- Increase `--max-concurrent` (if your LLM endpoint allows)
- Increase `--rpm` (if no 429 errors)
- Use `--no-summarize` for a fast first pass (indexes raw text, lower
  search quality but immediate)

**Study agent killed mid-run**: Re-run the same command. It reads
`summaries.json` and the file hash manifest to skip completed work.
Content hashes verify integrity — if a file was edited between runs,
its cached summary is discarded and the file is re-processed.

**R2R not responding**: `docker compose -f r2r/compose.yaml ps` to check.
It takes ~30s to start. `run.py` retries for 30s automatically.

## Post-run

```bash
# Check KB health
python status.py

# Start MCP server for Claude Code
# (already configured in .mcp.json — just open Claude Code in this directory)

# Run auditor to check for doc↔code conflicts
python -m auditor.auditor --model openai/gpt-4o-mini

# Measure KB effectiveness over time
python dashboard.py --period 7d --by-module
```

## Incremental updates

When source code changes, re-run with `--incremental`:
```bash
python run.py --codebase /path/to/src --incremental
```
Only files with changed content hashes are re-processed.

## Explorer agent (alternative)

For targeted deep exploration of a specific area rather than broad
coverage:

```bash
python -m explorer_agent.explorer_agent \
    --codebase /path/to/src/core/engine \
    --context-root /path/to/src
```

Output is compatible with study agent's `summaries.json` schema.

## Evaluating KB effectiveness

The MCP server logs every `search_codebase` call (question + answer)
to `mcp_server/query_log.jsonl`.

```bash
# Extract benchmark from real usage
python eval/eval_kb.py extract --query-log mcp_server/query_log.jsonl

# Replay queries against current KB
python eval/eval_kb.py replay --query-log mcp_server/query_log.jsonl

# Compare two KBs (e.g., old vs new)
python eval/eval_kb.py compare \
    --questions eval/test_questions.jsonl \
    --kb-a http://localhost:7272 --kb-b http://localhost:7273

# Score KB quality against expected files
python eval/eval_kb.py blind --questions eval/test_questions.jsonl
```

## Token savings dashboard

```bash
python dashboard.py                     # terminal summary
python dashboard.py --period 7d         # last 7 days only
python dashboard.py --by-module         # breakdown by module
python dashboard.py --json              # machine-readable output
```
