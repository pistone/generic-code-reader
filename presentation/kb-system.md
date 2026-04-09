---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: Calibri, sans-serif;
    font-size: 22px;
  }
  h1 { font-size: 44px; }
  h2 { font-size: 34px; }
  table { width: 100%; font-size: 18px; }
  th { background: #0F2A4A; color: white; }
  code { background: #f0f4f8; padding: 2px 6px; border-radius: 4px; }
  pre { background: #1e2d3d; color: #e2e8f0; padding: 20px; border-radius: 8px; }
  .muted { color: #64748b; font-size: 16px; }
---

# Your Codebase, Instantly Searchable
### — for the Whole Team

A shared AI knowledge base that pre-analyses your codebase once —
so the whole team can query it instantly.

---

## The Problem: Claude Can Read Files — But Should It?

| | Ad-hoc (Claude reads files) | With KB |
|---|---|---|
| First query | Slow — reads 10+ files each time | Instant — summaries pre-built |
| Context used | Huge — raw code is verbose | Small — dense summaries |
| 10th query on same code | Same cost again | Same instant retrieval |
| Large codebase | Context overflow risk | KB spans the whole codebase |
| New engineer onboarding | Claude re-reads every session | KB shared, always ready |

The bottleneck isn't Claude's ability — it's paying the reading cost on every query.

---

## What is a Vector DB?

A vector DB stores text as **numbers that capture meaning**, not just the words themselves.

```
"rate limit backoff"  ──► [0.82, 0.14, 0.93, ...]  ─┐
                                                       ├─► nearest neighbours
"retry after 429"     ──► [0.80, 0.15, 0.91, ...]  ─┘
```

- You search with a question in plain English
- The DB finds chunks whose **meaning** is closest — not just keyword matches
- "exponential backoff" finds results that say "wait before retrying" even if the words differ

**Why this matters for code:** function names and variable names are rarely the words engineers use when asking questions. Vector search bridges that gap.

---

## What is MCP?

**Model Context Protocol** — a standard way for Claude to call external tools.

```
Claude ──► [tool: search_codebase("rate limiting")] ──► MCP Server ──► R2R
                                                                          │
Claude ◄── ["RetryPolicy applies backoff after 429..."]  ◄───────────────┘
```

- Claude decides *when* to search — it calls the tool mid-conversation
- The MCP server is a small Python process running alongside Claude Code
- Configured once in `.mcp.json` — Claude Code picks it up automatically
- Any number of tools can be registered: search, suggest, look up tickets, etc.

Think of it as Claude having a well-defined API to call your internal systems.

---

## The Solution: Pre-analyse Once, Search Forever

```
Codebase ──► KB Builder ──► R2R Vector DB ──► Claude + MCP ──► Answers
```

- Claude reads your code **once**, deeply, with full domain context
- Every future query retrieves pre-built summaries in milliseconds
- One KB shared across the whole team — build once, everyone benefits

---

## What Goes Into the KB

**Three sources of knowledge:**

🗂️ **Code Summaries**
Every function, class, and module summarised with domain context — not just docstrings

📚 **Docs & Confluence**
Architecture decisions, runbooks, API specs, design docs

🎫 **Jira Tickets + MR Diffs**
Engineering lessons extracted from real fixes: _"to add a Rust checker, implement CheckerVisitor and register in CheckerRegistry::default()"_

---

## Indexing Pipeline Architecture

```
                        ┌─────────────────────────────────────┐
                        │           agent.py                  │
                        │    (Claude orchestrates tools)       │
                        └──┬──────┬──────┬────────────────────┘
                           │      │      │
              ┌────────────┘      │      └──────────────┐
              ▼                   ▼                      ▼
      ┌──────────────┐   ┌──────────────┐    ┌──────────────────┐
      │ Confluence / │   │ Jira tickets │    │   Codebase       │
      │ local docs   │   │  + MR diffs  │    │ /path/to/src     │
      └──────┬───────┘   └──────┬───────┘    └────────┬─────────┘
             │                  │                      │
             ▼                  ▼                      ▼
      ┌──────────────┐   ┌──────────────┐    ┌──────────────────┐
      │  doc_agent   │   │ ticket_agent │    │  study_agent     │
      │  chunk+summarise  extract lessons    │  discover modules│
      └──────┬───────┘   └──────┬───────┘    │  summarise files │
             │                  │            └────────┬─────────┘
             └──────────────────┴─────────────────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │   R2R Vector DB  │
                      │  (shared server) │
                      └──────────────────┘
```

---

## Building the KB: One Command

```bash
python agent.py "Index the knowledge base for /path/to/src"
```

Claude (the orchestrating agent) decides the sequence:

1. **Index docs & Confluence** — gives context for later code summarisation
2. **Fetch & mine Jira tickets** — extracts engineering lessons from real fixes
3. **Discover modules** — Claude reads the full tree, manifests, READMEs in one shot
4. **Summarise every file** — chunked at AST boundaries, guided by module questions

---

## Module Discovery: Claude Understands Your Codebase

The system asks Claude to read the directory tree, package manifests, READMEs,
and key index files — then identify the meaningful modules in one shot.

Claude is surprisingly good at this. Given the right context it produces
the same breakdown a senior engineer would: not just top-level directories,
but the actual subsystems and their responsibilities.

Each module gets a set of domain-specific questions that guide the summarisation —
_"what triggers a retry?", "what invariants does the cache maintain?"_ —
so summaries are focused on what engineers actually ask about.

---

## Domain-Aware Summaries, Not Just Docstrings

**Without KB context:**
> `def process_batch(items)` → _"Processes a batch of items"_

**With module questions guiding the LLM:**
> _"Applies rate-limit backoff and deduplication before pushing items to the ingestion queue; raises `RetryExhausted` if all 3 attempts fail. Called by the scheduler every 30 s."_

- AST-aware chunking respects function/class boundaries
- Module questions focus the LLM on what matters
- LLM searches the KB for cross-module types it doesn't recognise

---

## Engineering Lessons from Real Fixes

```
Jira ticket  ──►  linked MR diff  ──►  extracted lesson
```

Example lesson stored in KB:

> **To add a new Rust checker:**
> Implement `CheckerVisitor` trait in `src/checkers/`,
> register in `CheckerRegistry::default()`,
> add test fixtures under `tests/checkers/`.
> See `NullDerefChecker` as the canonical example.

These lessons answer "how do I do X in this codebase?" — the question
every new engineer asks and every senior engineer re-answers.

---

## Team Architecture: Claude + MCP + Review Loop

```
  Engineer A            Engineer B            Engineer C
  (Claude Code)         (Claude Code)         (Claude Code)
       │                     │                     │
       └──────────┬──────────┘                     │
                  │           MCP (search_codebase) │
                  ▼                                 │
        ┌──────────────────┐                        │
        │   R2R Vector DB  │ ◄──────────────────────┘
        │  (shared server) │
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │  staging_queue   │ ◄── suggest_index_item() from any engineer
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ reviewer_agent   │  verifies accuracy, deduplicates
        └────────┬─────────┘
                 │ approved
                 ▼
        ┌──────────────────┐
        │   R2R Vector DB  │  updated — all engineers get it instantly
        └──────────────────┘
```

---

## Using It: search_codebase in Claude Code

The `.mcp.json` is already configured. Open Claude Code in the repo — the tool is immediately available.

**Engineer asks:**
> "Where does rate limiting happen and what triggers a retry?"

**Claude searches KB, answers instantly:**
> "Rate limiting is in `src/http/retry.py` — `RetryPolicy` applies exponential backoff after 429s, gives up after `MAX_RETRIES=3` with `RetryExhausted`. Called from `HttpClient.send()` in `src/http/client.py`."

No file reading. No context overflow. No waiting.

---

## One KB, Whole Team

**Setup is three steps:**

**1. Run R2R on a shared server**
```bash
docker compose -f r2r/compose.yaml up -d
R2R_URL=http://your-server:7272 python agent.py "Index /path/to/src"
```

**2. Each teammate: clone + configure `.mcp.json`**
```json
{ "env": { "R2R_URL": "http://your-server:7272" } }
```
Open Claude Code — `search_codebase` is ready. No local build needed.

**3. Shared suggestion queue**
`suggest_index_item()` from any teammate → central reviewer → R2R

<span class="muted">R2R has no auth by default — keep it on your internal network or VPN.</span>

---

## Getting Started

```bash
# 1. Start R2R
docker compose -f r2r/compose.yaml up -d

# 2. Configure
cp .env.example .env   # set ANTHROPIC_API_KEY + LLM key
source .env && python preflight.py

# 3. Build the KB
python agent.py "Index the knowledge base for /path/to/src"

# 4. Open Claude Code in this directory
# search_codebase is ready
```

Questions? Happy to walk through any part in more detail.
