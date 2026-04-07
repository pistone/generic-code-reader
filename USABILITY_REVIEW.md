# Usability Review: generic-code-reader

## How I Expect a User to Use This Tool

### Persona A: Senior engineer onboarding to a large C++ codebase (primary target)

1. Clone the repo, set up venv, install deps
2. Configure `.env` with their company's LLM endpoint
3. Start R2R via Docker
4. Run `preflight.py`
5. Run `study_agent.py --dry-run` to estimate cost
6. Run `study_agent.py` with appropriate flags
7. Open Claude Code — start asking questions

### Persona B: Team lead setting this up for the whole team

Same as above, plus:
- Index design docs via `doc_agent`
- Index resolved tickets via `ticket_agent`
- Set up `reviewer_agent --watch` as a background service
- Run `auditor` periodically

### Persona C: Developer re-running after code changes

- `study_agent.py --incremental` to update changed files
- Maybe `doc_agent --incremental` if docs changed

---

## The User Journey Today (with friction points marked)

### Step 1: First Contact (README)

The Quick Start block is good — 9 lines, copy-paste ready. But:

- **F1: No "what will this cost me?"** — The user has to run `--dry-run` to find out, but doesn't know that until reading further. For a 10K-file C++ codebase, this could be $25-$300 depending on model. That's important to know upfront.

- **F2: `source .env` is easy to forget.** Every new shell needs it. If forgotten, LLM calls fail with a cryptic litellm error about missing API keys, not "did you `source .env`?"

- **F3: Docker startup takes time.** `docker compose up -d` returns immediately but R2R takes 15-30 seconds to be healthy. If the user runs `preflight.py` too fast, R2R check fails. No mention of this.

- **F4: ~~Step 6 in README says "Index summaries into R2R" manually for `--passes 1`.~~** Fixed — summaries are auto-indexed at the end of every run.

### Step 2: Configuration

- **F5: Two config files to understand.** `.env` for API keys + `r2r/r2r.toml` for embeddings. The embedding dimension mismatch error is a real trap.

- **F6: Company proxy setup is underdocumented.** The user mentioned they have a LiteLLM proxy with Bedrock. The `.env.example` mentions `OPENAI_API_BASE` but doesn't explain how to set up `bedrock/` model strings or authentication.

### Step 3: First Run (study_agent)

- **F7: ~~Too many flags.~~** Reduced — removed `--passes`, `--workers`, `--max-chunks`, `--include-tests`, `--bootstrap-docs`. Language is auto-detected. A first-time user only needs `--codebase`.

- **F8: `--language` is required knowledge.** If the user forgets it, it defaults to `python` and finds 0 files in a C++ codebase. The error is just "No source files found" with no hint about `--language`.

- **F9: No progress percentage for Pass 1.** The module discovery agent runs for minutes with cryptic "Round N: expand_dirs(...)" output. The user has no idea how long this will take or if it's stuck.

- **F10: Wall time not shown.** For a 10K-file run at 60 RPM, the user needs to know "this will take ~3 hours." The `--dry-run` estimates it, but the running agent doesn't show an overall ETA until chunk summarization starts.

### Step 4: Ongoing Usage

- **F11: No single "update" command.** After code changes, the user needs to remember to run `study_agent --incremental`, possibly `doc_agent --incremental`, and maybe `auditor`. This could be one command.

- **F12: No status command.** "How many files are indexed? When was the last run? How much did it cost total?" — there's no way to check this without reading JSON files manually.

### Step 5: Claude Code Integration

- **F13: MCP server starts but user doesn't know it works.** No feedback when Claude Code connects to the domain-kb server. If R2R is down, the first search silently fails.

- **F14: No module listing.** Claude Code (and the user) can't discover what modules exist or what's indexed without searching.

- **F15: Suggestion feedback loop is invisible.** When Claude suggests an item, it goes to `staging_queue.json`. The user has to know to run the reviewer agent. No notification, no status.

---

## Suggestions (Ordered by Impact)

### Tier 1: Remove Footguns (High impact, low effort)

**S1: Auto-detect language from file extensions.**
Scan the codebase for file types and pick the dominant one. Fall back to `--language` if ambiguous. Print "Detected: C++ (8,234 .cpp/.h files)".

**S2: ~~Auto-index after `--passes 1`.~~** Done — summaries are always auto-indexed. `--index-only` available for manual re-indexing.

**S3: Check `source .env` on startup.**
At the top of every agent's `main()`, check if `LLM_MODEL` or at least one API key env var is set. If not, print: "No LLM configuration found. Did you run `source .env`?"

**S4: Wait for R2R in preflight.**
Instead of a one-shot health check, retry for up to 30 seconds with a spinner: "Waiting for R2R to be ready... (15s)". Docker compose just started — this is expected.

**S5: Show cost estimate before proceeding.**
After collecting source files, before any LLM call, print the cost estimate (same as `--dry-run`) and ask "Proceed? [Y/n]" unless `--yes` is passed. This prevents surprise bills.

### Tier 2: Smooth the Happy Path (Medium impact, medium effort)

**S6: Add a `run.py` orchestrator.**
One command that does everything:
```bash
python run.py --codebase /path/to/src
# Automatically: detects language, estimates cost, runs study_agent,
# indexes into R2R, prints summary
```
Power users still use individual agents directly. But the first experience should be one command.

**S7: Add `status.py` command.**
```bash
python status.py
# KB Status:
#   Code: 8,234 files, 24,567 chunks indexed (last run: 2h ago)
#   Docs: 45 files, 312 chunks indexed
#   Tickets: 200 entries indexed
#   Total cost so far: $47.23
#   R2R: healthy (localhost:7272)
#   MCP: configured (.mcp.json present)
```

**S8: Better Pass 1 progress.**
Instead of "Round 3: expand_dirs(checkers/)", show:
```
[Pass 1] Exploring codebase structure...
  Discovered 12 modules so far (234/8234 files mapped)
  Current: analyzing checkers/ directory
```

**S9: Show overall ETA from the start.**
After collecting files, print: "Estimated time: ~2h 15m at 60 RPM (use --rpm 500 --max-concurrent 50 for ~15m)". Show this before chunk summarization begins, not just in `--dry-run`.

**S10: Unify `--incremental` across all agents.**
One place to run incremental updates:
```bash
python run.py --codebase /path/to/src --incremental
# Updates: study_agent (3 files changed), doc_agent (1 doc changed)
```

### Tier 3: Polish (Lower impact, nice-to-have)

**S11: Add MCP introspection tools.**
Add `kb_status()` and `list_modules()` MCP tools so Claude Code can self-diagnose:
```python
@mcp.tool()
def kb_status() -> str:
    """Check knowledge base health and stats."""
    # Returns: indexed count, R2R health, last update time

@mcp.tool()
def list_modules() -> str:
    """List all indexed modules with file counts."""
```

**S12: Smart RPM detection.**
If the user's company LiteLLM proxy returns rate limit headers (`x-ratelimit-limit-requests`), parse them and auto-set `--rpm`. Print: "Detected rate limit: 500 RPM from proxy headers."

**S13: Colored terminal output.**
Use ANSI colors for progress: green for success, yellow for warnings, red for errors. Currently everything is monochrome.

**S14: Post-run summary card.**
At the end of study_agent, print a clear summary:
```
╔══════════════════════════════════════╗
║  Study Complete                      ║
║  Files: 8,234  Chunks: 24,567       ║
║  Cost: $47.23  Time: 2h 14m         ║
║  Quality: 92% (186 refined)         ║
║                                      ║
║  Next: Open Claude Code in this dir  ║
║  Search: "how does X work?"          ║
╚══════════════════════════════════════╝
```

**S15: Consistent flag naming.**
- doc_agent lacks `--verbose` (other agents have it)
- study_agent uses `--output-dir` (directory) but auditor uses `--output` (file)
- Normalize to `--output-dir` everywhere, with per-agent defaults

**S16: Reviewer notification.**
When `suggest_index_item` is called, print a note: "Suggestion queued. Run `python reviewer/reviewer_agent.py --codebase /path` to review (or use --watch for continuous review)." Currently just "Suggestion queued (#N in staging)."

---

## Summary

The tool is powerful and well-engineered internally. The main usability gaps are:

| Area | Grade | Key Issue |
|------|-------|-----------|
| First-time setup | B- | Too many decisions before first run |
| Running the tool | B | Flag overload, missing auto-detect |
| Progress feedback | B- | Pass 1 opaque, no overall ETA |
| Error messages | B+ | Good but missing "did you source .env?" |
| Claude Code integration | B+ | Works well but no introspection |
| Ongoing maintenance | C+ | No status command, no unified update |
| Documentation | B+ | Good README but missing cost/time guidance |

The single highest-impact change would be **S6: a `run.py` orchestrator** that reduces the first experience from "read README, configure 2 files, run 4 commands" to "run one command."
