# Knowledge Base

This codebase has a pre-built domain knowledge base powered by semantic search.
Before reading source files to answer questions, **search the KB first** — it
returns targeted summaries in ~500 tokens instead of reading dozens of files.

## MCP Tools

| Tool | When to use |
|------|-------------|
| `search_codebase(query)` | **Always try first.** Semantic search over code summaries, documentation, and ticket knowledge. Returns the most relevant entries ranked by similarity. |
| `add_to_kb(topic, summary, source_files, reasoning, raw_code)` | When you've researched something not in the KB and found the answer. Indexes immediately — available to the whole team right away. |
| `kb_status()` | Check what's indexed, R2R health, and pending items. |
| `list_modules()` | See all modules, their descriptions, and domain questions. Useful for orientation. |

## Workflow

1. **Question comes in** → call `search_codebase` with a natural language query
2. **Good results?** → answer directly from the summaries
3. **Need more detail?** → `Read` the specific source files mentioned in the results
4. **Found something new?** → call `add_to_kb` so future queries benefit

## Tips

- Use domain terms in queries, not generic language. "authentication middleware" finds more than "how do users log in".
- If a search returns nothing useful, try rephrasing — vector search is meaning-based, not keyword-based.
- `list_modules` gives you the vocabulary the KB uses. Align your queries to module names and domain questions for better hits.
- The KB includes code summaries, documentation, and ticket/incident knowledge. If you only get code results, there may be relevant docs too — the tool will note this.
