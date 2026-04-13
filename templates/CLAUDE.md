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

## When to search the KB during exploration

Answering a question often involves sub-questions. Use the KB throughout, not
just at the start — but be smart about when a search beats a file read:

**Search the KB when:**
- The sub-question crosses into a different module or area of the codebase
  ("how does module X call module Y?", "where is this type defined?")
- You need to find something but don't know which file it's in
- You want to understand the purpose or contract of a class/function before
  reading its implementation
- The sub-question is about documentation, tickets, or design decisions

**Read files directly when:**
- You already know the exact file and need specific lines or details
- You're deep-diving into one file's implementation (read once, answer many
  sub-questions from the same content)
- You're looking at usage examples in a file the KB already pointed you to

**Rule of thumb:** if you're about to `Grep` the whole codebase for something,
`search_codebase` first — it's faster and cheaper. If you're about to `Read`
a file the KB already identified, just read it.

## Important

- To read source files referenced in search results, use the `Read` tool directly — do NOT use `readMcpResource`. The KB server provides search only, not file access.

## Tips

- Use domain terms in queries, not generic language. "authentication middleware" finds more than "how do users log in".
- If a search returns nothing useful, try rephrasing — vector search is meaning-based, not keyword-based.
- `list_modules` gives you the vocabulary the KB uses. Align your queries to module names and domain questions for better hits.
- The KB includes code summaries, documentation, and ticket/incident knowledge. If you only get code results, there may be relevant docs too — the tool will note this.
