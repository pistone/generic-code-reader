# Knowledge Base

This codebase has a domain knowledge base. Use it.

- **`search_codebase(query)`** — search before reading files or grepping. Covers code summaries, docs, and tickets.
- **`add_to_kb(topic, summary, source_files, reasoning, raw_code)`** — when you find something not in the KB, index it for the team.
- **`kb_status()`** / **`list_modules()`** — check what's indexed.

Search the KB whenever you'd otherwise grep the whole codebase or don't know which file to read. Use it throughout multi-step exploration, not just at the start — especially when crossing into a different module. Read files directly only when you already know the exact file.

Do NOT use `readMcpResource` to read files — use the `Read` tool.
