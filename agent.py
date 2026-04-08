#!/usr/bin/env python3
"""
agent.py — Orchestrating agent for building and maintaining the knowledge base.

Uses Claude with tool use to decide which tools to call based on the user's goal.

Usage:
  python agent.py "Index the knowledge base for project ABC codebase at /path/to/src"
  python agent.py "Download Confluence space ENG and index it"
  python agent.py "Fetch Jira tickets for project ABC and extract lessons"
  python agent.py --interactive   # conversational mode

Env vars required depend on which tools get called:
  ANTHROPIC_API_KEY   Always required (for the orchestrating agent)
  LLM_MODEL          Optional override for summarization model (default: openai/gpt-4o)
  R2R_URL            R2R knowledge base (default: http://localhost:7272)
  JIRA_URL, JIRA_EMAIL, JIRA_TOKEN     For ticket fetching
  CONFLUENCE_URL, CONFLUENCE_EMAIL, CONFLUENCE_TOKEN  For Confluence
  GITHUB_TOKEN / GITLAB_TOKEN          For MR/PR diff fetching
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import anthropic

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "discover_modules",
        "description": (
            "Analyse a codebase and identify its module structure using Claude. "
            "Claude reads the directory tree, manifests, READMEs, and index files "
            "in one shot and produces a module map with domain-specific questions. "
            "This is the first step before summarize_codebase. "
            "Writes module_map.json. Prefer this over manual module definition unless "
            "the user has provided a definition file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "codebase": {
                    "type": "string",
                    "description": "Absolute path to the codebase root directory.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (python, javascript, rust, go, java, cpp, auto).",
                    "default": "auto",
                },
                "docs_context": {
                    "type": "string",
                    "description": "Optional architecture documentation to include as extra context.",
                },
            },
            "required": ["codebase"],
        },
    },
    {
        "name": "import_modules",
        "description": (
            "Import module definitions from a hand-crafted JSON file and generate "
            "questions for each module. Use this when the user has already defined "
            "the module structure (e.g. indexer/analysis_modules.json). "
            "The JSON format: {\"modules\": [{\"name\": \"...\", \"description\": \"...\", "
            "\"directories\": [\"src/core\"]}]}. "
            "Writes module_map.json."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "definition_file": {
                    "type": "string",
                    "description": "Path to the module definition JSON file.",
                },
                "codebase": {
                    "type": "string",
                    "description": "Absolute path to the codebase root directory.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language.",
                    "default": "auto",
                },
            },
            "required": ["definition_file", "codebase"],
        },
    },
    {
        "name": "summarize_codebase",
        "description": (
            "Chunk each source file at AST boundaries and generate domain-aware summaries "
            "using the module map from discover_modules or import_modules. "
            "Automatically indexes all summaries into R2R. "
            "Requires module_map.json to exist (run discover_modules first). "
            "This is the most expensive step — estimate ~3 LLM calls per file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "codebase": {
                    "type": "string",
                    "description": "Absolute path to the codebase root directory.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language.",
                    "default": "auto",
                },
                "model": {
                    "type": "string",
                    "description": "LLM model for summarization. Use a fast/cheap model here "
                                   "(e.g. openai/gpt-4o-mini). Defaults to LLM_MODEL env var.",
                },
                "rpm": {
                    "type": "integer",
                    "description": "API rate limit in requests/minute.",
                    "default": 60,
                },
            },
            "required": ["codebase"],
        },
    },
    {
        "name": "fetch_tickets",
        "description": (
            "Fetch resolved Jira tickets for one or more projects and save them to disk. "
            "Requires JIRA_URL, JIRA_EMAIL, JIRA_TOKEN environment variables. "
            "Run process_tickets afterwards to extract lessons."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Jira project key (e.g. 'PROJ') or space-separated list ('PROJ ABC').",
                },
                "since": {
                    "type": "string",
                    "description": "How far back to fetch (e.g. '-365d', '-90d', '2024-01-01').",
                    "default": "-365d",
                },
                "key_pattern": {
                    "type": "string",
                    "description": "Optional regex to filter ticket keys (e.g. 'ABC-\\d+').",
                },
                "jql": {
                    "type": "string",
                    "description": "Override JQL query (ignores project/since if set).",
                },
            },
            "required": ["project"],
        },
    },
    {
        "name": "process_tickets",
        "description": (
            "Extract generalizable lessons from Jira tickets and their linked "
            "GitHub PR / GitLab MR diffs. Summarizes each ticket's solution as a "
            "how-to recipe and indexes them into R2R. "
            "Requires GITHUB_TOKEN or GITLAB_TOKEN for MR diff fetching. "
            "Run fetch_tickets first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key_pattern": {
                    "type": "string",
                    "description": "Regex to filter ticket keys (e.g. 'ABC-\\d+').",
                },
                "no_mr": {
                    "type": "boolean",
                    "description": "Skip MR/PR diff fetching. Not recommended — MRs are the main signal.",
                    "default": False,
                },
            },
            "required": [],
        },
    },
    {
        "name": "index_docs",
        "description": (
            "Parse, summarize, and index documentation files into R2R. "
            "Supports .md, .html, .rst, .txt files. "
            "Can take individual files or directories (recursively scanned). "
            "Run this before summarize_codebase so the codebase summaries can "
            "look up documentation context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file or directory paths to index.",
                },
                "model": {
                    "type": "string",
                    "description": "LLM model for summarization.",
                },
            },
            "required": ["paths"],
        },
    },
    {
        "name": "download_confluence",
        "description": (
            "Download pages from a Confluence space as Markdown files and optionally "
            "index them into R2R. "
            "Requires CONFLUENCE_URL, CONFLUENCE_EMAIL, CONFLUENCE_TOKEN env vars. "
            "Use also_index=true to combine download + indexing in one step."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "space": {
                    "type": "string",
                    "description": "Confluence space key (e.g. 'ENG', 'ARCH').",
                },
                "page_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific page IDs to fetch. Fetches whole space if omitted.",
                },
                "include_pattern": {
                    "type": "string",
                    "description": "Regex: only include pages whose title matches.",
                },
                "exclude_pattern": {
                    "type": "string",
                    "description": "Regex: exclude pages whose title matches.",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum number of pages to download.",
                    "default": 500,
                },
                "also_index": {
                    "type": "boolean",
                    "description": "Index downloaded pages into R2R immediately.",
                    "default": True,
                },
            },
            "required": ["space"],
        },
    },
    {
        "name": "search_kb",
        "description": (
            "Search the R2R knowledge base. Use this to verify what has been indexed "
            "or to answer questions about the codebase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
]

# ── Tool dispatch ─────────────────────────────────────────────────────────────

def _dispatch_tool(name: str, inputs: dict) -> Any:
    """Call the appropriate tool function and return a JSON-serializable result."""
    try:
        if name == "discover_modules":
            from tools.codebase import discover_modules
            return discover_modules(**inputs)

        elif name == "import_modules":
            from tools.codebase import import_modules
            return import_modules(**inputs)

        elif name == "summarize_codebase":
            from tools.codebase import summarize_codebase
            return summarize_codebase(**inputs)

        elif name == "fetch_tickets":
            from tools.tickets import fetch_tickets
            return fetch_tickets(**inputs)

        elif name == "process_tickets":
            from tools.tickets import process_tickets
            return process_tickets(**inputs)

        elif name == "index_docs":
            from tools.docs import index_docs
            return index_docs(**inputs)

        elif name == "download_confluence":
            from tools.confluence import download_confluence
            return download_confluence(**inputs)

        elif name == "search_kb":
            from codebase_shared.r2r_indexer import get_client
            client = get_client()
            results = client.retrieval.search(
                query=inputs["query"],
                search_settings={"limit": inputs.get("limit", 5)},
            )
            chunks = results.results.chunk_search_results or []
            return [{"score": c.score, "text": c.text[:500]} for c in chunks]

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"error": str(e), "tool": name}


# ── Agent system prompt ───────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert knowledge base builder. Your job is to index a software project's \
knowledge into R2R so engineers can query it.

You have tools to:
- Download and index documentation (Confluence pages, local docs)
- Fetch and process Jira tickets to extract engineering lessons
- Discover the codebase module structure using your own analysis
- Import module definitions from a hand-crafted file
- Summarize every file in the codebase and index into R2R
- Search the knowledge base to verify results

Recommended workflow:
1. Index docs first (Confluence / local docs) — gives context for code summarization
2. Fetch + process tickets — extracts engineering lessons
3. discover_modules or import_modules — identify the module structure
4. summarize_codebase — expensive but most valuable; do last

Always tell the user what you're about to do before calling a tool.
After each tool, report the result concisely.
If a tool fails with a missing credential error, tell the user exactly which \
env vars to set and stop.
"""

# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(goal: str, model: str = "claude-opus-4-5", interactive: bool = False):
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": goal}]

    print(f"\n{'='*60}")
    print(f"Agent goal: {goal}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Collect text and tool calls from response
        messages.append({"role": "assistant", "content": response.content})

        # Print any text the agent says
        for block in response.content:
            if hasattr(block, "text"):
                print(f"\n[Agent] {block.text}")

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                print(f"\n[Tool] {block.name}({json.dumps(block.input, indent=2)})")
                result = _dispatch_tool(block.name, block.input)
                result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
                print(f"[Result] {result_str[:500]}{'...' if len(result_str) > 500 else ''}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        if interactive and response.stop_reason == "end_turn":
            try:
                user_input = input("\nYou: ").strip()
                if not user_input or user_input.lower() in ("exit", "quit", "q"):
                    break
                messages.append({"role": "user", "content": user_input})
            except (EOFError, KeyboardInterrupt):
                break

    print("\n[Agent] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrating agent for building the knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "goal", nargs="?", default=None,
        help="Goal for the agent (e.g. 'Index the knowledge base for /path/to/src')",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive (conversational) mode after the initial goal",
    )
    parser.add_argument(
        "--model", default="claude-opus-4-5",
        help="Claude model for the orchestrating agent (default: claude-opus-4-5)",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is required for the orchestrating agent.")
        sys.exit(1)

    if args.interactive and not args.goal:
        print("Knowledge Base Agent — type your goal, or 'quit' to exit.")
        try:
            goal = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
    elif args.goal:
        goal = args.goal
    else:
        parser.print_help()
        sys.exit(1)

    run_agent(goal, model=args.model, interactive=args.interactive)


if __name__ == "__main__":
    main()
