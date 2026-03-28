#!/usr/bin/env python3
"""One-command orchestrator for generic-code-reader.

Usage:
    python run.py --codebase /path/to/src
    python run.py --codebase /path/to/src --docs /path/to/docs
    python run.py --codebase /path/to/src --incremental

This runs the full pipeline: detect language, estimate cost, analyze code,
index into R2R. For advanced options, use the individual agents directly.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from codebase_shared.colors import green, red, bold, dim, ok, err
except ImportError:
    green = red = bold = dim = ok = err = lambda x: x  # type: ignore


def _check_env():
    """Quick check that environment is configured."""
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"]
    if not any(os.getenv(k) for k in keys):
        # Check for Ollama
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return  # Ollama available
        except Exception:
            pass
        print("No LLM API key found.")
        print("  Set OPENAI_API_KEY (or another provider key) in .env, then run: source .env")
        print("  Or start Ollama for fully local operation.")
        sys.exit(1)


def _check_r2r():
    """Check R2R is healthy, with retry."""
    import time
    import urllib.request
    r2r_url = os.getenv("R2R_URL", "http://localhost:7272")
    for attempt in range(6):  # 30 seconds max
        try:
            req = urllib.request.Request(f"{r2r_url}/v3/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        if attempt == 0:
            print(f"Waiting for R2R at {r2r_url}...", end="", flush=True)
        print(".", end="", flush=True)
        time.sleep(5)
    print(" failed.")
    print(f"  R2R not reachable at {r2r_url}")
    print(f"  Start it: docker compose -f r2r/compose.yaml up -d")
    sys.exit(1)


def _run(cmd: list[str], description: str) -> int:
    """Run a subprocess, streaming output."""
    print(f"\n{green('='*60)}")
    print(f"  {bold(description)}")
    print(f"{green('='*60)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n{err('[Error]')} {description} failed (exit code {result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="One-command setup: analyze a codebase and make it searchable via Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run.py --codebase /path/to/src
  python run.py --codebase /path/to/src --docs /path/to/docs
  python run.py --codebase /path/to/src --incremental
  python run.py --codebase /path/to/src --model bedrock/anthropic.claude-sonnet
  python run.py --codebase /path/to/src --dry-run
""")
    parser.add_argument("--codebase", required=True,
                        help="Root directory of the codebase to analyze")
    parser.add_argument("--docs", nargs="+", default=None,
                        help="Design docs/directories to index (optional)")
    parser.add_argument("--tickets", default=None,
                        help="Exported ticket JSON directory (optional)")
    parser.add_argument("--model", default="openai/gpt-4o",
                        help="LLM model (default: openai/gpt-4o)")
    parser.add_argument("--model-fast", default=None,
                        help="Cheaper model for bulk summarization")
    parser.add_argument("--language", default=None,
                        help="Override language auto-detection")
    parser.add_argument("--passes", type=int, default=1,
                        help="Summarize+review iterations (default: 1)")
    parser.add_argument("--incremental", action="store_true",
                        help="Only process changed files/docs")
    parser.add_argument("--rpm", type=int, default=60,
                        help="Rate limit: LLM calls/minute (default: 60)")
    parser.add_argument("--max-concurrent", type=int, default=50,
                        help="Max async requests (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost without running")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompts")
    args = parser.parse_args()

    codebase = Path(args.codebase).resolve()
    if not codebase.is_dir():
        print(f"Error: '{codebase}' is not a directory")
        sys.exit(1)

    project_dir = Path(__file__).resolve().parent
    python = sys.executable

    print(f"\ngeneric-code-reader")
    print(f"Codebase: {codebase}")
    if args.docs:
        print(f"Docs:     {', '.join(args.docs)}")
    print()

    # Pre-flight checks
    _check_env()
    _check_r2r()
    print()

    # Step 1: Index docs (if provided)
    if args.docs:
        doc_cmd = [python, "-m", "doc_agent.doc_agent",
                   "--docs"] + args.docs + ["--model", args.model]
        if args.incremental:
            doc_cmd.append("--incremental")
        rc = _run(doc_cmd, "Step 1: Indexing documentation")
        if rc != 0 and not args.incremental:
            print("Doc indexing failed. Continuing with code analysis...")

    # Step 2: Study codebase
    study_cmd = [python, str(project_dir / "indexer" / "study_agent.py"),
                 "--codebase", str(codebase),
                 "--model", args.model,
                 "--passes", str(args.passes),
                 "--rpm", str(args.rpm),
                 "--max-concurrent", str(args.max_concurrent)]
    if args.language:
        study_cmd += ["--language", args.language]
    if args.model_fast:
        study_cmd += ["--model-fast", args.model_fast]
    if args.incremental:
        study_cmd.append("--incremental")
    if args.dry_run:
        study_cmd.append("--dry-run")
    if args.yes:
        study_cmd.append("--yes")
    if args.docs:
        study_cmd += ["--docs"] + args.docs
        study_cmd.append("--rag")

    step_label = "Step 2: Analyzing codebase" if args.docs else "Analyzing codebase"
    rc = _run(study_cmd, step_label)
    if rc != 0:
        sys.exit(rc)
    if args.dry_run:
        return

    # Step 3: Index tickets (if provided)
    if args.tickets:
        ticket_cmd = [python, "-m", "ticket_agent.ticket_agent",
                      "--tickets", args.tickets, "--model", args.model]
        if args.incremental:
            ticket_cmd.append("--incremental")
        _run(ticket_cmd, "Step 3: Extracting ticket knowledge")

    # Done
    print(f"\n{green('='*60)}")
    print(f"  {ok('Ready!')}")
    print(f"{green('='*60)}")
    print(f"\n  Open Claude Code in: {bold(str(project_dir))}")
    print(f"  The search_codebase tool is available.")
    print(f"\n  Try: {dim('\"How does X work?\"')} or {dim('\"Where is Y implemented?\"')}")
    print()


if __name__ == "__main__":
    main()
