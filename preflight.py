#!/usr/bin/env python3
"""Preflight check: validate all dependencies before running the pipeline.

Usage:
    python preflight.py
    python preflight.py --skip-llm   # skip LLM connectivity test
"""

import argparse
import importlib
import os
import sys
import urllib.request
from pathlib import Path


R2R_URL = os.getenv("R2R_URL", "http://localhost:7272")
CHECKS_PASSED = 0
CHECKS_FAILED = 0
CHECKS_WARNED = 0


def _ok(msg: str):
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  ✓ {msg}")


def _fail(msg: str, hint: str = ""):
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  ✗ {msg}")
    if hint:
        print(f"    → {hint}")


def _warn(msg: str, hint: str = ""):
    global CHECKS_WARNED
    CHECKS_WARNED += 1
    print(f"  ! {msg}")
    if hint:
        print(f"    → {hint}")


def check_python_version():
    """Python >= 3.10 required."""
    v = sys.version_info
    if v >= (3, 10):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python {v.major}.{v.minor}.{v.micro} (need >= 3.10)",
              "Install Python 3.10+ or use pyenv")


def check_dependencies():
    """All pip packages importable."""
    deps = {
        "r2r": "r2r[core]",
        "litellm": "litellm",
        "pydantic": "pydantic",
        "mcp": "mcp",
    }
    for module, pip_name in deps.items():
        try:
            importlib.import_module(module)
            _ok(f"{pip_name} installed")
        except ImportError:
            _fail(f"{pip_name} not installed",
                  f"pip install {pip_name}")

    # tree-sitter (optional but needed for study_agent)
    try:
        from tree_sitter_language_pack import get_parser
        _ok("tree-sitter-language-pack installed")
    except ImportError:
        _warn("tree-sitter-language-pack not installed",
              "pip install tree-sitter-language-pack (needed for study_agent)")


def check_r2r():
    """R2R vector database is reachable."""
    url = f"{R2R_URL}/v3/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                _ok(f"R2R healthy at {R2R_URL}")
            else:
                _fail(f"R2R returned status {resp.status}",
                      "docker compose -f r2r/compose.yaml up -d")
    except Exception as e:
        _fail(f"R2R not reachable at {R2R_URL}",
              "docker compose -f r2r/compose.yaml up -d")


def check_api_keys():
    """At least one LLM API key is set."""
    keys = {
        "OPENAI_API_KEY": "OpenAI / OpenAI-compatible",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GROQ_API_KEY": "Groq",
    }
    found = []
    for key, provider in keys.items():
        val = os.getenv(key, "")
        if val and not val.startswith("sk-your-"):
            found.append(provider)

    if found:
        _ok(f"API key(s) set: {', '.join(found)}")
    else:
        # Check if Ollama is available (no key needed)
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                _ok("Ollama detected (no API key needed)")
                return
        except Exception:
            pass
        _warn("No LLM API key found",
              "Set OPENAI_API_KEY (or ANTHROPIC_API_KEY, GROQ_API_KEY) in .env, or start Ollama")


def check_llm_connectivity():
    """Quick LLM call succeeds."""
    model = os.getenv("LLM_MODEL", "openai/gpt-4o")
    try:
        from litellm import completion
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": "Reply OK"}],
            max_tokens=5,
            timeout=10,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            _ok(f"LLM responds ({model})")
        else:
            _fail(f"LLM returned empty response ({model})")
    except ImportError:
        _warn("Cannot test LLM (litellm not installed)")
    except Exception as e:
        err = str(e)
        if "rate" in err.lower() or "429" in err.lower():
            _warn(f"LLM rate limited ({model})", "Try again in a moment")
        else:
            _fail(f"LLM call failed ({model}): {err[:100]}",
                  "Check API key, model string, and network")


def check_env_file():
    """.env file exists."""
    project_root = Path(__file__).resolve().parent
    env_file = project_root / ".env"
    if env_file.exists():
        _ok(".env file found")
    else:
        _warn(".env file not found",
              "cp .env.example .env && edit .env")


def check_r2r_config():
    """R2R config file exists."""
    project_root = Path(__file__).resolve().parent
    toml_file = project_root / "r2r" / "r2r.toml"
    if toml_file.exists():
        _ok("r2r/r2r.toml found")
    else:
        _fail("r2r/r2r.toml not found",
              "This file configures R2R embedding model and database")


def check_docker():
    """Docker is installed and running."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            _ok("Docker is running")
        else:
            _fail("Docker is installed but not running",
                  "Start Docker Desktop or run: sudo systemctl start docker")
    except FileNotFoundError:
        _fail("Docker not found",
              "Install Docker: https://docs.docker.com/get-docker/")
    except Exception as e:
        _warn(f"Could not check Docker: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Preflight check: validate all prerequisites"
    )
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM connectivity test (saves one API call)")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Preflight Check")
    print(f"{'='*50}\n")

    print("Environment:")
    check_python_version()
    check_env_file()
    check_api_keys()
    check_r2r_config()

    print("\nDependencies:")
    check_dependencies()

    print("\nServices:")
    check_docker()
    check_r2r()
    if not args.skip_llm:
        check_llm_connectivity()
    else:
        print("  - LLM test skipped (--skip-llm)")

    print(f"\n{'='*50}")
    print(f"  {CHECKS_PASSED} passed, {CHECKS_FAILED} failed, {CHECKS_WARNED} warnings")
    if CHECKS_FAILED == 0:
        print(f"  All clear — ready to run!")
    else:
        print(f"  Fix the failures above before running the pipeline.")
    print(f"{'='*50}\n")

    sys.exit(1 if CHECKS_FAILED > 0 else 0)


if __name__ == "__main__":
    main()
