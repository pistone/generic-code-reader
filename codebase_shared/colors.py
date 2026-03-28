"""Minimal ANSI color helpers for terminal output.

Auto-detects whether stdout is a TTY and disables colors when piped.
"""

import os
import sys

# Respect NO_COLOR convention (https://no-color.org/) and pipe detection
_COLORS_ENABLED = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM") != "dumb"
)

# ANSI codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _wrap(code: str, text: str) -> str:
    if not _COLORS_ENABLED:
        return text
    return f"{code}{text}{_RESET}"


def green(text: str) -> str:
    return _wrap(_GREEN, text)


def yellow(text: str) -> str:
    return _wrap(_YELLOW, text)


def red(text: str) -> str:
    return _wrap(_RED, text)


def cyan(text: str) -> str:
    return _wrap(_CYAN, text)


def bold(text: str) -> str:
    return _wrap(_BOLD, text)


def dim(text: str) -> str:
    return _wrap(_DIM, text)


def ok(text: str) -> str:
    """Green bold for success messages."""
    if not _COLORS_ENABLED:
        return text
    return f"{_GREEN}{_BOLD}{text}{_RESET}"


def warn(text: str) -> str:
    """Yellow bold for warnings."""
    if not _COLORS_ENABLED:
        return text
    return f"{_YELLOW}{_BOLD}{text}{_RESET}"


def err(text: str) -> str:
    """Red bold for errors."""
    if not _COLORS_ENABLED:
        return text
    return f"{_RED}{_BOLD}{text}{_RESET}"
