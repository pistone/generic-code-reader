"""Shared utilities used across all agents.

Provides:
  - TokenTracker: accumulates prompt/completion token counts per phase
  - llm_call: unified LLM call via litellm (supports streaming, JSON mode)
  - load_manifest / save_manifest: JSON file persistence for incremental mode
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

class TokenTracker:
    """Accumulates prompt/completion token counts per phase."""

    def __init__(self):
        self.phases: dict[str, dict[str, int]] = {}

    def _ensure_phase(self, phase: str) -> dict[str, int]:
        if phase not in self.phases:
            self.phases[phase] = {"prompt": 0, "completion": 0, "calls": 0}
        return self.phases[phase]

    def record(self, phase: str, response) -> None:
        """Extract usage from a litellm response object."""
        usage = getattr(response, "usage", None)
        p = self._ensure_phase(phase)
        if usage:
            p["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
            p["completion"] += getattr(usage, "completion_tokens", 0) or 0
        p["calls"] += 1

    def record_streaming(self, phase: str, chunks: list) -> None:
        """Extract usage from the last chunk of a streaming response."""
        if not chunks:
            return
        last = chunks[-1]
        usage = getattr(last, "usage", None)
        if not usage:
            choices = getattr(last, "choices", [])
            if choices:
                usage = getattr(choices[0], "usage", None)
        p = self._ensure_phase(phase)
        if usage:
            p["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
            p["completion"] += getattr(usage, "completion_tokens", 0) or 0
        p["calls"] += 1

    def record_estimate(self, phase: str, prompt_chars: int,
                        completion_chars: int) -> None:
        """Fallback: estimate tokens from character count (÷4)."""
        p = self._ensure_phase(phase)
        p["prompt"] += prompt_chars // 4
        p["completion"] += completion_chars // 4
        p["calls"] += 1

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = []
        total_p, total_c = 0, 0
        for phase, counts in self.phases.items():
            p, c, n = counts["prompt"], counts["completion"], counts["calls"]
            total_p += p
            total_c += c
            lines.append(
                f"[Tokens] {phase + ':':<12} {p:>8,} prompt + {c:>7,} completion"
                f"   ({n} call{'s' if n != 1 else ''})"
            )
        lines.append(
            f"[Tokens] {'Total:':<12} {total_p:>8,} prompt + {total_c:>7,} completion"
            f" = {total_p + total_c:,} tokens"
        )
        return "\n".join(lines)

    def to_log_entry(self, model: str, agent: str = "",
                     codebase: str = "") -> dict:
        """Return a dict suitable for JSONL logging."""
        total = sum(p["prompt"] + p["completion"]
                    for p in self.phases.values())
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "phases": dict(self.phases),
            "total_tokens": total,
        }
        if agent:
            entry["agent"] = agent
        if codebase:
            entry["codebase"] = codebase
        return entry


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def llm_call(model: str, system: str, user: str,
             max_tokens: int = 4096,
             json_mode: bool = False,
             stream: bool = False,
             tracker: Optional[TokenTracker] = None,
             phase: str = ""):
    """
    Unified LLM call via litellm.

    Returns the full response text, or a generator of text chunks if
    stream=True.  If tracker and phase are provided, records token usage.
    """
    from litellm import completion

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens,
                  stream=stream)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if stream:
        kwargs["stream_options"] = {"include_usage": True}

    response = completion(**kwargs)

    if stream:
        def _gen():
            collected_chunks = []
            for chunk in response:
                collected_chunks.append(chunk)
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            if tracker and phase:
                tracker.record_streaming(phase, collected_chunks)
        return _gen()
    else:
        if tracker and phase:
            tracker.record(phase, response)
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Manifest (JSON file persistence for incremental mode)
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    """Load a JSON manifest from disk.  Returns {} on missing/corrupt file."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    """Write a JSON manifest to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def load_queue(path: Path) -> list:
    """Load a JSON queue (list) from disk.  Returns [] on missing/corrupt file."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return []


def save_queue(path: Path, queue: list) -> None:
    """Write a JSON queue (list) to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, indent=2))
