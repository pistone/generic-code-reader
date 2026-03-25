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

def llm_call_multi(model: str, system: str, messages: list[dict],
                    max_tokens: int = 4096,
                    json_mode: bool = False,
                    tracker: Optional[TokenTracker] = None,
                    phase: str = ""):
    """
    Multi-turn LLM call via litellm.

    messages is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    The system prompt is prepended automatically.
    Returns the assistant's response text.
    """
    from litellm import completion

    full_messages = [{"role": "system", "content": system}] + messages

    kwargs = dict(model=model, messages=full_messages, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = completion(**kwargs)

    if tracker and phase:
        tracker.record(phase, response)
    return response.choices[0].message.content or ""


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


def llm_tool_loop(
    model: str,
    system: str,
    initial_messages: list[dict],
    tools: list[dict],
    dispatch: "Callable[[str, dict], str]",
    terminal_tools: Optional[set[str]] = None,
    max_rounds: int = 10,
    max_tokens: int = 4096,
    tracker: Optional["TokenTracker"] = None,
    phase: str = "",
    on_round: Optional["Callable[[int, str, dict], None]"] = None,
) -> tuple[list[dict], Optional[dict]]:
    """
    Generic tool-use agent loop via litellm.

    The LLM can call any tool in `tools`. The `dispatch` callback handles
    each tool call: dispatch(tool_name, args_dict) -> result_string.

    When the LLM calls a tool listed in `terminal_tools`, the loop ends
    immediately and returns (conversation, tool_args).  The dispatch
    function is NOT called for terminal tools.

    If the LLM responds without tool calls (text-only) or max_rounds is
    exhausted, returns (conversation, None).

    on_round(round_num, tool_name, args) is an optional callback for logging.
    """
    from litellm import completion

    terminal_tools = terminal_tools or set()
    messages = [{"role": "system", "content": system}] + list(initial_messages)

    for round_num in range(1, max_rounds + 1):
        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens,
                      tools=tools, tool_choice="auto")
        response = completion(**kwargs)

        if tracker and phase:
            tracker.record(f"{phase} (round {round_num})", response)

        msg = response.choices[0].message

        # Append assistant message (preserving tool_calls metadata)
        messages.append(msg.model_dump())

        # No tool calls → LLM is done talking (shouldn't happen often with tools)
        if not msg.tool_calls:
            return messages, None

        # Process each tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            if on_round:
                on_round(round_num, name, args)

            # Terminal tool → return immediately
            if name in terminal_tools:
                return messages, args

            # Dispatch and feed result back
            result = dispatch(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    # Max rounds exhausted
    return messages, None


# ---------------------------------------------------------------------------
# Rate-limited parallel executor
# ---------------------------------------------------------------------------

class RateLimitedExecutor:
    """ThreadPoolExecutor with token-bucket rate limiting and retry on rate errors.

    Usage:
        executor = RateLimitedExecutor(workers=4, calls_per_minute=60)
        futures = [executor.submit(fn, arg) for arg in work_items]
        for future in executor.as_completed(futures):
            result = future.result()  # or handle exceptions
        executor.shutdown()

    Rate limit errors (HTTP 429 / "rate" in error message) are retried
    with exponential backoff up to max_retries times.
    """

    def __init__(self, workers: int = 4, calls_per_minute: int = 60,
                 max_retries: int = 3):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._interval = 60.0 / calls_per_minute  # seconds between calls
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._max_retries = max_retries

    def _wait_for_slot(self):
        """Block until the next rate-limit slot is available."""
        import time as _time
        with self._lock:
            now = _time.monotonic()
            wait = self._last_call + self._interval - now
            if wait < 0:
                wait = 0
            # Reserve the slot now, sleep outside the lock
            self._last_call = now + wait
        if wait > 0:
            _time.sleep(wait)

    def submit(self, fn, *args, **kwargs):
        """Submit a callable with rate limiting and retry."""
        from concurrent.futures import Future

        def _wrapped():
            last_err = None
            for attempt in range(self._max_retries + 1):
                self._wait_for_slot()
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    err_str = str(e).lower()
                    if "rate" in err_str or "429" in err_str or "too many" in err_str:
                        last_err = e
                        import time as _time
                        backoff = min(60, 10 * (2 ** attempt))
                        _time.sleep(backoff)
                    else:
                        raise
            raise last_err  # type: ignore[misc]

        return self._pool.submit(_wrapped)

    def as_completed(self, futures):
        """Yield futures as they complete."""
        from concurrent.futures import as_completed
        yield from as_completed(futures)

    def shutdown(self, wait: bool = True):
        self._pool.shutdown(wait=wait)


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
