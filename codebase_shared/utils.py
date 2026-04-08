"""Shared utilities used across all agents.

Provides:
  - TokenTracker: accumulates prompt/completion token counts per phase
  - llm_call / allm_call: unified LLM call via litellm (sync and async)
  - RateLimitedExecutor: thread-based parallel executor (legacy)
  - AsyncRateLimiter: asyncio-based rate limiter for high-concurrency workloads
  - load_manifest / save_manifest: JSON file persistence for incremental mode
"""

import asyncio
import json
import re
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

class TokenTracker:
    """Accumulates prompt/completion token counts per phase.

    Thread-safe: uses a lock to protect concurrent updates from
    async tasks sharing the same tracker instance.
    """

    def __init__(self):
        import threading
        self.phases: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def _ensure_phase(self, phase: str) -> dict[str, int]:
        if phase not in self.phases:
            self.phases[phase] = {"prompt": 0, "completion": 0, "calls": 0}
        return self.phases[phase]

    def record(self, phase: str, response) -> None:
        """Extract usage from a litellm response object."""
        usage = getattr(response, "usage", None)
        with self._lock:
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
        with self._lock:
            p = self._ensure_phase(phase)
            if usage:
                p["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
                p["completion"] += getattr(usage, "completion_tokens", 0) or 0
            p["calls"] += 1

    def record_estimate(self, phase: str, prompt_chars: int,
                        completion_chars: int) -> None:
        """Fallback: estimate tokens from character count (÷4)."""
        with self._lock:
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


def _extract_content(response) -> str:
    """Safely extract text content from a litellm response."""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    return choices[0].message.content or ""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

# Models that require max_completion_tokens instead of max_tokens.
# litellm handles this for known models, but custom proxy endpoints may not.
_MAX_COMPLETION_TOKEN_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini"}


def _is_context_overflow(err: Exception) -> bool:
    """Return True if the error is a context-length overflow."""
    msg = str(err).lower()
    return any(phrase in msg for phrase in (
        "context length", "context_length", "maximum context",
        "input is too long", "input_tokens", "prompt is too long",
        "reduce the length", "tokens exceeds",
    ))


def _trim_messages_for_retry(messages: list[dict], fraction: float = 0.15) -> list[dict]:
    """Return a copy of messages with the largest user/human message trimmed by fraction."""
    import copy
    trimmed = copy.deepcopy(messages)
    # Find the longest user message and trim it
    longest_idx = max(
        (i for i, m in enumerate(trimmed) if m["role"] in ("user", "human")),
        key=lambda i: len(trimmed[i].get("content") or ""),
        default=None,
    )
    if longest_idx is not None:
        content = trimmed[longest_idx].get("content") or ""
        keep = int(len(content) * (1 - fraction))
        trimmed[longest_idx]["content"] = content[:keep] + "\n[...truncated to fit context window...]"
    return trimmed


def _apply_max_tokens(kwargs: dict, model: str, max_tokens: int) -> None:
    """Set the right max tokens parameter for the model.

    Some newer models (o1, o3) only accept max_completion_tokens.
    Most models accept max_tokens. We check the model name and set
    the appropriate parameter.
    """
    model_lower = model.lower().split("/")[-1]  # strip provider prefix
    if any(model_lower.startswith(m) for m in _MAX_COMPLETION_TOKEN_MODELS):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

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

    kwargs = dict(model=model, messages=full_messages)
    _apply_max_tokens(kwargs, model, max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for _attempt in range(3):
        try:
            response = completion(**kwargs)
            break
        except Exception as e:
            if _is_context_overflow(e) and _attempt < 2:
                print(f"  [warn] Context overflow on attempt {_attempt + 1}, "
                      f"trimming prompt by {15 * (_attempt + 1)}% and retrying...")
                kwargs["messages"] = _trim_messages_for_retry(
                    kwargs["messages"], fraction=0.15 * (_attempt + 1))
            else:
                raise
    else:
        raise RuntimeError("Context overflow: prompt could not be trimmed enough to fit.")

    if tracker and phase:
        tracker.record(phase, response)
    return _extract_content(response)


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

    kwargs = dict(model=model, messages=messages, stream=stream)
    _apply_max_tokens(kwargs, model, max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if stream:
        kwargs["stream_options"] = {"include_usage": True}

    for _attempt in range(3):
        try:
            response = completion(**kwargs)
            break
        except Exception as e:
            if _is_context_overflow(e) and _attempt < 2:
                print(f"  [warn] Context overflow on attempt {_attempt + 1}, "
                      f"trimming prompt by {15 * (_attempt + 1)}% and retrying...")
                kwargs["messages"] = _trim_messages_for_retry(
                    kwargs["messages"], fraction=0.15 * (_attempt + 1))
            else:
                raise
    else:
        raise RuntimeError("Context overflow: prompt could not be trimmed enough to fit.")

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
        return _extract_content(response)


async def allm_call(model: str, system: str, user: str,
                    max_tokens: int = 4096,
                    json_mode: bool = False,
                    tracker: Optional[TokenTracker] = None,
                    phase: str = ""):
    """
    Async LLM call via litellm.acompletion.

    Same interface as llm_call but awaitable. Use with AsyncRateLimiter
    for high-concurrency workloads.
    """
    from litellm import acompletion

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    kwargs = dict(model=model, messages=messages)
    _apply_max_tokens(kwargs, model, max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for _attempt in range(3):
        try:
            response = await acompletion(**kwargs)
            break
        except Exception as e:
            if _is_context_overflow(e) and _attempt < 2:
                print(f"  [warn] Context overflow on attempt {_attempt + 1}, "
                      f"trimming prompt by {15 * (_attempt + 1)}% and retrying...")
                kwargs["messages"] = _trim_messages_for_retry(
                    kwargs["messages"], fraction=0.15 * (_attempt + 1))
            else:
                raise
    else:
        raise RuntimeError("Context overflow: prompt could not be trimmed enough to fit.")

    if tracker and phase:
        tracker.record(phase, response)
    return _extract_content(response)


async def allm_call_multi(model: str, system: str, messages: list[dict],
                          max_tokens: int = 4096,
                          json_mode: bool = False,
                          tracker: Optional[TokenTracker] = None,
                          phase: str = ""):
    """
    Async multi-turn LLM call via litellm.acompletion.

    messages is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    The system prompt is prepended automatically.
    """
    from litellm import acompletion

    full_messages = [{"role": "system", "content": system}] + messages

    kwargs = dict(model=model, messages=full_messages)
    _apply_max_tokens(kwargs, model, max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await acompletion(**kwargs)

    if tracker and phase:
        tracker.record(phase, response)
    return _extract_content(response)


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
        kwargs = dict(model=model, messages=messages,
                      tools=tools, tool_choice="auto")
        _apply_max_tokens(kwargs, model, max_tokens)

        for _attempt in range(3):
            try:
                response = completion(**kwargs)
                break
            except Exception as e:
                if _is_context_overflow(e) and _attempt < 2:
                    print(f"  [warn] Context overflow in tool loop (attempt {_attempt + 1}), "
                          f"trimming prompt by {15 * (_attempt + 1)}% and retrying...")
                    kwargs["messages"] = _trim_messages_for_retry(
                        kwargs["messages"], fraction=0.15 * (_attempt + 1))
                else:
                    raise
        else:
            raise RuntimeError("Context overflow: prompt could not be trimmed enough to fit.")

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

            # Dispatch can signal loop termination by returning __ACCEPTED__
            if result == "__ACCEPTED__":
                return messages, args

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
# Async rate-limited concurrency
# ---------------------------------------------------------------------------

class QuotaExhaustedError(Exception):
    """Raised when LLM quota/budget is exhausted (not a transient rate limit)."""
    pass


def _is_quota_error(err_str: str) -> bool:
    """Detect hard quota exhaustion vs transient rate limits.

    Uses multi-word phrases to avoid false positives on generic words
    like 'exceeded' appearing in unrelated errors.
    """
    quota_signals = (
        "quota", "insufficient_quota", "budget exceeded",
        "budget limit", "billing", "spending limit",
        "allowance exceeded", "credits exhausted",
        "token limit exceeded", "exceeded your current quota",
        "exceeded your quota",
    )
    return any(s in err_str for s in quota_signals)


def _is_rate_limit_error(err_str: str) -> bool:
    """Detect transient rate-limit errors (retryable)."""
    return ("rate" in err_str or "429" in err_str
            or "too many" in err_str or "throttl" in err_str)


class AsyncRateLimiter:
    """Asyncio-based rate limiter with semaphore for concurrency control.

    Usage:
        limiter = AsyncRateLimiter(max_concurrent=50, calls_per_minute=500)
        tasks = [limiter.run(coro_fn(arg)) for arg in work_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # or use run_many() for callback-based processing

    Compared to RateLimitedExecutor (thread-based):
    - Supports 50-200+ concurrent requests (vs 4 threads)
    - No GIL contention on I/O
    - Token-bucket sleeps don't block other coroutines
    - Rate limit retries with exponential backoff

    Quota detection: when a hard quota/budget error is detected (not a
    transient 429), all pending tasks are cancelled immediately so progress
    can be saved.  Check limiter.quota_exhausted after run_many() returns.
    """

    def __init__(self, max_concurrent: int = 50,
                 calls_per_minute: int = 500,
                 max_retries: int = 3):

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._interval = 60.0 / calls_per_minute
        self._lock = asyncio.Lock()         # protects _last_call for rate limiting
        self._counter_lock = asyncio.Lock() # protects completed/skipped counts
        self._last_call = 0.0
        self._max_retries = max_retries
        self._halted = False
        self.quota_exhausted = False
        self.completed_count = 0
        self.skipped_count = 0

    async def _wait_for_slot(self):
        """Reserve a rate-limit slot, sleep if needed."""


        async with self._lock:
            now = _time.monotonic()
            wait = self._last_call + self._interval - now
            if wait < 0:
                wait = 0
            self._last_call = now + wait
        if wait > 0:
            await asyncio.sleep(wait)

    def halt(self, reason: str = ""):
        """Stop all pending work. Already-running tasks will finish."""
        self._halted = True
        if reason:
            print(f"\n⚠ Halting: {reason}")

    async def run(self, coro):
        """Run a single coroutine with rate limiting and concurrency control.

        No retry logic — a coroutine can only be awaited once.
        For retries, use run_many() with coroutine factories instead.
        """
        if self._halted:
            raise QuotaExhaustedError("Limiter halted")
        async with self._semaphore:
            if self._halted:
                raise QuotaExhaustedError("Limiter halted")
            await self._wait_for_slot()
            try:
                return await coro
            except Exception as e:
                err_str = str(e).lower()
                if _is_quota_error(err_str):
                    self.quota_exhausted = True
                    self.halt(f"Quota exhausted: {e}")
                    raise QuotaExhaustedError(str(e)) from e
                raise

    async def run_many(self, coro_factories: list,
                       on_result=None, on_error=None):
        """Run many coroutine factories concurrently with rate limiting.

        Each factory is a no-arg callable that returns an awaitable.
        This allows retry (a coroutine can only be awaited once).

        on_result(index, result) and on_error(index, exception) are called
        as tasks complete.

        When a quota/budget error is detected, all remaining tasks are
        cancelled and the method returns early.  Check self.quota_exhausted
        and self.completed_count / self.skipped_count afterward.

        Returns list of results (None for failed/skipped tasks).
        """


        results = [None] * len(coro_factories)

        async def _run_one(idx, factory):
            if self._halted:
                async with self._counter_lock:
                    self.skipped_count += 1
                return

            last_err = None
            for attempt in range(self._max_retries + 1):
                if self._halted:
                    async with self._counter_lock:
                        self.skipped_count += 1
                    return

                async with self._semaphore:
                    if self._halted:
                        async with self._counter_lock:
                            self.skipped_count += 1
                        return

                    await self._wait_for_slot()
                    try:
                        result = await factory()
                        results[idx] = result
                        async with self._counter_lock:
                            self.completed_count += 1
                        if on_result:
                            _cb_result = on_result(idx, result)
                            if asyncio.iscoroutine(_cb_result):
                                await _cb_result
                        return
                    except Exception as e:
                        err_str = str(e).lower()
                        if _is_quota_error(err_str):
                            self.quota_exhausted = True
                            self.halt(
                                f"Quota exhausted — {self.completed_count} "
                                f"completed, saving progress. "
                                f"Re-run to resume."
                            )
                            if on_error:
                                on_error(idx, e)
                            return
                        if _is_rate_limit_error(err_str):
                            last_err = e
                            backoff = min(60, 10 * (2 ** attempt))
                            await asyncio.sleep(backoff)
                        else:
                            if on_error:
                                on_error(idx, e)
                            return
            if on_error and last_err:
                on_error(idx, last_err)

        await asyncio.gather(*[_run_one(i, f) for i, f in enumerate(coro_factories)])

        if self.quota_exhausted:
            print(f"\n⚠ Quota exhausted after {self.completed_count} completions. "
                  f"{self.skipped_count} tasks skipped. "
                  f"Progress saved — re-run to resume.")

        return results


# ---------------------------------------------------------------------------
# Manifest (JSON file persistence for incremental mode)
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    """Load a JSON manifest from disk.  Returns {} on missing/corrupt file."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            print(f"[warn] Could not load manifest {path}: {e}")
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


# ---------------------------------------------------------------------------
# Smart RPM detection
# ---------------------------------------------------------------------------

def detect_rpm_from_proxy(model: str) -> Optional[int]:
    """Probe the LLM proxy for rate limit headers and return detected RPM.

    Makes a minimal LLM call and inspects response headers for standard
    rate limit info (x-ratelimit-limit-requests, x-ratelimit-limit, etc.).
    Returns None if detection fails or no headers found.
    """
    try:
        from litellm import completion
        response = completion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        # litellm stores raw response headers in _hidden_params
        hidden = getattr(response, "_hidden_params", {})
        headers = hidden.get("additional_headers", {}) or {}
        if not headers:
            # Some providers put headers elsewhere
            headers = hidden.get("headers", {}) or {}

        # Common header names for request rate limits
        for header_name in (
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit",
            "ratelimit-limit",
            "x-rate-limit-requests",
        ):
            val = headers.get(header_name)
            if val:
                try:
                    rpm = int(val)
                    if rpm > 0:
                        return rpm
                except (ValueError, TypeError):
                    continue
        return None
    except Exception:
        return None
