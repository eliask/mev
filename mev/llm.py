"""Shared LLM infrastructure for mekanismirealismi scripts.

Pattern mirrors LawVM/src/lawvm/finland/grafter.py:call_llm().
Uses raw aiohttp (not litellm) + diskcache for persistent cross-run caching.

Usage:
    import asyncio
    import aiohttp
    from llm import call_llm, LLMContextExhausted, CACHE_VERSION

    async def main():
        async with aiohttp.ClientSession() as session:
            result = await call_llm(session, "system prompt", "user prompt", ctx="myctx")

    # Batch parallel calls:
    from llm import batch_llm
    results = asyncio.run(batch_llm(items, lambda s, item: call_llm(s, SYS, item['text']),
                                    parallel=16, desc="Tagging"))
"""


import asyncio
import hashlib
import json
import os
import time
import weakref
from pathlib import Path
from typing import Any

import aiohttp
from diskcache import Cache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLAMA_BASE = os.environ.get("LLAMA_API_BASE", "http://localhost:8080").rstrip("/").removesuffix("/v1")
LLAMA_URL = LLAMA_BASE + "/v1/chat/completions"
_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = (_ROOT if (_ROOT / "pyproject.toml").is_file() else Path.cwd()) / "data" / "mev_llm_cache"
CACHE_VERSION = "v2-source-model-bound"
CACHE_TTL = 90 * 24 * 3600   # 90 days
# Per-request timeout for localhost LLM (sock_connect fast, total covers generation time)
LLM_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=180, sock_connect=5)
# cache_prompt=True tells llama.cpp to keep the KV state cached for prefix reuse.
# Useful for full-attention models with a shared system prompt prefix.
# HARMFUL for SWA/hybrid models (Gemma 3, etc.): cache entries eat VRAM/RAM but SWA
# invalidates them on prefix mismatch anyway → full recompute + memory bloat.
# Set LLM_CACHE_PROMPT=1 to enable (only if you know the model benefits).
LLM_CACHE_PROMPT: bool = os.environ.get("LLM_CACHE_PROMPT", "0") == "1"

_cache = Cache(str(CACHE_DIR))


# ---------------------------------------------------------------------------
# Token throughput tracker (module singleton)
# ---------------------------------------------------------------------------

class TokenTracker:
    """Tracks aggregate LLM token throughput across all calls.

    Prints periodic summaries to stderr.  Safe for single-threaded asyncio.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.requests = 0
        self.cache_hits = 0
        self.errors = 0
        self.prefill_ms = 0.0       # actual GPU prefill time
        self.decode_ms = 0.0        # actual GPU decode time
        self.start_time: float | None = None
        self._monitor_task: asyncio.Task | None = None
        self._prev_in = 0
        self._prev_out = 0
        self._prev_time: float | None = None

    def record(self, tokens_in: int = 0, tokens_out: int = 0,
               cached: bool = False, error: bool = False,
               timings: dict | None = None) -> None:
        if self.start_time is None:
            self.start_time = time.time()
        self.requests += 1
        if cached:
            self.cache_hits += 1
        elif error:
            self.errors += 1
        else:
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            if timings:
                self.prefill_ms += timings.get("prompt_ms", 0)
                self.decode_ms += timings.get("predicted_ms", 0)

    def summary_line(self) -> str:
        if not self.start_time:
            return ""
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        ts = f"{hrs}:{mins:02d}:{secs:02d}" if hrs else f"{mins}:{secs:02d}"

        # Rates since start
        if elapsed > 0:
            rate_in = self.tokens_in / (elapsed / 60)
            rate_out = self.tokens_out / (elapsed / 60)
        else:
            rate_in = rate_out = 0

        # Interval rates (since last print)
        now = time.time()
        if self._prev_time and now > self._prev_time:
            dt = (now - self._prev_time) / 60
            int_in = (self.tokens_in - self._prev_in) / dt
            int_out = (self.tokens_out - self._prev_out) / dt
            interval = f" | last {int_in:,.0f}/{int_out:,.0f}"
        else:
            interval = ""

        total = self.tokens_in + self.tokens_out
        live = self.requests - self.cache_hits

        parts = [
            f"[LLM {ts}]",
            f"{_fmt_tok(self.tokens_in)} in / {_fmt_tok(self.tokens_out)} out",
            f"({_fmt_tok(total)} total)",
            f"| avg {rate_in:,.0f}/{rate_out:,.0f} tok/min{interval}",
            f"| {self.requests} req ({self.cache_hits} cached, {live} live)",
        ]
        if self.errors:
            parts.append(f"| {self.errors} errors")

        # Real GPU throughput from llama.cpp timings
        if self.prefill_ms > 0 or self.decode_ms > 0:
            gpu_s = (self.prefill_ms + self.decode_ms) / 1000
            gpu_pct = (gpu_s / elapsed * 100) if elapsed > 0 else 0
            real_prefill = (self.tokens_in / (self.prefill_ms / 1000)) if self.prefill_ms > 0 else 0
            real_decode = (self.tokens_out / (self.decode_ms / 1000)) if self.decode_ms > 0 else 0
            parts.append(
                f"| GPU: {real_prefill:,.0f} prefill/{real_decode:,.0f} decode tok/s"
                f" ({gpu_pct:.0f}% util)"
            )

        return "  ".join(parts)

    def _snapshot(self) -> None:
        """Save current counters for interval rate calculation."""
        self._prev_in = self.tokens_in
        self._prev_out = self.tokens_out
        self._prev_time = time.time()

    async def _monitor_loop(self, interval: float) -> None:
        """Background task: print throughput every `interval` seconds."""
        try:
            await asyncio.sleep(interval)  # first print after one interval
            while True:
                line = self.summary_line()
                if line:
                    print(f"\n  {line}", flush=True)
                self._snapshot()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def start_monitor(self, interval: float = 60) -> None:
        """Start background throughput printer. Call from within an event loop."""
        if self._monitor_task is not None:
            return
        self._snapshot()
        self._monitor_task = asyncio.ensure_future(self._monitor_loop(interval))

    def stop_monitor(self) -> str:
        """Stop background printer, return final summary line."""
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None
        return self.summary_line()


def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 1_000:
        return f"{n/1e3:.1f}K"
    return str(n)


TRACKER = TokenTracker()


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class LLMContextExhausted(Exception):
    """finish_reason='length' — output was truncated. Caller must reduce context."""


class LLMRequestFailed(RuntimeError):
    """Processing failed; this is never evidence of an empty source result."""


_session_models: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


async def _model_identity(session: aiohttp.ClientSession) -> dict:
    """Bind a session to the advertised model, template and serving build.

    Identity is discovered once per session; callers must start a new session
    after changing the server model. A missing API weight digest remains an
    explicit limitation, not an invented attestation of the loaded weights.
    """
    if session in _session_models:
        return _session_models[session]
    async with session.get(LLAMA_BASE + "/v1/models", timeout=LLM_REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        models = (await response.json(content_type=None)).get("data") or []
    if not models:
        raise LLMRequestFailed("Local server advertises no model; cache identity is unavailable")
    async with session.get(LLAMA_BASE + "/props", timeout=LLM_REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        props = await response.json(content_type=None)
    async with session.post(LLAMA_URL, json={
        "model": models[0]["id"], "messages": [{"role": "user", "content": "Reply with a period."}],
        "max_tokens": 1, "temperature": 0, "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=LLM_REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        runtime = await response.json(content_type=None)
    identity = {
        "endpoint": LLAMA_BASE, "model_id": models[0]["id"],
        "model_meta": models[0].get("meta") or {}, "weight_digest": models[0].get("digest"),
        "model_path": props.get("model_path"), "quantization": props.get("model_ftype"),
        "template_sha256": hashlib.sha256(str(props.get("chat_template")).encode()).hexdigest(),
        "server_build": runtime.get("system_fingerprint"),
        "loaded_weight_bytes_attested": bool(models[0].get("digest")),
    }
    _session_models[session] = identity
    return identity


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _cache_key(system: str, user: str, max_tokens: int, model_identity: dict) -> str:
    payload = json.dumps(
        {"v": CACHE_VERSION, "s": system, "u": user, "mt": max_tokens,
         "model": model_identity, "temperature": 0, "seed": 42,
         "cache_prompt": LLM_CACHE_PROMPT, "enable_thinking": False},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

async def call_llm(
    session: aiohttp.ClientSession,
    system: str,
    user: str,
    max_tokens: int = 500,
    ctx: str = "",
    use_cache: bool = True,
) -> str:
    """Call llama-server with caching.

    Returns a response string. Raises LLMRequestFailed after exhausted retries.
    Raises LLMContextExhausted if finish_reason='length'.
    """
    model = await _model_identity(session)
    key = _cache_key(system, user, max_tokens, model)

    if use_cache and key in _cache:
        TRACKER.record(cached=True)
        return _cache[key]  # type: ignore[return-value]

    payload: dict[str, Any] = {
        "model": model["model_id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": LLM_CACHE_PROMPT,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.post(LLAMA_URL, json=payload, timeout=LLM_REQUEST_TIMEOUT) as resp:
                data = await resp.json(content_type=None)
                if "choices" not in data:
                    err = data.get("error", {})
                    if isinstance(err, dict) and err.get("type") == "exceed_context_size_error":
                        msg = (f"[{ctx}] input too long: {err.get('n_prompt_tokens')} tokens, "
                               f"ctx={err.get('n_ctx')} — skipping (no retry)")
                        print(f"  LLM overflow {msg}")
                        TRACKER.record(error=True)
                        raise LLMContextExhausted(msg)
                    raise ValueError(f"no 'choices' in response (status {resp.status}): {str(data)[:200]}")
                choice = data["choices"][0]
                content = choice["message"].get("content", "").strip()
                finish = choice.get("finish_reason", "unknown")
                usage = data.get("usage", {})

                if finish == "length":
                    raise LLMContextExhausted(
                        f"[{ctx}] output truncated at max_tokens={max_tokens}"
                    )

                TRACKER.record(
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    timings=data.get("timings"),
                )
                if content and use_cache:
                    _cache.set(key, content, expire=CACHE_TTL)
                return content

        except LLMContextExhausted:
            raise
        except Exception as e:
            err_str = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                print(f"  LLM error [{ctx}]: {err_str} — retry {attempt+1}/{MAX_RETRIES} in {delay}s")
                await asyncio.sleep(delay)
            else:
                TRACKER.record(error=True)
                print(f"  LLM error [{ctx}]: {err_str} — giving up")
                raise LLMRequestFailed(f"[{ctx}] inference failed after retries: {err_str}") from e
    raise LLMRequestFailed(f"[{ctx}] inference failed after retries")


# ---------------------------------------------------------------------------
# Full-response variant (for callers that need token counts / timing)
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 15]  # seconds


async def call_llm_full(
    session: aiohttp.ClientSession,
    system: str,
    user: str,
    max_tokens: int = 500,
    ctx: str = "",
    use_cache: bool = True,
) -> dict:
    """Like call_llm but returns full response dict with tokens/timing.

    Returns: {content, tokens_in, tokens_out, elapsed, finish_reason}

    Cache hit: returns {content, tokens_in: 0, tokens_out: 0, elapsed: 0,
                        finish_reason: 'cached'}.
    Cache miss: makes live LLM call with retry on transient errors.
    """
    model = await _model_identity(session)
    key = _cache_key(system, user, max_tokens, model)

    if use_cache and key in _cache:
        TRACKER.record(cached=True)
        return {
            "content": _cache[key],
            "tokens_in": 0,
            "tokens_out": 0,
            "elapsed": 0.0,
            "finish_reason": "cached",
        }

    payload: dict[str, Any] = {
        "model": model["model_id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": LLM_CACHE_PROMPT,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error = None
    t0 = time.time()
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.post(LLAMA_URL, json=payload, timeout=LLM_REQUEST_TIMEOUT) as resp:
                data = await resp.json(content_type=None)
                if "choices" not in data:
                    err = data.get("error", {})
                    if isinstance(err, dict) and err.get("type") == "exceed_context_size_error":
                        elapsed = time.time() - t0
                        msg = (f"input too long: {err.get('n_prompt_tokens')} tokens, "
                               f"ctx={err.get('n_ctx')}")
                        if ctx:
                            print(f"  LLM overflow [{ctx}]: {msg} — skipping (no retry)")
                        TRACKER.record(error=True)
                        return {"content": "", "tokens_in": 0, "tokens_out": 0,
                                "elapsed": elapsed, "finish_reason": "context_overflow",
                                "error": msg}
                    raise ValueError(f"no 'choices' in response (status {resp.status}): {str(data)[:200]}")
                choice = data["choices"][0]
                content = choice["message"].get("content", "").strip()
                finish = choice.get("finish_reason", "unknown")
                usage = data.get("usage", {})
                elapsed = time.time() - t0

                tok_in = usage.get("prompt_tokens", 0)
                tok_out = usage.get("completion_tokens", 0)

                if finish == "length":
                    print(f"  LLM TRUNCATED [{ctx}]: output hit max_tokens={max_tokens}, {len(content)} chars returned")

                TRACKER.record(tokens_in=tok_in, tokens_out=tok_out,
                              timings=data.get("timings"))

                # Don't cache truncated output
                if content and use_cache and finish != "length":
                    _cache.set(key, content, expire=CACHE_TTL)

                return {
                    "content": content,
                    "tokens_in": tok_in,
                    "tokens_out": tok_out,
                    "elapsed": elapsed,
                    "finish_reason": finish,
                }

        except (aiohttp.ClientError, TimeoutError, KeyError, IndexError, ValueError) as e:
            last_error = e
            err_str = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                print(f"  LLM error [{ctx}]: {err_str} — retry {attempt+1}/{MAX_RETRIES} in {delay}s")
                await asyncio.sleep(delay)
            else:
                elapsed = time.time() - t0
                TRACKER.record(error=True)
                print(f"  LLM error [{ctx}]: {err_str} — giving up after {MAX_RETRIES} retries")
                return {
                    "content": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "elapsed": elapsed,
                    "finish_reason": "error",
                    "error": str(last_error),
                }
    # unreachable but satisfies type checker
    return {"content": "", "finish_reason": "error", "error": "max retries"}


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

async def batch_llm(
    items: list,
    call_fn,          # async fn(session, item) -> anything
    parallel: int = 16,
    desc: str = "LLM batch",
) -> list:
    """Run call_fn over items with bounded parallelism.

    call_fn signature: async (session: aiohttp.ClientSession, item) -> result

    Returns results in the same order as items.
    """
    sem = asyncio.Semaphore(parallel)
    hits = misses = errors = 0

    async def _run(session: aiohttp.ClientSession, item, idx: int):
        nonlocal hits, misses, errors
        async with sem:
            try:
                result = await call_fn(session, item)
                return idx, result
            except Exception as e:  # noqa: BLE001 — isolate and report arbitrary caller failures per item
                errors += 1
                print(f"  [{desc}] item {idx} error: {e}")
                return idx, None

    async with aiohttp.ClientSession() as session:
        tasks = [_run(session, item, i) for i, item in enumerate(items)]
        indexed = await asyncio.gather(*tasks)

    results = [None] * len(items)
    for idx, result in indexed:
        results[idx] = result

    print(f"  {desc}: {len(items)} items, {errors} errors")
    return results
