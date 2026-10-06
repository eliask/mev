"""Benchmark runner: call live LLM with different output formats and measure results.

For each (format, run) combination:
  - Batches gold sentences into calls of batch_size each
  - Sends them to the LLM using /v1/chat/completions with cache_prompt + no thinking
  - Parses the output with the format's parse_fn
  - Computes format compliance rate and per-dimension F1 vs gold labels
  - Records token counts and latency from the response

Key implementation choices:
  - use_cache=False: we want live LLM calls every time (timing/token data must be real)
  - cache_prompt=True: KV cache for repeated system prompts across batches
  - enable_thinking=False: suppress reasoning tokens for Qwen3 models
  - aiohttp directly: mirrors mev.llm pattern
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

from mev.bench.formats import FORMATS, OutputFormat
from mev.bench.gold import GoldItem
from mev.config import LLAMA_API_BASE, ROOT
from mev.llm import LLM_CACHE_PROMPT

BENCH_RESULTS_DIR = ROOT / ".tmp" / "bench_results"

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    format_name:             str
    run_number:              int
    batch_size:              int
    n_sentences:             int          # total sentences in this run
    n_calls:                 int          # number of LLM calls
    tokens_out:              int          # total completion tokens
    tokens_in:               int          # total prompt tokens
    prefill_ms:              float        # cumulative GPU prefill time
    decode_ms:               float        # cumulative GPU decode time
    elapsed_sec:             float        # wall-clock time for the whole run
    format_compliance_rate:  float        # fraction of outputs that parsed correctly
    f1_role:                 float        # macro F1 for Role dimension
    f1_quality:              float        # macro F1 for Quality dimension
    f1_topic:                float        # macro F1 for Topic dimension
    n_parsed:                int          # sentences successfully parsed
    n_correct_role:          int
    n_correct_quality:       int
    n_correct_topic:         int
    model_label:             str          = ""    # e.g. "qwen35-moe-iq4", set by CLI
    consistent_with_run1:    Optional[bool] = None  # set by caller
    raw_outputs:             list[str]    = field(default_factory=list)
    errors:                  list[str]    = field(default_factory=list)

    @property
    def tokens_per_sent(self) -> float:
        return self.tokens_out / max(self.n_sentences, 1)

    @property
    def decode_tok_per_sec(self) -> float:
        if self.decode_ms <= 0:
            return 0.0
        return self.tokens_out / (self.decode_ms / 1000)

    @property
    def prefill_tok_per_sec(self) -> float:
        if self.prefill_ms <= 0:
            return 0.0
        return self.tokens_in / (self.prefill_ms / 1000)

    @property
    def net_sents_per_sec(self) -> float:
        """Real throughput: sentences classified per wall-clock second.

        This is the key Pareto metric — it combines decode speed with format
        compression. A format that saves 30% tokens but only runs at 70% compliance
        will score lower here than a format that saves 10% but runs at 100%.
        """
        if self.elapsed_sec <= 0:
            return 0.0
        return self.n_sentences / self.elapsed_sec


# ---------------------------------------------------------------------------
# F1 computation (macro-averaged)
# ---------------------------------------------------------------------------

def _macro_f1(
    gold_labels: list[str],
    pred_labels: list[str],
    all_classes: set[str],
) -> float:
    """Compute macro-averaged F1 over provided classes.

    Missing predictions are treated as empty string (wrong).
    """
    if not gold_labels:
        return 0.0

    f1s = []
    for cls in all_classes:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s) if f1s else 0.0


# ---------------------------------------------------------------------------
# LLM call (no cache, full timing)
# ---------------------------------------------------------------------------

_TIMEOUT = aiohttp.ClientTimeout(total=300, sock_connect=5)

async def _llm_call(
    session: aiohttp.ClientSession,
    fmt: OutputFormat,
    batch_prompt: str,
    max_tokens: int,
    llm_url: str,
) -> dict:
    """Single LLM call. Returns dict with content, tokens_in, tokens_out,
    prefill_ms, decode_ms, finish_reason, error."""
    payload = {
        "messages": [
            {"role": "system", "content": fmt.system_prompt.strip()},
            {"role": "user",   "content": batch_prompt.strip()},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_prompt": LLM_CACHE_PROMPT,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.time()
    try:
        async with session.post(
            llm_url + "/v1/chat/completions",
            json=payload,
            timeout=_TIMEOUT,
        ) as resp:
            data = await resp.json(content_type=None)
            elapsed = time.time() - t0

            if "error" in data:
                return {"error": str(data["error"]), "elapsed": elapsed}

            choice  = data["choices"][0]
            content = choice["message"].get("content", "").strip()
            finish  = choice.get("finish_reason", "unknown")
            usage   = data.get("usage", {})
            timings = data.get("timings", {})

            return {
                "content":    content,
                "tokens_in":  usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "prefill_ms": timings.get("prompt_ms", 0.0),
                "decode_ms":  timings.get("predicted_ms", 0.0),
                "finish_reason": finish,
                "elapsed":    elapsed,
            }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "elapsed": time.time() - t0}


# ---------------------------------------------------------------------------
# Build user prompt for a batch of sentences
# ---------------------------------------------------------------------------

def _build_batch_prompt(items: list[GoldItem], batch_start: int = 0) -> tuple[str, dict[int, int]]:
    """Format gold items as the LLM input (numbered sentence list).

    Uses sequential 1-based numbers (batch_start+1, batch_start+2, ...) so
    that numbers are unique within each call even when gold items come from
    different atoms (where sent_idx resets to 0).

    Returns:
        prompt: the formatted input string
        num_to_pos: mapping from prompt line number → position in items list
    """
    lines = []
    num_to_pos: dict[int, int] = {}
    for pos, item in enumerate(items):
        num = batch_start + pos + 1   # 1-based, globally unique per run
        num_to_pos[num] = pos
        lines.append(f"[{num}] {item.text}")
    return '\n'.join(lines), num_to_pos


# ---------------------------------------------------------------------------
# Core benchmark: one format, one run
# ---------------------------------------------------------------------------

async def _run_one(
    fmt: OutputFormat,
    gold: list[GoldItem],
    run_number: int,
    batch_size: int,
    llm_url: str,
    verbose: bool = True,
    model_label: str = "",
) -> BenchResult:
    """Run one (format, run_number) pair. Returns BenchResult."""
    t_start = time.time()

    total_tokens_in  = 0
    total_tokens_out = 0
    total_prefill_ms = 0.0
    total_decode_ms  = 0.0
    n_calls          = 0
    n_parsed_ok      = 0   # sentences where LLM returned a parseable tag
    raw_outputs:     list[str] = []
    errors:          list[str] = []

    # Per-sentence prediction accumulation (keyed by position in gold list)
    gold_role:  list[str] = []
    gold_qual:  list[str] = []
    gold_topic: list[str] = []
    pred_role:  list[str] = []
    pred_qual:  list[str] = []
    pred_topic: list[str] = []

    # Estimate max_tokens: ~(tokens_per_sent * batch_size * 1.3) overhead
    max_tok = int(fmt.expected_tokens_per_sent * batch_size * 1.5) + 20

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(gold), batch_size):
            batch = gold[batch_start : batch_start + batch_size]
            prompt, num_to_pos = _build_batch_prompt(batch, batch_start)

            resp = await _llm_call(session, fmt, prompt, max_tok, llm_url)
            n_calls += 1

            if "error" in resp:
                err = resp["error"]
                errors.append(f"call {n_calls}: {err}")
                if verbose:
                    print(f"    [run {run_number}] call {n_calls} ERROR: {err}")
                # Mark all sentences in batch as missing
                for item in batch:
                    gold_role.append(item.role)
                    gold_qual.append(item.quality)
                    gold_topic.append(item.topic)
                    pred_role.append('')
                    pred_qual.append('')
                    pred_topic.append('')
                continue

            content = resp["content"]
            raw_outputs.append(content)
            total_tokens_in  += resp["tokens_in"]
            total_tokens_out += resp["tokens_out"]
            total_prefill_ms += resp["prefill_ms"]
            total_decode_ms  += resp["decode_ms"]

            # Parse: prompt line number → (role, qual, topic)
            parsed_by_num = fmt.parse_fn(content)

            # Build position → prediction mapping via num_to_pos
            parsed_by_pos: dict[int, tuple[str, str, str]] = {
                num_to_pos[num]: pred
                for num, pred in parsed_by_num.items()
                if num in num_to_pos
            }

            for pos, item in enumerate(batch):
                gold_role.append(item.role)
                gold_qual.append(item.quality)
                gold_topic.append(item.topic)

                pred = parsed_by_pos.get(pos)
                if pred is not None:
                    n_parsed_ok += 1
                    pred_role.append(pred[0])
                    pred_qual.append(pred[1])
                    pred_topic.append(pred[2])
                else:
                    pred_role.append('')
                    pred_qual.append('')
                    pred_topic.append('')

            if verbose and n_calls % 5 == 0:
                print(
                    f"    [run {run_number}] {n_calls} calls, "
                    f"{total_tokens_out} out-tok, "
                    f"{n_parsed_ok}/{batch_start + len(batch)} parsed",
                    flush=True,
                )

    elapsed = time.time() - t_start
    n_total = len(gold)

    # Compliance: fraction of gold sentences that got a parsed prediction
    compliance = n_parsed_ok / max(n_total, 1)

    # F1 per dimension (only over sentences that had predictions)
    all_roles   = set('PEVKL')
    all_quals   = set('GMAHT')
    all_topics  = set('FWSCINDYJRX')

    f1_r = _macro_f1(gold_role,  pred_role,  all_roles)
    f1_q = _macro_f1(gold_qual,  pred_qual,  all_quals)
    f1_t = _macro_f1(gold_topic, pred_topic, all_topics)

    n_correct_r = sum(1 for g, p in zip(gold_role,  pred_role)  if g == p and p)
    n_correct_q = sum(1 for g, p in zip(gold_qual,  pred_qual)  if g == p and p)
    n_correct_t = sum(1 for g, p in zip(gold_topic, pred_topic) if g == p and p)

    return BenchResult(
        format_name=fmt.name,
        run_number=run_number,
        batch_size=batch_size,
        n_sentences=n_total,
        n_calls=n_calls,
        tokens_out=total_tokens_out,
        tokens_in=total_tokens_in,
        prefill_ms=total_prefill_ms,
        decode_ms=total_decode_ms,
        elapsed_sec=elapsed,
        format_compliance_rate=compliance,
        f1_role=f1_r,
        f1_quality=f1_q,
        f1_topic=f1_t,
        n_parsed=n_parsed_ok,
        n_correct_role=n_correct_r,
        n_correct_quality=n_correct_q,
        n_correct_topic=n_correct_t,
        model_label=model_label,
        raw_outputs=raw_outputs,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

def _check_consistency(results: list[BenchResult]) -> list[BenchResult]:
    """Mark run 2/3 as consistent if their parsed outputs match run 1."""
    # Group by format name
    by_fmt: dict[str, list[BenchResult]] = defaultdict(list)
    for r in results:
        by_fmt[r.format_name].append(r)

    annotated = []
    for fmt_name, runs in by_fmt.items():
        runs_sorted = sorted(runs, key=lambda r: r.run_number)
        if not runs_sorted:
            continue
        run1 = runs_sorted[0]
        run1.consistent_with_run1 = True   # run 1 is by definition consistent with itself
        for run in runs_sorted[1:]:
            # Compare raw_outputs list as proxy (same content = consistent)
            run.consistent_with_run1 = (run.raw_outputs == run1.raw_outputs)
        annotated.extend(runs_sorted)

    return annotated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_benchmark(
    formats: list[OutputFormat],
    gold: list[GoldItem],
    batch_size: int = 10,
    runs: int = 3,
    llm_url: str = LLAMA_API_BASE,
    verbose: bool = True,
    model_label: str = "",
) -> list[BenchResult]:
    """Run benchmark for each (format, run) pair.

    Args:
        formats: List of OutputFormat objects to test.
        gold: Gold set (list of GoldItem).
        batch_size: Sentences per LLM call.
        runs: Number of repetitions per format (for consistency check).
        llm_url: Base URL of LLM server (no trailing slash, no path).
        verbose: Print progress.
        model_label: Optional label for the model being tested (e.g. "qwen35-moe-iq4").
            Stored in each BenchResult for cross-run comparison.

    Returns:
        List of BenchResult, one per (format, run).
    """
    all_results: list[BenchResult] = []

    for fmt in formats:
        if verbose:
            print(f"\n  Format: {fmt.name} — {fmt.description}")
        for run_num in range(1, runs + 1):
            if verbose:
                print(f"    Run {run_num}/{runs} ...", flush=True)
            result = await _run_one(
                fmt=fmt,
                gold=gold,
                run_number=run_num,
                batch_size=batch_size,
                llm_url=llm_url,
                verbose=verbose,
                model_label=model_label,
            )
            if verbose:
                print(
                    f"    -> {result.tokens_out} out-tok, "
                    f"comply={result.format_compliance_rate:.1%}, "
                    f"F1(R/Q/T)={result.f1_role:.2f}/{result.f1_quality:.2f}/{result.f1_topic:.2f}"
                )
            all_results.append(result)

    return _check_consistency(all_results)


def save_results(results: list[BenchResult], out_dir: Path = BENCH_RESULTS_DIR) -> Path:
    """Save results to JSON in bench_results directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"bench_{ts}.json"
    data = [asdict(r) for r in results]
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point (called from mev.cli)
# ---------------------------------------------------------------------------

def run_benchmark_cli(extra: list[str]) -> None:
    """Parse extra CLI args and run the benchmark."""
    import argparse

    ap = argparse.ArgumentParser(prog="mev detect bench-format")
    ap.add_argument(
        "--formats", default="baseline,compact,delta",
        help="Comma-separated format names (default: baseline,compact,delta)",
    )
    ap.add_argument(
        "--n-gold", type=int, default=200,
        help="Gold set size (default: 200; use --create-gold to regenerate)",
    )
    ap.add_argument(
        "--runs", type=int, default=3,
        help="Repetitions per format for consistency check (default: 3)",
    )
    ap.add_argument(
        "--batch-size", type=int, default=10,
        help="Sentences per LLM call (default: 10)",
    )
    ap.add_argument(
        "--llm-url", default=LLAMA_API_BASE,
        help=f"LLM server base URL (default: {LLAMA_API_BASE})",
    )
    ap.add_argument(
        "--create-gold", action="store_true",
        help="Regenerate gold set from enrichments DB even if it exists",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for gold set sampling (default: 42)",
    )
    ap.add_argument(
        "--save", action="store_true", default=True,
        help="Save results JSON to .tmp/bench_results/ (default: true)",
    )
    ap.add_argument(
        "--model-label", default="",
        help="Label for the model being tested (e.g. qwen35-moe-iq4). "
             "Stored in each BenchResult JSON for cross-model comparison.",
    )
    args = ap.parse_args(extra)

    # --- Gold set ---
    from mev.bench.gold import (
        create_gold_set, gold_set_exists, load_gold_set, save_gold_set, validate_gold_set,
    )

    if args.create_gold or not gold_set_exists():
        print(f"Creating gold set (n={args.n_gold}, seed={args.seed}) ...")
        gold = create_gold_set(n=args.n_gold, seed=args.seed)
        path = save_gold_set(gold)
        print(f"  Saved {len(gold)} items to {path}")
    else:
        gold = load_gold_set()
        # Trim or warn if size mismatch
        if len(gold) > args.n_gold:
            print(f"  Using first {args.n_gold} of {len(gold)} gold items")
            gold = gold[:args.n_gold]
        elif len(gold) < args.n_gold:
            print(
                f"  Warning: gold set has {len(gold)} items, "
                f"requested {args.n_gold}. Using what's available."
            )

    # --- Validate gold set diversity ---
    print("Validating gold set diversity ...")
    validate_gold_set(gold, warn=True)

    # --- Format selection ---
    fmt_names = [n.strip() for n in args.formats.split(',')]
    missing = [n for n in fmt_names if n not in FORMATS]
    if missing:
        print(f"ERROR: unknown format(s): {missing}")
        print(f"Available: {list(FORMATS.keys())}")
        return

    selected_formats = [FORMATS[n] for n in fmt_names]

    # --- LLM health check ---
    import urllib.request
    server_ok = False
    for path_suffix in ("/health", "/v1/models"):
        try:
            urllib.request.urlopen(args.llm_url + path_suffix, timeout=2)
            server_ok = True
            break
        except Exception:
            continue
    if not server_ok:
        print(f"WARNING: LLM server not reachable at {args.llm_url}")
        print("Proceeding anyway (will collect errors in results)")

    # --- Run ---
    print(f"\nRunning benchmark:")
    print(f"  Formats:     {fmt_names}")
    print(f"  Gold items:  {len(gold)}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Runs:        {args.runs}")
    print(f"  LLM URL:     {args.llm_url}")
    if args.model_label:
        print(f"  Model label: {args.model_label}")
    print()

    results = asyncio.run(
        run_benchmark(
            formats=selected_formats,
            gold=gold,
            batch_size=args.batch_size,
            runs=args.runs,
            llm_url=args.llm_url,
            verbose=True,
            model_label=args.model_label,
        )
    )

    # --- Report ---
    from mev.bench.report import print_report
    print_report(results)

    # --- Save ---
    if args.save:
        out = save_results(results)
        print(f"\nResults saved to {out}")
