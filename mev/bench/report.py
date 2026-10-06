"""Report generation for LLM output format benchmark results.

Produces a comparison table suitable for console output and for
copying into the research notes.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from mev.bench.runner import BenchResult


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def print_report(results: list[BenchResult], reference_format: str = "baseline") -> None:
    """Print a comparison table to stdout.

    One row per format. If multiple runs exist for a format, shows
    the mean across runs and a consistency indicator (e.g. 3/3).

    Columns:
      Format | Tok/sent | vs base | Comply% | EffSav% | Sents/s |
      F1(R) | F1(Q) | F1(T) | Decode t/s | Consist.

    Format-relative savings are shown vs the reference_format (default: baseline).
    EffSav% = savings_pct * compliance_rate (accounts for compliance degradation).
    Sents/s = net sentences per wall-clock second (key Pareto metric).
    """
    if not results:
        print("No results to report.")
        return

    # Aggregate by format name
    by_fmt: dict[str, list[BenchResult]] = defaultdict(list)
    for r in results:
        by_fmt[r.format_name].append(r)

    # Determine reference values
    ref_f1_r = ref_f1_q = ref_f1_t = None
    ref_tok_per_sent = None
    ref_sents_per_sec = None
    if reference_format in by_fmt:
        ref_runs = by_fmt[reference_format]
        ref_f1_r = _mean(r.f1_role    for r in ref_runs)
        ref_f1_q = _mean(r.f1_quality for r in ref_runs)
        ref_f1_t = _mean(r.f1_topic   for r in ref_runs)
        ref_tok_per_sent  = _mean(r.tokens_per_sent  for r in ref_runs)
        ref_sents_per_sec = _mean(r.net_sents_per_sec for r in ref_runs)

    # Show model label if present
    labels = set(r.model_label for r in results if r.model_label)
    if labels:
        print(f"\nModel: {', '.join(sorted(labels))}")

    # Header
    sep = "-" * 130
    print()
    print(sep)
    print(
        f"{'Format':<16} | "
        f"{'Tok/sent':>8} | "
        f"{'vs base':>7} | "
        f"{'Comply%':>7} | "
        f"{'EffSav%':>7} | "
        f"{'Sents/s':>7} | "
        f"{'F1(R)':>6} | "
        f"{'F1(Q)':>6} | "
        f"{'F1(T)':>6} | "
        f"{'Decode t/s':>10} | "
        f"{'Consist.':>9}"
    )
    print(sep)

    # Order: reference format first, then others in insertion order
    fmt_order = [reference_format] + [
        n for n in by_fmt if n != reference_format
    ]

    for fmt_name in fmt_order:
        if fmt_name not in by_fmt:
            continue
        runs = sorted(by_fmt[fmt_name], key=lambda r: r.run_number)
        n_runs = len(runs)

        tok_per_sent    = _mean(r.tokens_per_sent          for r in runs)
        comply          = _mean(r.format_compliance_rate   for r in runs)
        f1_r            = _mean(r.f1_role                  for r in runs)
        f1_q            = _mean(r.f1_quality               for r in runs)
        f1_t            = _mean(r.f1_topic                 for r in runs)
        decode_tps      = _mean(r.decode_tok_per_sec       for r in runs)
        sents_per_sec   = _mean(r.net_sents_per_sec        for r in runs)

        # Consistency: how many runs produced the same raw output as run 1?
        n_consistent = sum(1 for r in runs if r.consistent_with_run1)
        consist_str  = f"{n_consistent}/{n_runs}"

        # Savings vs reference (positive = fewer tokens = better)
        if ref_tok_per_sent and tok_per_sent > 0 and fmt_name != reference_format:
            savings_pct = (1 - tok_per_sent / ref_tok_per_sent) * 100
            vs_base = f"{savings_pct:+.0f}%"
            # Effective savings = raw savings * compliance rate
            # (a format saving 60% but 50% compliant = 30% effective)
            eff_sav = savings_pct * comply
            eff_sav_str = f"{eff_sav:+.0f}%"
        elif fmt_name == reference_format:
            vs_base     = "ref"
            eff_sav_str = "ref"
        else:
            vs_base     = "n/a"
            eff_sav_str = "n/a"

        # F1 relative to reference (show delta, not absolute, for non-baseline)
        def _f1_str(val: float, ref: Optional[float]) -> str:
            if ref is None or fmt_name == reference_format:
                return "ref" if fmt_name == reference_format else f"{val:.2f}"
            delta = val - ref
            sign  = "+" if delta >= 0 else ""
            return f"{val:.2f}({sign}{delta:.2f})"

        f1_r_str = "ref" if fmt_name == reference_format else _f1_str(f1_r, ref_f1_r)
        f1_q_str = "ref" if fmt_name == reference_format else _f1_str(f1_q, ref_f1_q)
        f1_t_str = "ref" if fmt_name == reference_format else _f1_str(f1_t, ref_f1_t)

        sents_s_str = f"{sents_per_sec:.1f}" if sents_per_sec > 0 else "n/a"

        print(
            f"{fmt_name:<16} | "
            f"{tok_per_sent:>8.1f} | "
            f"{vs_base:>7} | "
            f"{comply:>7.1%} | "
            f"{eff_sav_str:>7} | "
            f"{sents_s_str:>7} | "
            f"{f1_r_str:>6} | "
            f"{f1_q_str:>6} | "
            f"{f1_t_str:>6} | "
            f"{decode_tps:>10,.0f} | "
            f"{consist_str:>9}"
        )

    print(sep)

    # Per-format run detail if multiple runs
    for fmt_name in fmt_order:
        if fmt_name not in by_fmt:
            continue
        runs = sorted(by_fmt[fmt_name], key=lambda r: r.run_number)
        if len(runs) <= 1:
            continue
        print(f"\n  {fmt_name} — per-run detail:")
        for r in runs:
            print(
                f"    Run {r.run_number}: "
                f"{r.tokens_out} out-tok ({r.tokens_per_sent:.1f}/sent), "
                f"comply={r.format_compliance_rate:.1%}, "
                f"sents/s={r.net_sents_per_sec:.1f}, "
                f"F1={r.f1_role:.2f}/{r.f1_quality:.2f}/{r.f1_topic:.2f}, "
                f"consistent={r.consistent_with_run1}"
            )
            if r.errors:
                for e in r.errors[:3]:
                    print(f"      ERROR: {e}")

    print()

    # Summary observation
    _print_observations(by_fmt, reference_format, ref_tok_per_sent, ref_sents_per_sec)


def _print_observations(
    by_fmt: dict[str, list[BenchResult]],
    reference_format: str,
    ref_tok_per_sent: Optional[float] = None,
    ref_sents_per_sec: Optional[float] = None,
) -> None:
    """Print a few high-level observations."""
    print("Key observations:")

    ref_runs = by_fmt.get(reference_format, [])
    ref_tps  = ref_tok_per_sent or (
        _mean(r.tokens_per_sent for r in ref_runs) if ref_runs else None
    )

    for fmt_name, runs in by_fmt.items():
        if fmt_name == reference_format:
            continue
        comply     = _mean(r.format_compliance_rate for r in runs)
        tps        = _mean(r.tokens_per_sent        for r in runs)
        sents_s    = _mean(r.net_sents_per_sec      for r in runs)
        n_cons     = sum(1 for r in runs if r.consistent_with_run1)
        n_runs     = len(runs)

        savings_str = ""
        if ref_tps and tps > 0:
            savings     = (1 - tps / ref_tps) * 100
            eff_savings = savings * comply
            savings_str = f", {savings:.0f}% tok savings, {eff_savings:.0f}% effective"

        throughput_str = ""
        if ref_sents_per_sec and sents_s > 0:
            throughput_str = f", {sents_s:.1f} sents/s"

        if comply >= 0.95:
            print(
                f"  {fmt_name}: reliable ({comply:.0%} comply, "
                f"{n_cons}/{n_runs} consistent{savings_str}{throughput_str})"
            )
        elif comply < 0.50:
            print(
                f"  {fmt_name}: UNRELIABLE — only {comply:.0%} compliance "
                f"(effective savings:{savings_str})"
            )
        else:
            print(
                f"  {fmt_name}: partial compliance {comply:.0%}, "
                f"{n_cons}/{n_runs} consistent{savings_str}{throughput_str}"
            )
    print()


# ---------------------------------------------------------------------------
# Load results from JSON (for post-hoc analysis)
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[BenchResult]:
    """Load BenchResult list from a saved JSON file."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [BenchResult(**d) for d in data]


def load_latest_results(bench_dir: Optional[Path] = None) -> list[BenchResult]:
    """Load the most recent bench results JSON from bench_results directory."""
    from mev.bench.runner import BENCH_RESULTS_DIR
    d = bench_dir or BENCH_RESULTS_DIR
    jsons = sorted(d.glob("bench_*.json"))
    if not jsons:
        raise FileNotFoundError(f"No bench results found in {d}")
    return load_results(jsons[-1])


def compare_runs(path1: Path, path2: Path) -> None:
    """Compare two bench result files (e.g., MoE vs dense model).

    Prints a side-by-side comparison table showing how each format's
    compliance, F1, and net throughput changed between the two runs.

    Typical use: run bench with MoE, swap server to 27B dense, run bench again,
    then call compare_runs(path_moe, path_dense) to see the Pareto shift.

    Args:
        path1: Path to first results JSON (shown as 'A' in the table).
        path2: Path to second results JSON (shown as 'B' in the table).
    """
    results_a = load_results(path1)
    results_b = load_results(path2)

    def _label(results: list[BenchResult], path: Path) -> str:
        labels = set(r.model_label for r in results if r.model_label)
        return ', '.join(sorted(labels)) if labels else path.stem

    label_a = _label(results_a, path1)
    label_b = _label(results_b, path2)

    def _agg(results: list[BenchResult]) -> dict[str, dict]:
        by_fmt: dict[str, list[BenchResult]] = defaultdict(list)
        for r in results:
            by_fmt[r.format_name].append(r)
        out = {}
        for fmt_name, runs in by_fmt.items():
            out[fmt_name] = {
                "comply":    _mean(r.format_compliance_rate for r in runs),
                "f1_role":   _mean(r.f1_role                for r in runs),
                "f1_qual":   _mean(r.f1_quality             for r in runs),
                "f1_topic":  _mean(r.f1_topic               for r in runs),
                "tok_sent":  _mean(r.tokens_per_sent        for r in runs),
                "sents_s":   _mean(r.net_sents_per_sec      for r in runs),
            }
        return out

    agg_a = _agg(results_a)
    agg_b = _agg(results_b)

    all_formats = sorted(set(agg_a) | set(agg_b))

    sep = "-" * 110
    print(f"\nComparison: A={label_a}  vs  B={label_b}")
    print(f"  {path1}")
    print(f"  {path2}")
    print()
    print(sep)
    print(
        f"{'Format':<16} | "
        f"{'Comply A':>8} {'Comply B':>8} {'Δcomply':>8} | "
        f"{'F1(R) A':>7} {'F1(R) B':>7} | "
        f"{'Tok/s A':>7} {'Tok/s B':>7} | "
        f"{'Snt/s A':>7} {'Snt/s B':>7}"
    )
    print(sep)

    for fmt_name in all_formats:
        a = agg_a.get(fmt_name)
        b = agg_b.get(fmt_name)

        def _fmt_val(d: Optional[dict], key: str, fmt_str: str) -> str:
            if d is None:
                return "n/a".rjust(8)
            return format(d[key], fmt_str).rjust(8)

        comply_a = a["comply"] if a else None
        comply_b = b["comply"] if b else None
        delta_comply = ""
        if comply_a is not None and comply_b is not None:
            d = comply_b - comply_a
            delta_comply = f"{d:+.1%}".rjust(8)
        else:
            delta_comply = "n/a".rjust(8)

        print(
            f"{fmt_name:<16} | "
            f"{_fmt_val(a, 'comply', '.1%')} "
            f"{_fmt_val(b, 'comply', '.1%')} "
            f"{delta_comply} | "
            f"{_fmt_val(a, 'f1_role', '.2f')} "
            f"{_fmt_val(b, 'f1_role', '.2f')} | "
            f"{_fmt_val(a, 'tok_sent', '.1f')} "
            f"{_fmt_val(b, 'tok_sent', '.1f')} | "
            f"{_fmt_val(a, 'sents_s', '.1f')} "
            f"{_fmt_val(b, 'sents_s', '.1f')}"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(vals) -> float:
    lst = list(vals)
    return sum(lst) / len(lst) if lst else 0.0
