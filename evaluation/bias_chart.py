"""
evaluation/bias_chart.py – Bar chart + box plot for bias audit results.

USAGE
-----
  # Option 1: generate charts from saved audit JSON
  python -m evaluation.bias_chart --input audit_result.json

  # Option 2: run with synthetic demo data (no real resume needed)
  python -m evaluation.bias_chart --demo

Both modes save:
  evaluation/bias_bar.png  – mean delta per attribute with 95% CI error bars
  evaluation/bias_box.png  – distribution of individual deltas

HOW TO INTERPRET THE CHARTS
----------------------------
Bar chart:
  X-axis  = attribute ("name", "university")
  Y-axis  = mean(delta) = mean(score_B - score_A)
  Error bar = ± 1.96 × std / sqrt(n)   (approximate 95% CI)
  A bar near zero means the pipeline does not favour group B over group A.

Box plot:
  Each box shows the spread of individual deltas.
  A tight box around 0 means consistent non-bias.
  Outliers (dots beyond whiskers) deserve manual inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _bar_chart(results: List[Dict[str, Any]], out_path: Path, demo: bool = False) -> None:
    """Save a grouped bar chart of mean deltas with 95% CI error bars."""
    attributes  = [r["attribute"]   for r in results]
    mean_deltas = [r["mean_delta"]  for r in results]
    std_deltas  = [r.get("std_delta", 0.0) for r in results]
    n_obs       = [r["n_observations"] for r in results]

    # 95% CI half-width: 1.96 * std / sqrt(n)
    ci_errors = [
        1.96 * std / (n ** 0.5) if n > 1 else 0.0
        for std, n in zip(std_deltas, n_obs)
    ]

    colours = ["#4C72B0" if d >= 0 else "#DD8452" for d in mean_deltas]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        attributes, mean_deltas,
        color=colours, edgecolor="black", linewidth=0.8,
        yerr=ci_errors, capsize=6, error_kw={"elinewidth": 1.5},
        zorder=3,
    )

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=2)
    ax.set_xlabel("Attribute swapped", fontsize=12)
    ax.set_ylabel("Mean Δ score  (score_B − score_A)", fontsize=12)
    # When running in --demo mode the title includes a clear warning so the chart
    # cannot be mistaken for a result computed on real resumes.
    bar_title = "Bias Audit — Mean Score Delta by Attribute"
    if demo:
        bar_title += "\n[DEMO – NOT REAL DATA]"
    bar_title += "\n(error bars: 95 % CI;  Δ ≈ 0 means no measurable bias)"
    ax.set_title(bar_title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=1)

    # Annotate p-values
    for i, (bar, r) in enumerate(zip(bars, results)):
        p = r.get("p_value", float("nan"))
        p_label = f"p={p:.3f}" if not (isinstance(p, float) and p != p) else "p=n/a"
        sig_star = "*" if r.get("significant", False) else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci_errors[i] + 0.3,
            f"{p_label}{sig_star}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved → {out_path}")


def _box_plot(results: List[Dict[str, Any]], out_path: Path, demo: bool = False) -> None:
    """Save a box plot showing the distribution of individual deltas."""
    # Build a flat dataframe-like structure for seaborn
    all_deltas:     List[float] = []
    all_attributes: List[str]   = []

    for r in results:
        deltas = [row["delta"] for row in r.get("raw_rows", [])]
        all_deltas.extend(deltas)
        all_attributes.extend([r["attribute"]] * len(deltas))

    if not all_deltas:
        print("No delta data to plot (box plot skipped).")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        x=all_attributes, y=all_deltas, hue=all_attributes,
        palette="Set2", width=0.4, linewidth=1.2,
        flierprops={"marker": "o", "markersize": 4},
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        x=all_attributes, y=all_deltas,
        color="black", alpha=0.4, size=4, jitter=True, ax=ax,
    )

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Attribute swapped", fontsize=12)
    ax.set_ylabel("Δ score  (score_B − score_A)", fontsize=12)
    # Same demo warning as the bar chart so both saved images are clearly labelled.
    box_title = "Bias Audit — Delta Distribution by Attribute"
    if demo:
        box_title += "\n[DEMO – NOT REAL DATA]"
    box_title += "\n(each dot is one identity-pair comparison; box shows IQR)"
    ax.set_title(box_title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Box plot saved  → {out_path}")


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def _demo_results() -> List[Dict[str, Any]]:
    """Return synthetic audit results for chart demonstration.

    Name audit: delta ≈ 0 — expected (name-blind pipeline).
    University audit: some positive delta — well-known universities may carry
    higher semantic weight in certain JDs.
    """
    rng = np.random.default_rng(42)

    # Name pairs: tiny Gaussian noise around 0 (pipeline strips names)
    name_deltas = rng.normal(loc=0.02, scale=0.15, size=10).tolist()
    name_rows = [
        {"resume_index": 0, "identity_a": f"NameA_{i}", "identity_b": f"NameB_{i}",
         "score_a": 65.0, "score_b": round(65.0 + d, 2), "delta": round(d, 4)}
        for i, d in enumerate(name_deltas)
    ]

    # University pairs: slight positive delta for well-known (group A)
    # (score_B is lesser-known, so delta = score_B - score_A < 0 on average)
    uni_deltas = rng.normal(loc=-3.5, scale=2.0, size=10).tolist()
    uni_rows = [
        {"resume_index": 0, "identity_a": f"WellKnownUni_{i}", "identity_b": f"LesserKnownUni_{i}",
         "score_a": 68.0, "score_b": round(68.0 + d, 2), "delta": round(d, 4)}
        for i, d in enumerate(uni_deltas)
    ]

    name_arr = np.array(name_deltas)
    uni_arr  = np.array(uni_deltas)

    from scipy import stats as sp_stats

    _, name_p = sp_stats.ttest_1samp(name_arr, 0)
    _, uni_p  = sp_stats.ttest_1samp(uni_arr,  0)

    return [
        {
            "attribute":     "name",
            "group_a_label": "white_american",
            "group_b_label": "black_american",
            "mean_delta":    round(float(name_arr.mean()), 4),
            "std_delta":     round(float(name_arr.std(ddof=1)), 4),
            "p_value":       round(float(name_p), 6),
            "significant":   bool(name_p < 0.05),
            "n_observations": len(name_rows),
            "raw_rows":      name_rows,
        },
        {
            "attribute":     "university",
            "group_a_label": "well_known",
            "group_b_label": "lesser_known",
            "mean_delta":    round(float(uni_arr.mean()), 4),
            "std_delta":     round(float(uni_arr.std(ddof=1)), 4),
            "p_value":       round(float(uni_p), 6),
            "significant":   bool(uni_p < 0.05),
            "n_observations": len(uni_rows),
            "raw_rows":      uni_rows,
        },
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bias audit charts.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo",  action="store_true", help="Use synthetic demo data.")
    group.add_argument("--input", type=str,            help="Path to audit result JSON file.")
    args = parser.parse_args()

    if args.demo:
        results = _demo_results()
        # Prominent banner — impossible to miss — so demo output is never
        # confused with results computed on real resumes.
        _BANNER = "=" * 60
        print(_BANNER)
        print("  DEMO DATA — NOT REAL AUDIT RESULTS")
        print("  Charts will be labelled [DEMO – NOT REAL DATA].")
        print(_BANNER)
    else:
        path = Path(args.input)
        if not path.exists():
            sys.exit(f"File not found: {path}")
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        # Accept either a single result dict or a list
        results = payload if isinstance(payload, list) else [payload]

    _bar_chart(results, _OUT_DIR / "bias_bar.png", demo=args.demo)
    _box_plot( results, _OUT_DIR / "bias_box.png", demo=args.demo)
    print("\nDone. Charts saved to evaluation/")

    if args.demo:
        _BANNER = "=" * 60
        print(_BANNER)
        print("  DEMO DATA — NOT REAL AUDIT RESULTS")
        print(_BANNER)


if __name__ == "__main__":
    main()
