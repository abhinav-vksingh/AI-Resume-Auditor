"""
evaluation/run_bias_audit.py
----------------------------
Run counterfactual bias audit across a dataset of resumes paired with
category-appropriate job descriptions.

Statistical Methodology:
  - For each resume i and identity pair j: compute delta = score_B - score_A.
  - Resume-level aggregation: observations from the same resume are not independent.
    We compute the mean delta per resume: mean_d_i = mean(delta_{i, :}).
  - Statistical testing: Paired t-test across resumes (sample size n = number of resumes):
    Testing H0: mean(mean_d_i) == 0.
  - Confidence interval: 95% Student's t confidence interval for the mean resume delta.
  - Full transparency: Per-pair comparison rows are also preserved and saved to CSV.

Usage:
    python evaluation/run_bias_audit.py
    python evaluation/run_bias_audit.py --resume-dir evaluation/sample_resumes --n-pairs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix import path for running from any cwd
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from backend.app.services.bias import build_variants, _score_variant, _score_variant_raw, DISCLAIMER


# ---------------------------------------------------------------------------
# Category & JD Helpers
# ---------------------------------------------------------------------------

_DATA_DIR = _REPO / "backend" / "data"
_SKILLS_PATH = _DATA_DIR / "skills.json"
_JD_DIR = _REPO / "evaluation" / "job_descriptions"


def load_category_jds(jd_dir: Path) -> Dict[str, str]:
    """Load all .txt job descriptions from the JD folder keyed by category name."""
    jds: Dict[str, str] = {}
    if not jd_dir.is_dir():
        return jds
    for f in jd_dir.glob("*.txt"):
        category = f.stem.lower()
        jds[category] = f.read_text(encoding="utf-8")
    return jds


def detect_resume_category(filename: str, text: str, available_categories: List[str]) -> str:
    """Detect the category for a resume using path/filename hints or skill vocabulary."""
    fname_lower = filename.lower().replace("-", "_").replace(" ", "_")

    # 1. Direct filename or path match
    for cat in available_categories:
        if cat in fname_lower:
            return cat

    # Common aliases in Kaggle datasets
    alias_map = {
        "hr": "human_resources",
        "human_resource": "human_resources",
        "data_science": "data_science",
        "datascience": "data_science",
        "data_scientist": "data_science",
        "web": "web_development",
        "web_developer": "web_development",
        "web_design": "web_development",
        "software": "software_engineering",
        "software_developer": "software_engineering",
        "developer": "software_engineering",
        "sales": "sales",
    }
    for alias, cat in alias_map.items():
        if alias in fname_lower and cat in available_categories:
            return cat

    # 2. Content-based skill matching fallback
    if _SKILLS_PATH.is_file():
        try:
            skills_data = json.loads(_SKILLS_PATH.read_text(encoding="utf-8"))
            text_lower = text.lower()
            scores: Dict[str, int] = {}
            for cat in available_categories:
                cat_skills = skills_data.get(cat, [])
                scores[cat] = sum(1 for s in cat_skills if s in text_lower)
            best_cat = max(scores, key=scores.get)
            if scores[best_cat] > 0:
                return best_cat
        except Exception:
            pass

    # Default fallback
    return available_categories[0] if available_categories else "software_engineering"


def load_resumes(folder: Path) -> List[Tuple[str, str]]:
    """Load text and PDF resumes from directory, returning list of (filename, text)."""
    resumes: List[Tuple[str, str]] = []
    if not folder.is_dir():
        return resumes

    for p in sorted(folder.glob("**/*")):
        if p.is_file():
            if p.suffix.lower() == ".txt":
                text = p.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    resumes.append((p.name, text))
            elif p.suffix.lower() == ".pdf":
                try:
                    import fitz
                    doc = fitz.open(str(p))
                    pages = [doc[i].get_text("text") for i in range(len(doc))]
                    doc.close()
                    text = "\n".join(pages)
                    if text.strip():
                        resumes.append((p.name, text))
                except Exception as exc:
                    print(f"  [WARN] Failed to read PDF {p.name}: {exc}")
    return resumes


# ---------------------------------------------------------------------------
# Core Audit Execution
# ---------------------------------------------------------------------------

def run_dataset_audit(
    resumes: List[Tuple[str, str]],
    category_jds: Dict[str, str],
    attribute: str,
    mode: str,
    n_pairs: int = 5,
) -> Dict[str, Any]:
    """
    Run audit across all resumes with statistical correction:
    1. Score each pair for each resume using the resume's category JD.
    2. Compute resume mean delta: mean_d_i = mean(deltas for resume i).
    3. Run paired t-test across the n resume mean deltas.
    4. Compute 95% confidence interval for the mean resume delta.
    """
    scorer = _score_variant_raw if mode == "raw" else _score_variant
    available_cats = list(category_jds.keys())

    pair_rows: List[Dict[str, Any]] = []
    resume_mean_deltas: List[float] = []
    resume_summaries: List[Dict[str, Any]] = []

    group_a_label = ""
    group_b_label = ""

    for res_idx, (fname, text) in enumerate(resumes):
        cat = detect_resume_category(fname, text, available_cats)
        jd_text = category_jds.get(cat, next(iter(category_jds.values())))

        variants, label_a, label_b = build_variants(text, attribute, n_pairs=n_pairs)
        group_a_label = label_a
        group_b_label = label_b

        res_deltas: List[float] = []

        for p_idx, v in enumerate(variants):
            score_a = scorer(v["text_a"], jd_text)
            score_b = scorer(v["text_b"], jd_text)
            delta = round(score_b - score_a, 4)
            res_deltas.append(delta)

            pair_rows.append({
                "resume_index": res_idx,
                "resume_file": fname,
                "category": cat,
                "attribute": attribute,
                "mode": mode,
                "pair_index": p_idx,
                "identity_a": v["identity_a"],
                "identity_b": v["identity_b"],
                "score_a": round(score_a, 2),
                "score_b": round(score_b, 2),
                "delta": delta,
            })

        mean_res_delta = float(np.mean(res_deltas)) if res_deltas else 0.0
        resume_mean_deltas.append(mean_res_delta)
        resume_summaries.append({
            "resume_index": res_idx,
            "resume_file": fname,
            "category": cat,
            "mean_delta": round(mean_res_delta, 4),
            "n_pairs": len(res_deltas),
        })

    # -- Statistical Analysis across resumes (n = len(resumes)) --
    n_resumes = len(resume_mean_deltas)
    overall_mean = float(np.mean(resume_mean_deltas)) if n_resumes > 0 else 0.0

    if n_resumes >= 2:
        std_dev = float(np.std(resume_mean_deltas, ddof=1))
        # Check for zero-variance case (e.g. all 0.0 in name-redacted mode)
        if std_dev < 1e-9:
            std_dev = 0.0
            sem = 0.0
            ci_lower = round(overall_mean, 4)
            ci_upper = round(overall_mean, 4)
            p_value = float("nan")
            significant = False
        else:
            sem = std_dev / math.sqrt(n_resumes)
            t_crit = float(stats.t.ppf(0.975, df=n_resumes - 1))
            ci_lower = round(overall_mean - t_crit * sem, 4)
            ci_upper = round(overall_mean + t_crit * sem, 4)

            # Paired t-test across resume means against 0 (H0: mean delta == 0)
            t_stat, p_val = stats.ttest_1samp(resume_mean_deltas, 0.0)
            p_value = round(float(p_val), 6)
            significant = bool(p_value < 0.05)
    else:
        std_dev = float("nan")
        ci_lower = float("nan")
        ci_upper = float("nan")
        p_value = float("nan")
        significant = False

    return {
        "attribute": attribute,
        "mode": mode,
        "group_a_label": group_a_label,
        "group_b_label": group_b_label,
        "n_resumes": n_resumes,
        "n_pairs_per_resume": n_pairs,
        "total_observations": len(pair_rows),
        "mean_delta": round(overall_mean, 4),
        "std_delta": round(std_dev, 4) if not math.isnan(std_dev) else float("nan"),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_value,
        "significant": significant,
        "resume_summaries": resume_summaries,
        "pair_rows": pair_rows,
    }


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run counterfactual bias audit across Kaggle resumes.")
    parser.add_argument("--resume-dir", default="evaluation/sample_resumes", help="Path to resume files")
    parser.add_argument("--jd-dir", default="evaluation/job_descriptions", help="Path to category JDs")
    parser.add_argument("--n-pairs", type=int, default=5, help="Number of identity pairs per resume")
    parser.add_argument("--out-dir", default="evaluation/results", help="Directory to save output files")
    args = parser.parse_args()

    resume_dir = Path(args.resume_dir)
    jd_dir = Path(args.jd_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Category JDs
    category_jds = load_category_jds(jd_dir)
    if not category_jds:
        print(f"[ERROR] No job descriptions found in {jd_dir}")
        sys.exit(1)
    print(f"[INFO] Loaded {len(category_jds)} category JDs from {jd_dir}: {list(category_jds.keys())}")

    # 2. Load Resumes
    resumes = load_resumes(resume_dir)
    if not resumes:
        print(f"[STATUS] Resume folder '{resume_dir}' is empty. No resumes to audit.")
        sys.exit(0)

    print(f"[INFO] Loaded {len(resumes)} resume(s) from {resume_dir}")

    # 3. Execute all 4 conditions
    attributes = ["name", "university"]
    modes = ["raw", "redacted"]

    summary_results: List[Dict[str, Any]] = []
    all_pair_rows: List[Dict[str, Any]] = []
    plot_data: Dict[Tuple[str, str], List[float]] = {}

    for attr in attributes:
        for mode in modes:
            print(f"\n[RUNNING] attribute={attr:<10} mode={mode:<10} resumes={len(resumes)} n_pairs={args.n_pairs}")
            res = run_dataset_audit(resumes, category_jds, attr, mode, n_pairs=args.n_pairs)
            summary_results.append(res)
            all_pair_rows.extend(res["pair_rows"])
            plot_data[(attr, mode)] = [s["mean_delta"] for s in res["resume_summaries"]]

            ci_str = f"[{res['ci_95_lower']}, {res['ci_95_upper']}]" if not math.isnan(res['ci_95_lower']) else "N/A"
            p_str = f"{res['p_value']:.6f}" if not math.isnan(res['p_value']) else "nan"
            print(f"  → mean_delta={res['mean_delta']:+.4f}  std={res['std_delta']:.4f}  "
                  f"95% CI={ci_str}  p={p_str}  sig={res['significant']}")

    # 4. Save Summary CSV
    summary_csv = out_dir / "bias_results.csv"
    summary_fields = [
        "attribute", "mode", "group_a_label", "group_b_label",
        "n_resumes", "n_pairs_per_resume", "total_observations",
        "mean_delta", "std_delta", "ci_95_lower", "ci_95_upper",
        "p_value", "significant"
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_results)
    print(f"\n[SAVED] Summary table → {summary_csv}")

    # 5. Save Per-Pair CSV
    pair_csv = out_dir / "bias_pair_rows.csv"
    pair_fields = [
        "resume_index", "resume_file", "category", "attribute", "mode",
        "pair_index", "identity_a", "identity_b", "score_a", "score_b", "delta"
    ]
    with pair_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(all_pair_rows)
    print(f"[SAVED] Detailed pairwise rows → {pair_csv}")

    # 6. Generate Publication Charts
    # Bar Chart: Mean Delta with 95% Confidence Interval Error Bars
    labels = [f"{a.capitalize()}\n({m})" for (a, m) in plot_data.keys()]
    means = [r["mean_delta"] for r in summary_results]
    yerr_lower = []
    yerr_upper = []
    for r in summary_results:
        if not math.isnan(r["ci_95_lower"]):
            yerr_lower.append(abs(r["mean_delta"] - r["ci_95_lower"]))
            yerr_upper.append(abs(r["ci_95_upper"] - r["mean_delta"]))
        else:
            yerr_lower.append(0.0)
            yerr_upper.append(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4A90E2" if m == "raw" else "#E94E77" for (_, m) in plot_data.keys()]
    bars = ax.bar(labels, means, yerr=[yerr_lower, yerr_upper], capsize=5, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Counterfactual Bias Audit – Mean Delta per Resume (with 95% CI)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Delta (score_B - score_A)", fontsize=10)
    ax.set_xlabel("Condition (Mode)", fontsize=10)
    for bar, val in zip(bars, means):
        offset = 0.08 if val >= 0 else -0.15
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:+.3f}", ha="center", fontsize=9, fontweight="bold")

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#4A90E2", label="raw (no PII stripping)"),
        mpatches.Patch(color="#E94E77", label="redacted (strip_pii applied)"),
    ], loc="upper left", fontsize=9)
    plt.tight_layout()
    bar_path = out_dir / "bias_bar.png"
    fig.savefig(str(bar_path), dpi=130)
    plt.close(fig)
    print(f"[SAVED] Bar chart with CI → {bar_path}")

    # Box Plot: Distribution of resume-level mean deltas
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    box_vals = list(plot_data.values())
    ax2.boxplot(box_vals, tick_labels=labels, patch_artist=True,
                boxprops=dict(facecolor="#ECEFF1", color="black"),
                medianprops=dict(color="#D32F2F", linewidth=2))
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_title("Distribution of Resume-Level Mean Deltas", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Resume Mean Delta (score points)", fontsize=10)
    plt.tight_layout()
    box_path = out_dir / "bias_box.png"
    fig2.savefig(str(box_path), dpi=130)
    plt.close(fig2)
    print(f"[SAVED] Box plot → {box_path}")

    # 7. Print Formatted Results Table
    print("\n" + "=" * 90)
    header = f"{'Attribute':<12} {'Mode':<10} {'Resumes':>8} {'Pairs/Res':>10} {'Mean Delta':>12} {'Std':>8} {'95% CI':>18} {'p-value':>10} {'Sig':>6}"
    print(header)
    print("-" * 90)
    for r in summary_results:
        ci_str = f"[{r['ci_95_lower']:+.2f}, {r['ci_95_upper']:+.2f}]" if not math.isnan(r['ci_95_lower']) else "N/A"
        p_str = f"{r['p_value']:.4f}" if not math.isnan(r['p_value']) else "N/A (0 var)"
        std_str = f"{r['std_delta']:.4f}" if not math.isnan(r['std_delta']) else "0.0000"
        sig_str = "YES" if r["significant"] else "NO"
        print(f"{r['attribute']:<12} {r['mode']:<10} {r['n_resumes']:>8} {r['n_pairs_per_resume']:>10} "
              f"{r['mean_delta']:>+12.4f} {std_str:>8} {ci_str:>18} {p_str:>10} {sig_str:>6}")
    print("=" * 90)


if __name__ == "__main__":
    main()
