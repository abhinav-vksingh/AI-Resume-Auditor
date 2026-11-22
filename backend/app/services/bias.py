"""
services/bias.py – Counterfactual bias audit for the resume scoring pipeline.

DISCLAIMER
----------
This is a small project-level audit conducted on synthetic resume variants.
The results do NOT constitute proof of fairness or unfairness in any real-world
hiring context. For meaningful bias assessment, a large and representative
sample of real resumes is required, together with domain-expert review and
statistical power analysis. The audit is intended to demonstrate methodology,
not to certify the system as unbiased.

METHODOLOGY (Bertrand & Mullainathan style)
--------------------------------------------
Principle: compare identity X vs identity Y on the SAME resume content.
Only the attribute under test (name or university) changes; every other
character is byte-identical between the two variants.

Steps:
  1. Take the ORIGINAL resume text (pre-preprocessing, from /extract).
  2. Build variant_A: swap the attribute with an identity-A value.
  3. Build variant_B: swap the attribute with an identity-B value.
  4. Run each variant through the scoring pipeline independently.
  5. Record delta = score_B - score_A.
  6. Repeat for N pairs; report mean delta, std dev, and paired t-test.

TWO AUDIT MODES
---------------
raw      : Score WITHOUT strip_pii.  The embedding model sees the injected
           name or university.  This reveals whether the model's semantic
           representations associate identity signals with score differences.

redacted : Score through the normal production pipeline (with strip_pii).
           Names are removed before encoding, so this proves name-blindness
           by design.  University names survive PII stripping and may still
           influence scores.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy import stats

from backend.app.services.preprocessing import (
    clean_text,
    detect_sections,
    normalize_aliases,
    preprocess,
)
from backend.app.services.scoring import compute_final_score
from backend.app.services.semantic import compute_semantic_similarity
from backend.app.services.skills import run_skill_analysis

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

_DATA = Path(__file__).resolve().parent.parent.parent / "data"
_NAMES_PATH = _DATA / "names.json"
_UNIS_PATH  = _DATA / "universities.json"

# ---------------------------------------------------------------------------
# Disclaimer — also returned in the API response
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "DISCLAIMER: This is a small project-level audit conducted on synthetic "
    "resume variants. The results do NOT constitute proof of fairness or "
    "unfairness in any real-world hiring context. A meaningful bias assessment "
    "requires a large, representative sample of real resumes plus domain-expert "
    "review. This audit is for educational demonstration only."
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_names() -> dict:
    with _NAMES_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    # Drop internal metadata keys that start with '_'
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _load_universities() -> dict:
    with _UNIS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Attribute swap helpers
# ---------------------------------------------------------------------------

def _swap_name(original_text: str, new_name: str) -> str:
    """Replace the first non-empty line of *original_text* with *new_name*.

    The first non-empty line of a plain-text resume is assumed to be the
    candidate's name.  Everything else stays byte-identical.

    If the resume has no non-empty lines (pathological case), the new name
    is prepended.
    """
    lines = original_text.split("\n")
    for i, line in enumerate(lines):
        if line.strip():
            lines[i] = new_name  # swap — length may change, content identical elsewhere
            return "\n".join(lines)
    # Fallback: no non-empty line found
    return new_name + "\n" + original_text


def _swap_university(original_text: str, new_university: str) -> str:
    """Replace an institution name in *original_text* with *new_university*.

    Strategy:
      1. Scan for any university from our JSON list (case-insensitive).
         If found, replace the first occurrence only.
      2. If none from JSON list is found, inspect lines for an existing college,
         university, institute, or academy name and replace it in-place within that line,
         guaranteeing exactly ONE institution exists in the text in the identical location.
      3. If no educational institution keyword is found on any line, insert
         'University: <new_university>' beneath the education header or at line 2.
    """
    unis = _load_universities()
    all_unis: List[str] = []
    for group_unis in unis.values():
        all_unis.extend(group_unis)

    # 1. Try known universities from JSON first (longest first)
    for uni in sorted(all_unis, key=len, reverse=True):
        pattern = re.compile(re.escape(uni), re.IGNORECASE)
        if pattern.search(original_text):
            return pattern.sub(new_university, original_text, count=1)

    # 2. Line-by-line search for an existing institution to replace in place
    lines = original_text.split("\n")
    inst_line_re = re.compile(
        r'\b([A-Za-z0-9&.\'-]+(?:\s+[A-Za-z0-9&.\'-]+){0,6}\s+(?:University|College|Institute|Academy)(?:\s+(?:of|for|&|and)\s+[A-Za-z0-9&.\'-]+(?:\s+[A-Za-z0-9&.\'-]+){0,4})?)',
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):
        m = inst_line_re.search(line)
        if m:
            # Replace exactly the matched institution span within that line
            lines[i] = line[:m.start(1)] + new_university + line[m.end(1):]
            return "\n".join(lines)

    # 3. If no institution found, look for Education section header to insert directly under it
    for i, line in enumerate(lines):
        if re.search(r'\b(?:education|qualification|academics)\b', line, re.IGNORECASE):
            lines.insert(i + 1, f"University: {new_university}")
            return "\n".join(lines)

    # 4. Fallback: insert on line 2
    if len(lines) > 1:
        lines.insert(1, f"University: {new_university}")
        return "\n".join(lines)

    return original_text.rstrip("\n") + f"\nUniversity: {new_university}\n"


# ---------------------------------------------------------------------------
# Pipeline runners (single variant) — two modes
# ---------------------------------------------------------------------------

def _score_variant(text: str, job_description: str) -> float:
    """REDACTED mode: run the full production pipeline (with strip_pii).

    Pipeline:
        original text → preprocess (incl. strip_pii) → semantic + skill
                      → score fusion → final_score (float, 0–100)

    Name-swap deltas will be ≈ 0 because strip_pii removes the name before
    encoding.  This is intentional and proves name-blindness by design.
    """
    preprocessed  = preprocess(text)
    core_text     = preprocessed["core_text"]
    sections      = preprocessed["sections"]

    semantic_score = compute_semantic_similarity(core_text, job_description, sections)
    skill_result   = run_skill_analysis(job_description, core_text, sections)
    score_result   = compute_final_score(semantic_score, skill_result["skill_overlap"])

    return score_result["final_score"]


def _score_variant_raw(text: str, job_description: str) -> float:
    """RAW mode: score WITHOUT strip_pii so the embedding model sees the name/university.

    Pipeline:
        original text → clean_text → detect_sections → normalize_aliases
                      → (NO strip_pii) → semantic + skill → score fusion

    This measures whether the embedding model's representations associate
    identity signals (names, universities) with score differences.
    """
    cleaned   = clean_text(text)
    sections  = detect_sections(cleaned)
    core_text = normalize_aliases(cleaned)

    semantic_score = compute_semantic_similarity(core_text, job_description, sections)
    skill_result   = run_skill_analysis(job_description, core_text, sections)
    score_result   = compute_final_score(semantic_score, skill_result["skill_overlap"])

    return score_result["final_score"]


# ---------------------------------------------------------------------------
# Public API — build_variants
# ---------------------------------------------------------------------------

def build_variants(
    original_text: str,
    attribute: Literal["name", "university"],
    n_pairs: int = 5,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Build N pairs of resume text variants differing only in one attribute.

    Parameters
    ----------
    original_text : str
        Raw text from extraction — BEFORE preprocessing (so name/university
        is still present and can be swapped).
    attribute : "name" | "university"
        Which attribute to swap between pairs.
    n_pairs : int
        How many identity pairs to generate (capped by available data).

    Returns
    -------
    (variants, group_a_label, group_b_label)
        variants : List[Dict[str, Any]] with keys
            "identity_a" – the value used for group A
            "identity_b" – the value used for group B
            "text_a"     – resume text with group-A attribute
            "text_b"     – resume text with group-B attribute
        group_a_label : str – e.g. "white_american"
        group_b_label : str – e.g. "black_american"
    """
    if attribute == "name":
        data         = _load_names()
        group_keys   = list(data.keys())          # e.g. ["white_american", "black_american"]
        label_a, label_b = group_keys[0], group_keys[1]
        values_a     = data[label_a]
        values_b     = data[label_b]
        swap_fn      = _swap_name

    elif attribute == "university":
        data         = _load_universities()
        group_keys   = list(data.keys())          # e.g. ["well_known", "lesser_known"]
        label_a, label_b = group_keys[0], group_keys[1]
        values_a     = data[label_a]
        values_b     = data[label_b]
        swap_fn      = _swap_university

    else:
        raise ValueError(f"Unsupported attribute '{attribute}'. Choose 'name' or 'university'.")

    n = min(n_pairs, len(values_a), len(values_b))
    variants: List[Dict[str, Any]] = []

    for i in range(n):
        variants.append({
            "identity_a": values_a[i],
            "identity_b": values_b[i],
            "text_a":     swap_fn(original_text, values_a[i]),
            "text_b":     swap_fn(original_text, values_b[i]),
        })

    return variants, label_a, label_b


# ---------------------------------------------------------------------------
# Public API — run_audit
# ---------------------------------------------------------------------------

def run_audit(
    original_texts: List[str],
    job_description: str,
    attribute: Literal["name", "university"],
    n_pairs: int = 5,
    mode: Literal["raw", "redacted"] = "redacted",
) -> Dict[str, Any]:
    """Run the counterfactual bias audit across N resumes × M identity pairs.

    Parameters
    ----------
    original_texts : List[str]
        One or more raw resume texts (from /extract, before preprocessing).
        More resumes → more statistical power.
    job_description : str
        The job description to score against (held constant throughout).
    attribute : "name" | "university"
        The attribute whose effect we are measuring.
    n_pairs : int
        Number of identity pairs from the JSON file to test.
    mode : "raw" | "redacted"
        "raw"      – score WITHOUT strip_pii (embedding sees name/university).
        "redacted" – score through production pipeline (with strip_pii).
                     For names this yields delta ≈ 0 by design.

    Returns
    -------
    Dict[str, Any] with keys:
        attribute        : str
        group_a_label    : str
        group_b_label    : str
        mode             : str  — "raw" or "redacted"
        mean_delta       : float  — mean(score_B - score_A) across all pairs
        std_delta        : float  — standard deviation of deltas
        p_value          : float  — two-tailed paired t-test; NaN if < 2 observations
        significant      : bool   — p_value < 0.05
        n_observations   : int    — total (resumes × pairs) observations
        raw_rows         : List[Dict[str, Any]]  — per-pair detail
        disclaimer       : str

    Raw row keys: resume_index, identity_a, identity_b, score_a, score_b, delta
    """
    # Choose scorer based on mode
    scorer = _score_variant_raw if mode == "raw" else _score_variant

    raw_rows: List[Dict[str, Any]] = []

    for resume_idx, original_text in enumerate(original_texts):
        variants, label_a, label_b = build_variants(original_text, attribute, n_pairs)

        for v in variants:
            score_a = scorer(v["text_a"], job_description)
            score_b = scorer(v["text_b"], job_description)
            delta   = round(score_b - score_a, 4)

            raw_rows.append({
                "resume_index": resume_idx,
                "identity_a":   v["identity_a"],
                "identity_b":   v["identity_b"],
                "score_a":      round(score_a, 2),
                "score_b":      round(score_b, 2),
                "delta":        delta,
            })

    # --- Statistical summary ---
    deltas = [r["delta"] for r in raw_rows]
    n_obs  = len(deltas)

    mean_delta = round(float(np.mean(deltas)), 4) if deltas else 0.0
    std_delta  = round(float(np.std(deltas, ddof=1)), 4) if n_obs >= 2 else float("nan")

    # Paired t-test requires >= 2 paired observations
    if n_obs >= 2:
        scores_a = [r["score_a"] for r in raw_rows]
        scores_b = [r["score_b"] for r in raw_rows]
        _, p_value = stats.ttest_rel(scores_a, scores_b)
        p_value = round(float(p_value), 6)
    else:
        p_value = float("nan")

    significant = (not math.isnan(p_value)) and (p_value < 0.05)

    return {
        "attribute":      attribute,
        "group_a_label":  label_a,
        "group_b_label":  label_b,
        "mode":           mode,
        "mean_delta":     mean_delta,
        "std_delta":      std_delta,
        "p_value":        p_value,
        "significant":    significant,
        "n_observations": n_obs,
        "raw_rows":       raw_rows,
        "disclaimer":     DISCLAIMER,
    }
