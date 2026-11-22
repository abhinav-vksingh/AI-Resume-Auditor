"""
services/explain.py – Rule-based explanation generator.

Generates human-readable positive and negative factors from the pipeline's
numeric outputs.  ZERO LLM calls happen here — every sentence is produced
by a deterministic if/elif/else rule applied to the numbers.

Why rule-based?
  • It is fully auditable: change a threshold → change the output, predictably.
  • It satisfies the spec principle "everything must be explainable".
  • It is fast (no network call, no model inference).
  • In a viva you can point at a threshold and explain exactly why it was chosen.

RULES OVERVIEW
--------------
Positive factors:
  P1  final_score >= 75     → "Overall strong candidate match …"
  P2  semantic_score >= 75  → "Strong semantic alignment …"
  P2b semantic_score >= 50  → "Moderate semantic alignment …"
  P3  matched skills > 0    → "Skills directly evidenced: …"
  P4  skill_overlap >= 70   → "Good skill coverage …"

Negative factors:
  N1  semantic_score < 50   → "Low semantic similarity …"
  N2  skill_overlap < 40    → "Skill coverage is low …"
  N3  per missing skill      → '"{skill}" is required by the JD but no evidence …'
      (capped at MAX_MISSING_SHOWN; a summary line added if more exist)
  N4  per weak skill         → '"{skill}" was mentioned only once …'

Order: positive factors are listed most-impactful-first; negative factors put
per-skill messages after thematic messages so the reader sees the summary before
the detail.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Maximum number of individual missing-skill lines before we collapse the rest
# into a summary.  Keeps the response readable.
_MAX_MISSING_SHOWN = 5


def generate_explanation(
    semantic_score: float,
    skill_overlap: float,
    final_score: float,
    skill_table: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Produce positive and negative explanation factors from pipeline numbers.

    Parameters
    ----------
    semantic_score : float
        Cosine similarity score [0, 100] from semantic.py.
    skill_overlap : float
        Skill overlap score [0, 100] from skills.py.
    final_score : float
        Weighted final score [0, 100] from scoring.py.
    skill_table : List[Dict[str, Any]]
        Each dict has "skill" (str) and "status" ("Matched"/"Weak"/"Missing").

    Returns
    -------
    Dict[str, List[str]] with keys:
        "positive_factors" : List[str] — things working in the candidate's favour.
        "negative_factors" : List[str] — gaps or weaknesses to highlight.

    Notes
    -----
    Either list may be empty if the data gives no signal in that direction
    (e.g. a perfect resume has no negative factors).
    """
    positive: List[str] = []
    negative: List[str] = []

    # Bucket the skill table once — used by multiple rules below
    matched_skills = [r["skill"] for r in skill_table if r["status"] == "Matched"]
    weak_skills    = [r["skill"] for r in skill_table if r["status"] == "Weak"]
    missing_skills = [r["skill"] for r in skill_table if r["status"] == "Missing"]
    total_required = len(skill_table)

    # -----------------------------------------------------------------------
    # POSITIVE FACTORS
    # -----------------------------------------------------------------------

    # P1 — Overall match strength
    if final_score >= 75:
        positive.append(
            f"Overall strong candidate match (final score: {final_score}/100)."
        )

    # P2 — Semantic alignment
    if semantic_score >= 75:
        positive.append(
            f"Strong semantic alignment between resume and job description "
            f"({semantic_score}/100) — the resume language closely mirrors the JD."
        )
    elif semantic_score >= 50:
        positive.append(
            f"Moderate semantic alignment between resume and job description "
            f"({semantic_score}/100) — the resume is broadly relevant to the role."
        )

    # P3 — Matched skills list
    if matched_skills:
        display = matched_skills[:5]
        suffix  = (
            f" and {len(matched_skills) - 5} more"
            if len(matched_skills) > 5
            else ""
        )
        positive.append(
            f"Skills directly evidenced in the resume: "
            f"{', '.join(display)}{suffix}."
        )

    # P4 — Skill coverage band
    if total_required > 0 and skill_overlap >= 70:
        positive.append(
            f"Good skill coverage ({skill_overlap}/100) — "
            f"{len(matched_skills)} of {total_required} required skills are matched."
        )

    # -----------------------------------------------------------------------
    # NEGATIVE FACTORS
    # -----------------------------------------------------------------------

    # N1 — Low semantic similarity (thematic, comes first)
    if semantic_score < 50:
        negative.append(
            f"Low semantic similarity ({semantic_score}/100) — the resume content "
            "may not be tailored to this role; consider rewriting the summary and "
            "experience bullets to mirror the JD language."
        )

    # N2 — Weak skill coverage (thematic, before per-skill details)
    if total_required > 0 and skill_overlap < 40:
        negative.append(
            f"Skill coverage is low ({skill_overlap}/100) — the resume does not "
            f"demonstrate most of the {total_required} skills required by the JD."
        )

    # N3 — Per missing skill (the most actionable feedback)
    for skill in missing_skills[:_MAX_MISSING_SHOWN]:
        negative.append(
            f'"{skill}" is required by the JD but no evidence was found in the resume.'
        )
    if len(missing_skills) > _MAX_MISSING_SHOWN:
        remaining = len(missing_skills) - _MAX_MISSING_SHOWN
        negative.append(
            f"… and {remaining} more required skill(s) are absent from the resume."
        )

    # N4 — Per weak skill
    for skill in weak_skills:
        negative.append(
            f'"{skill}" was mentioned only once and not in a core skills or '
            "experience section — add a concrete example to strengthen this claim."
        )

    return {
        "positive_factors": positive,
        "negative_factors": negative,
    }
