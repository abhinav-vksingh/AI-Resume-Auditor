"""
services/skills.py – Skill extraction and Matched / Weak / Missing classification.

DESIGN
------
1. Load the curated skill dictionary from data/skills.json (all categories).
2. Extract REQUIRED skills from the job description — a skill is "required" if
   it appears in the JD (after cleaning + alias normalisation).
3. For each required skill, classify against the resume:

   Matched  — skill found in a PRIMARY section (skills / experience / projects).
              These sections prove the candidate actively uses the skill.
              OR found more than once anywhere in the resume (repeated mention
              signals genuine familiarity).
   Weak     — skill found exactly once, but NOT in a primary section.
              Could be a passing mention ("familiar with X") — counted at half
              weight so it nudges the score upward but does not fully credit it.
   Missing  — skill not found anywhere in the resume.

4. skill_overlap = (matched + 0.5 × weak) / total_required × 100
   The 0.5 weight is a deliberate penalty: weak evidence is worth half credit.

Whole-word matching (\\b) prevents "ml" hitting "html", "node" hitting "nodejs".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Pattern

from backend.app.services.preprocessing import clean_text, normalize_aliases

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SKILLS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "skills.json"

# Sections where finding a skill counts as a STRONG (Matched) signal.
# Skills listed here prove active usage, not just passing familiarity.
_PRIMARY_SECTIONS = {"skills", "experience", "projects"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all_skills() -> List[str]:
    """Return a flat list of all skills from skills.json (all categories).

    The skills are already lowercase strings; they are matched against
    lowercased resume/JD text.
    """
    if not _SKILLS_PATH.exists():
        return []
    with _SKILLS_PATH.open(encoding="utf-8") as fh:
        data: Dict[str, List[str]] = json.load(fh)
    # Flatten all categories into one list
    return [skill for skills in data.values() for skill in skills]


def _make_pattern(skill: str) -> Pattern:
    """Compile a whole-word, case-insensitive regex for *skill*."""
    return re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SkillStatus = Literal["Matched", "Weak", "Missing"]


def extract_required_skills(job_description: str) -> List[str]:
    """Find which skills from our dictionary appear in the job description.

    Parameters
    ----------
    job_description : str
        Raw JD text (not yet preprocessed).

    Returns
    -------
    List[str]
        Skills from skills.json that are mentioned in the JD, deduplicated,
        in the order they appear in the skills dictionary.
    """
    all_skills = _load_all_skills()

    # Lightly preprocess the JD: lowercase + alias normalisation.
    # We do NOT strip PII — JDs don't have PII to strip.
    jd_cleaned = clean_text(job_description)
    jd_normalised = normalize_aliases(jd_cleaned)

    # Some skills contain punctuation (e.g. "next.js") where alias normalisation
    # can corrupt the token (the "js" alias expands the ".js" suffix into
    # ".javascript").  We therefore check each skill against BOTH the cleaned
    # text (before alias expansion) and the normalised text (after expansion) so
    # that neither punctuated skill names nor abbreviation-only references are
    # missed.
    required: List[str] = []
    for skill in all_skills:
        pattern = _make_pattern(skill)
        if pattern.search(jd_normalised) or pattern.search(jd_cleaned):
            required.append(skill)

    return required


def classify_skill(
    skill: str,
    core_text: str,
    sections: Dict[str, str],
) -> SkillStatus:
    """Classify a single skill as Matched, Weak, or Missing.

    Parameters
    ----------
    skill : str
        Canonical skill name (lowercase, from skills.json).
    core_text : str
        Full preprocessed resume text (for counting total occurrences).
    sections : dict[str, str]
        Section-labelled resume text from detect_sections().

    Returns
    -------
    "Matched" | "Weak" | "Missing"
    """
    pattern = _make_pattern(skill)

    # --- Check primary sections first (strong signal) ---
    for sec_name in _PRIMARY_SECTIONS:
        if sec_name in sections and pattern.search(sections[sec_name]):
            return "Matched"

    # --- Count total occurrences in core_text ---
    total_occurrences = len(pattern.findall(core_text))

    if total_occurrences > 1:
        # Repeated mention across the document = genuine familiarity
        return "Matched"
    elif total_occurrences == 1:
        # Mentioned once, not in a primary section → weak signal
        return "Weak"
    else:
        return "Missing"


def build_skill_table(
    required_skills: List[str],
    core_text: str,
    sections: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Classify every required skill and return the full table.

    Parameters
    ----------
    required_skills : List[str]
        Output of extract_required_skills().
    core_text : str
        Preprocessed resume text.
    sections : Dict[str, str]
        Section dict from detect_sections().

    Returns
    -------
    List[Dict[str, Any]]
        Each dict has keys "skill" (str) and "status" ("Matched"/"Weak"/"Missing").
    """
    table = []
    for skill in required_skills:
        status = classify_skill(skill, core_text, sections)
        table.append({"skill": skill, "status": status})
    return table


def compute_skill_overlap(matched: int, weak: int, total: int) -> float:
    """Return skill_overlap score in [0, 100].

    Formula (from project spec):
        skill_overlap = (matched + 0.5 × weak) / total × 100

    Weak skills get half credit because a single passing mention is evidence
    but not proof of proficiency.

    Returns 0.0 when total == 0 (no required skills found in JD).
    """
    if total == 0:
        return 0.0
    raw = (matched + 0.5 * weak) / total
    return round(float(raw * 100), 2)


def run_skill_analysis(
    job_description: str,
    core_text: str,
    sections: Dict[str, str],
) -> Dict[str, Any]:
    """Full skill analysis pipeline — called by the /match router.

    Returns
    -------
    dict with keys:
        "skill_table"    : list[dict]  — per-skill status
        "skill_overlap"  : float       — 0–100 score
        "matched_count"  : int
        "weak_count"     : int
        "missing_count"  : int
        "total_required" : int
    """
    required = extract_required_skills(job_description)
    table = build_skill_table(required, core_text, sections)

    matched = sum(1 for r in table if r["status"] == "Matched")
    weak    = sum(1 for r in table if r["status"] == "Weak")
    missing = sum(1 for r in table if r["status"] == "Missing")
    total   = len(table)

    return {
        "skill_table":    table,
        "skill_overlap":  compute_skill_overlap(matched, weak, total),
        "matched_count":  matched,
        "weak_count":     weak,
        "missing_count":  missing,
        "total_required": total,
    }
