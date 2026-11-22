"""
services/scoring.py – Final score fusion.

Formula (from project spec):
    final_score = semantic_weight * semantic_score + skill_weight * skill_overlap

Both components and the weights are returned so every caller can audit the
arithmetic — no black box.

PRINCIPLE: This module computes the number.  An LLM must NEVER override or
recalculate this score (spec rule 1).  The explain module only reads the
output; it cannot change it.
"""

from __future__ import annotations

import numpy as np

from backend.app.config import settings


def compute_final_score(
    semantic_score: float,
    skill_overlap: float,
) -> dict:
    """Fuse semantic similarity and skill overlap into one final score.

    Parameters
    ----------
    semantic_score : float
        Output of semantic.compute_semantic_similarity() — in [0, 100].
    skill_overlap : float
        Output of skills.compute_skill_overlap() — in [0, 100].

    Returns
    -------
    dict with keys:
        "final_score"   : float — weighted sum, clamped to [0, 100].
        "semantic_score": float — the semantic component (passed through).
        "skill_overlap" : float — the skill component (passed through).
        "weights"       : dict  — {"semantic": w1, "skill_overlap": w2}
                          read from config so a .env change propagates here.

    Why read weights from settings at call time (not module-load time)?
        Tests can monkeypatch settings without restarting the app.  It also
        makes the dependency explicit: whoever calls this function implicitly
        accepts whatever weights are in config.
    """
    w_sem   = settings.semantic_weight   # e.g. 0.6
    w_skill = settings.skill_weight      # e.g. 0.4

    # Weighted sum — both components already in [0, 100]
    raw = w_sem * semantic_score + w_skill * skill_overlap

    # Clamp: floating-point arithmetic can produce 100.000000001 etc.
    final_score = round(float(np.clip(raw, 0.0, 100.0)), 2)

    return {
        "final_score":    final_score,
        "semantic_score": round(float(semantic_score), 2),
        "skill_overlap":  round(float(skill_overlap), 2),
        "weights": {
            "semantic":      w_sem,
            "skill_overlap": w_skill,
        },
    }
