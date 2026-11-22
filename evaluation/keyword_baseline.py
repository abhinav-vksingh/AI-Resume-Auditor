"""
evaluation/keyword_baseline.py – Literal keyword-overlap scorer.

PURPOSE
-------
This module is EVALUATION-ONLY.  It is NEVER imported or called by the real
pipeline.  Its only job is to give us a simple baseline to compare against
our semantic + skill pipeline in the evaluation report.

WHY A BASELINE?
---------------
If our embedding pipeline scores 72/100 on a resume–JD pair, is that good?
We only know if we have something to compare against.  The baseline answers:
"a dumb word-counter would give 45 — so our model adds real value."

TWO SCORERS
-----------
1. tfidf_cosine_score  – TF-IDF vectorisation + cosine similarity.
   Handles common words (stop-words) intelligently but still misses synonyms.

2. jaccard_overlap_score – Raw Jaccard on unique word sets.
   The simplest possible scorer.  Useful to show how low the floor is.

HOW TO USE (from the project root)
-----------------------------------
    python -m evaluation.keyword_baseline

The __main__ block runs a small self-contained demo.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Scorer 1: TF-IDF cosine similarity
# ---------------------------------------------------------------------------

def tfidf_cosine_score(resume_text: str, job_description: str) -> float:
    """Return TF-IDF cosine similarity between resume and JD, scaled 0–100.

    Both documents are vectorised jointly so the IDF weights are consistent.

    Parameters
    ----------
    resume_text : str
        Any plain-text version of the resume.
    job_description : str
        Raw job description text.

    Returns
    -------
    float
        Score in [0.0, 100.0].
    """
    vectoriser = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams to capture "machine learning"
        min_df=1,
    )
    tfidf_matrix = vectoriser.fit_transform([resume_text, job_description])
    sim: float = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(float(sim) * 100, 2)


# ---------------------------------------------------------------------------
# Scorer 2: Jaccard word-set overlap
# ---------------------------------------------------------------------------

def jaccard_overlap_score(resume_text: str, job_description: str) -> float:
    """Return Jaccard similarity on unique word sets, scaled 0–100.

    Jaccard = |intersection| / |union|

    This is the simplest possible text overlap scorer.  It treats every unique
    word equally and ignores word frequency entirely.

    Parameters
    ----------
    resume_text : str
    job_description : str

    Returns
    -------
    float
        Score in [0.0, 100.0].
    """
    res_words = set(resume_text.lower().split())
    jd_words  = set(job_description.lower().split())

    if not jd_words:
        return 0.0

    intersection = res_words & jd_words
    union        = res_words | jd_words
    return round(len(intersection) / len(union) * 100, 2)


# ---------------------------------------------------------------------------
# CLI demo — run with: python -m evaluation.keyword_baseline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _SAMPLE_RESUME = (
        "Software engineer with 3 years of Python and machine learning experience. "
        "Built REST APIs with FastAPI and deployed on AWS using Docker and Kubernetes. "
        "Experience with TensorFlow, scikit-learn, and data analysis using pandas."
    )
    _SAMPLE_JD = (
        "Looking for a backend engineer with Python, REST API, and cloud experience. "
        "Must know Docker, Kubernetes, and have exposure to machine learning frameworks "
        "such as TensorFlow or PyTorch."
    )

    print("=== Keyword Baseline Evaluation ===\n")

    tfidf_score = tfidf_cosine_score(_SAMPLE_RESUME, _SAMPLE_JD)
    jacc_score  = jaccard_overlap_score(_SAMPLE_RESUME, _SAMPLE_JD)

    print(f"TF-IDF cosine score : {tfidf_score:.1f}/100")
    print(f"Jaccard overlap     : {jacc_score:.1f}/100")

    # Compute the actual embedding (MiniLM) score on the same pair.
    # This requires the backend package — run from the project root.
    try:
        from backend.app.services.semantic import compute_semantic_similarity
        embed_score = compute_semantic_similarity(_SAMPLE_RESUME, _SAMPLE_JD, sections=None)
        print(f"Embedding (MiniLM)  : {embed_score:.1f}/100  [actual, computed on this pair]")
        print(f"\nDelta (Embedding − TF-IDF)  : {embed_score - tfidf_score:+.1f} points")
        print(f"Delta (Embedding − Jaccard) : {embed_score - jacc_score:+.1f} points")
    except ImportError as exc:
        print(f"\n[WARNING] Could not import backend service to compute embedding score: {exc}")
        print("Run from the project root: python -m evaluation.keyword_baseline")
