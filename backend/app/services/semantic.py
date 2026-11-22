"""
services/semantic.py – Semantic similarity between resume and job description.

Uses sentence-transformers (all-MiniLM-L6-v2) to encode both texts as dense
vectors, then computes cosine similarity scaled to 0–100.

CHUNKING RATIONALE (important for viva)
----------------------------------------
all-MiniLM-L6-v2 has a hard limit of 256 word-pieces.  A typical one-page
resume already has 300–600 tokens, so the last third of the document gets
silently TRUNCATED if we pass the whole text at once.  That would systematically
penalise long resumes and make experience/projects sections invisible.

Fix: split the resume into logical chunks (sections if available, else
paragraphs), encode each chunk separately, then AVERAGE the resulting vectors.
Averaging works because:
  • Each chunk embedding lives in the same 384-dimensional space.
  • The average is a valid point in that space and captures the centroid of
    the resume's semantic content.
  • It is fast (single matmul at the end) and requires no training.

We do NOT chunk the job description — JDs are usually short (<256 tokens).
If the JD is long, we truncate it (this is acceptable because JDs front-load
the most important requirements).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

# If the sentence-transformers model is already cached locally, default to offline
# mode so HuggingFace does not perform slow HTTP retries checking for updates.
_hub_folder = "models--sentence-transformers--all-MiniLM-L6-v2"
_cache_path = Path.home() / ".cache" / "huggingface" / "hub" / _hub_folder
if _cache_path.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from backend.app.config import settings

# ---------------------------------------------------------------------------
# Singleton model — loaded on first call, not at import time.
#
# Why not at import time?
#   Importing this module during tests or CLI usage would download/load the
#   700 MB model even when not needed.  Lazy loading keeps `import` fast.
#
# Why not inside compute_semantic_similarity every call?
#   Loading the model takes ~2 s.  Reloading it on every request would make
#   the API unusable.
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it if needed."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.model_name)
    return _model


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, sections: Optional[Dict[str, str]]) -> List[str]:
    """Split resume text into encode-able chunks.

    Priority:
      1. Use section values if sections dict is provided and non-empty.
         Section text is semantically coherent and already below token limit.
      2. Fall back to paragraph splitting (double-newline boundary).
      3. If still nothing, return the full text as one chunk.

    Parameters
    ----------
    text : str
        Full resume core_text (from preprocessing).
    sections : Optional[Dict[str, str]]
        Output of detect_sections(), or None if not available.

    Returns
    -------
    List[str]
        Non-empty text chunks ready for encoding.
    """
    # Strategy 1: use sections
    if sections:
        chunks = [v.strip() for v in sections.values() if v.strip()]
        if chunks:
            return chunks

    # Strategy 2: paragraph splitting
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs

    # Strategy 3: entire text as one chunk (last resort)
    return [text.strip()] if text.strip() else [""]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_semantic_similarity(
    resume_text: str,
    job_description: str,
    sections: Optional[Dict[str, str]] = None,
) -> float:
    """Return cosine similarity between the resume and JD, scaled 0–100.

    Parameters
    ----------
    resume_text : str
        core_text from preprocessing (lowercased, PII-stripped).
    job_description : str
        Raw or lightly cleaned job description text.
    sections : dict[str, str] | None
        Section dict from detect_sections(); used for chunking the resume.

    Returns
    -------
    float
        Similarity score in [0.0, 100.0].

    Algorithm
    ---------
    1. Split resume into chunks (section → paragraph → full text).
    2. Encode all chunks in one batched call (efficient GPU/CPU usage).
    3. Average the chunk embeddings → single resume vector.
    4. Encode the full JD as one vector.
    5. Cosine similarity between resume vector and JD vector.
    6. Clip to [0, 1] (cosine can be slightly negative for unrelated texts)
       then scale × 100.
    """
    model = _get_model()

    # --- Encode resume (chunked) ---
    resume_chunks = _chunk_text(resume_text, sections)
    # batch_size=32 avoids OOM on machines with limited RAM
    resume_embeddings = model.encode(resume_chunks, batch_size=32, show_progress_bar=False)
    # Shape: (n_chunks, 384)

    # Average chunk embeddings → single resume vector (shape: (1, 384))
    resume_vector = np.mean(resume_embeddings, axis=0, keepdims=True)

    # --- Encode JD (single call; JDs are usually short) ---
    jd_vector = model.encode([job_description], show_progress_bar=False)
    # Shape: (1, 384)

    # --- Cosine similarity ---
    # cosine_similarity returns a (1,1) matrix
    sim: float = cosine_similarity(resume_vector, jd_vector)[0][0]

    # Clip to [0, 1] — raw cosine can be slightly negative for orthogonal texts.
    # We treat negative similarity as 0 (no relation) rather than a penalty.
    sim = float(np.clip(sim, 0.0, 1.0))

    return round(sim * 100, 2)
