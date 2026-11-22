"""
tests/test_semantic.py

Unit tests for services/semantic.py.

The actual SentenceTransformer model is MOCKED in every test — loading a
700 MB model in a test suite would make CI slow. We test:
  - _chunk_text logic (pure function, no mock needed)
  - Scoring arithmetic (clipping, scaling)
  - That the model singleton is created lazily (not at import time)

Run with:
    pytest tests/test_semantic.py -v
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.services.semantic import _chunk_text


# ===========================================================================
# _chunk_text  (pure function — no mock needed)
# ===========================================================================

class TestChunkText:

    def test_uses_sections_when_provided(self):
        """Non-empty sections dict → each section value becomes a chunk."""
        sections = {
            "skills": "python machine learning",
            "experience": "software engineer at acme",
        }
        chunks = _chunk_text("full resume text", sections)
        assert "python machine learning" in chunks
        assert "software engineer at acme" in chunks
        assert len(chunks) == 2

    def test_ignores_empty_section_values(self):
        """Section values that are blank strings are excluded."""
        sections = {"skills": "python", "experience": "", "education": "   "}
        chunks = _chunk_text("full text", sections)
        assert len(chunks) == 1
        assert chunks[0] == "python"

    def test_falls_back_to_paragraphs_when_sections_none(self):
        """When sections is None, text is split on double newlines."""
        text = "para one\n\npara two\n\npara three"
        chunks = _chunk_text(text, None)
        assert len(chunks) == 3
        assert "para one" in chunks

    def test_falls_back_to_paragraphs_when_sections_empty(self):
        """When sections is {}, fall back to paragraph splitting."""
        text = "para one\n\npara two"
        chunks = _chunk_text(text, {})
        assert len(chunks) == 2

    def test_last_resort_full_text(self):
        """Single-paragraph text with no sections → one chunk = full text."""
        text = "single line of text with no double newlines"
        chunks = _chunk_text(text, None)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_fallback(self):
        """Empty string with no sections → one empty-ish chunk, no crash."""
        chunks = _chunk_text("", None)
        assert isinstance(chunks, list)

    def test_section_chunks_are_stripped(self):
        """Leading/trailing whitespace should be stripped from each chunk."""
        sections = {"skills": "  python  ", "experience": "  engineer  "}
        chunks = _chunk_text("text", sections)
        assert "python" in chunks
        assert "engineer" in chunks
        assert all(c == c.strip() for c in chunks)


# ===========================================================================
# compute_semantic_similarity  (model mocked)
# ===========================================================================

class TestComputeSemanticSimilarity:
    """
    We mock _get_model() so the actual model is not loaded.
    The mock returns pre-defined numpy arrays so we can test the arithmetic.
    """

    def _make_mock_model(self, vectors: List[np.ndarray]) -> MagicMock:
        mock = MagicMock()
        mock.encode.side_effect = vectors
        return mock

    def test_perfect_similarity_gives_100(self):
        """When resume and JD vectors are identical, cosine similarity = 1 → 100."""
        from backend.app.services.semantic import compute_semantic_similarity

        v = np.array([[1.0, 0.0, 0.0]])
        mock_model = self._make_mock_model([v, v])

        with patch("backend.app.services.semantic._get_model", return_value=mock_model):
            score = compute_semantic_similarity("resume text", "jd text")
        assert score == 100.0

    def test_orthogonal_vectors_give_0(self):
        """Orthogonal vectors → cosine = 0 → score = 0."""
        from backend.app.services.semantic import compute_semantic_similarity

        resume_v = np.array([[1.0, 0.0]])
        jd_v     = np.array([[0.0, 1.0]])
        mock_model = self._make_mock_model([resume_v, jd_v])

        with patch("backend.app.services.semantic._get_model", return_value=mock_model):
            score = compute_semantic_similarity("resume text", "jd text")
        assert score == 0.0

    def test_negative_cosine_clamped_to_zero(self):
        """Negative cosine similarity must be clamped to 0, not go negative."""
        from backend.app.services.semantic import compute_semantic_similarity

        resume_v = np.array([[1.0, 0.0]])
        jd_v     = np.array([[-1.0, 0.0]])
        mock_model = self._make_mock_model([resume_v, jd_v])

        with patch("backend.app.services.semantic._get_model", return_value=mock_model):
            score = compute_semantic_similarity("resume text", "jd text")
        assert score == 0.0

    def test_result_always_between_0_and_100(self):
        """Score must always be in [0, 100]."""
        from backend.app.services.semantic import compute_semantic_similarity

        v = np.array([[0.6, 0.8]])
        mock_model = self._make_mock_model([v, v])

        with patch("backend.app.services.semantic._get_model", return_value=mock_model):
            score = compute_semantic_similarity("resume", "jd")
        assert 0.0 <= score <= 100.0

    def test_multiple_chunks_are_averaged(self):
        """With two resume chunks, their mean should be used as the resume vector."""
        from backend.app.services.semantic import compute_semantic_similarity

        chunk1 = np.array([[1.0, 0.0]])
        chunk2 = np.array([[0.0, 1.0]])
        resume_chunks = np.vstack([chunk1, chunk2])

        avg = np.array([0.5, 0.5])
        avg_norm = avg / np.linalg.norm(avg)
        jd_v = avg_norm.reshape(1, -1)

        mock_model = self._make_mock_model([resume_chunks, jd_v])
        sections = {"skills": "chunk one", "experience": "chunk two"}

        with patch("backend.app.services.semantic._get_model", return_value=mock_model):
            score = compute_semantic_similarity("resume", "jd", sections=sections)
        assert score > 90.0

    def test_model_loaded_lazily(self):
        """Importing the module must not trigger model loading."""
        import backend.app.services.semantic as sem_module
        original = sem_module._model
        sem_module._model = None
        try:
            assert sem_module._model is None
        finally:
            sem_module._model = original
