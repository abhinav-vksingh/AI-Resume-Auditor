"""
tests/test_routers.py – FastAPI TestClient integration tests.

Covers every HTTP endpoint with one success case and one error/validation case each:
  GET  /health
  POST /extract
  POST /preprocess
  POST /match
  POST /analyze
  POST /bias-audit
  POST /feedback

Heavy services (SentenceTransformer, LLM) are mocked so the model is never
loaded and no API key is ever used or printed.

HOW TO RUN (from the project root)
-----------------------------------
    python -m pytest tests/test_routers.py -v
"""

from __future__ import annotations

import io
from typing import Any, Dict
from unittest.mock import patch

import fitz  # PyMuPDF — already a project dependency
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

# ---------------------------------------------------------------------------
# Shared client
# ---------------------------------------------------------------------------

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper: build a minimal valid PDF in memory using fitz
# ---------------------------------------------------------------------------

def _make_pdf(text: str = "Python developer with Docker and AWS experience.") -> bytes:
    """Return bytes of a single-page PDF containing *text*.

    Uses PyMuPDF (fitz), which is already installed in the project.
    """
    doc = fitz.open()          # empty document
    page = doc.new_page()      # add one page (A4 by default)
    page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# Reusable minimal PDF bytes for the whole test session
_PDF_BYTES = _make_pdf()

# A short resume text that the bias-audit and feedback endpoints accept directly
_RESUME_TEXT = (
    "John Smith\n"
    "Experience: 5 years of Python and machine learning.\n"
    "Skills: Python, Docker, AWS, scikit-learn, REST API.\n"
)
_JD_TEXT = "Looking for a Python developer with Docker and AWS skills."

# Minimal structured feedback dict returned by the mock
_MOCK_FEEDBACK_RESULT: Dict[str, Any] = {
    "prompt_version": "v3",
    "format_valid": True,
    "feedback": {
        "matched_skills": ["python", "docker"],
        "gaps": [],
        "rewritten_bullet": "Built Python microservices deployed on Docker.",
        "explanation": "Strong match.",
        "recommendations": ["Quantify impact."],
    },
    "raw_response": '{"matched_skills":["python","docker"]}',
    "error": None,
}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /health should return 200 with status=ok and a version string."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


# ---------------------------------------------------------------------------
# POST /extract
# ---------------------------------------------------------------------------

class TestExtractEndpoint:
    def test_extract_valid_pdf(self):
        """POST /extract with a real in-memory PDF should return 200 and text."""
        resp = client.post(
            "/extract",
            files={"file": ("resume.pdf", _PDF_BYTES, "application/pdf")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert body["page_count"] >= 1
        # The inserted text should be in the extraction result
        assert "python" in body["text"].lower()

    def test_extract_empty_file_returns_400(self):
        """POST /extract with an empty file should return 400."""
        resp = client.post(
            "/extract",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /preprocess
# ---------------------------------------------------------------------------

class TestPreprocessEndpoint:
    def test_preprocess_valid_text(self):
        """POST /preprocess with real text should return 200 and all three fields."""
        resp = client.post(
            "/preprocess",
            json={"text": _RESUME_TEXT},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "core_text" in body
        assert "original_text" in body
        assert "sections" in body
        assert isinstance(body["sections"], dict)

    def test_preprocess_empty_text_returns_422(self):
        """POST /preprocess with an empty string should fail Pydantic validation (422)."""
        resp = client.post("/preprocess", json={"text": ""})
        # Pydantic min_length=1 → FastAPI returns 422 Unprocessable Entity
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /match
# ---------------------------------------------------------------------------

class TestMatchEndpoint:
    def test_match_success(self):
        """POST /match with mocked semantic service should return 200."""
        with patch(
            "backend.app.routers.match.compute_semantic_similarity",
            return_value=72.5,
        ):
            resp = client.post(
                "/match",
                json={
                    "core_text": "python developer docker aws",
                    "job_description": _JD_TEXT,
                    "sections": {},
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["semantic_score"] == 72.5
        assert "skill_table" in body

    def test_match_no_skills_returns_422(self):
        """POST /match with a JD containing no known skills should return 422."""
        with patch(
            "backend.app.routers.match.compute_semantic_similarity",
            return_value=10.0,
        ):
            resp = client.post(
                "/match",
                json={
                    "core_text": "experienced professional",
                    "job_description": "looking for a team player",  # no known skills
                    "sections": {},
                },
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoint:
    def test_analyze_success(self):
        """POST /analyze with a valid PDF and JD should return 200 with score fields."""
        with (
            patch(
                "backend.app.routers.analyze.compute_semantic_similarity",
                return_value=65.0,
            ),
            patch(
                "backend.app.routers.analyze.run_skill_analysis",
                return_value={
                    "skill_table": [{"skill": "python", "status": "Matched"}],
                    "skill_overlap": 100.0,
                    "matched_count": 1,
                    "weak_count": 0,
                    "missing_count": 0,
                    "total_required": 1,
                },
            ),
        ):
            resp = client.post(
                "/analyze",
                files={"file": ("resume.pdf", _PDF_BYTES, "application/pdf")},
                data={"job_description": _JD_TEXT},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "final_score" in body
        assert "components" in body
        assert "skills" in body
        assert "original_text" in body  # B2 requirement: original_text returned

    def test_analyze_empty_file_returns_400(self):
        """POST /analyze with empty bytes should return 400."""
        resp = client.post(
            "/analyze",
            files={"file": ("empty.pdf", b"", "application/pdf")},
            data={"job_description": _JD_TEXT},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /bias-audit
# ---------------------------------------------------------------------------

class TestBiasAuditEndpoint:
    def test_bias_audit_success(self):
        """POST /bias-audit with valid payload should return 200."""
        with patch(
            "backend.app.routers.bias.run_audit",
            return_value={
                "attribute": "name",
                "group_a_label": "white_american",
                "group_b_label": "black_american",
                "mode": "redacted",
                "mean_delta": 0.0,
                "std_delta": 0.0,
                "p_value": None,
                "significant": False,
                "n_observations": 2,
                "raw_rows": [
                    {
                        "resume_index": 0,
                        "identity_a": "Emily Walsh",
                        "identity_b": "Lakisha Jones",
                        "score_a": 70.0,
                        "score_b": 70.0,
                        "delta": 0.0,
                    },
                    {
                        "resume_index": 0,
                        "identity_a": "Anne Schmidt",
                        "identity_b": "Jamal Jones",
                        "score_a": 70.0,
                        "score_b": 70.0,
                        "delta": 0.0,
                    },
                ],
                "disclaimer": "Project-level audit only.",
            },
        ):
            resp = client.post(
                "/bias-audit",
                json={
                    "original_text": _RESUME_TEXT,
                    "job_description": _JD_TEXT,
                    "attribute": "name",
                    "n_pairs": 2,
                    "mode": "redacted",
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["attribute"] == "name"
        assert "mean_delta" in body
        assert len(body["raw_rows"]) == 2

    def test_bias_audit_invalid_attribute_returns_422(self):
        """POST /bias-audit with an unknown attribute should return 422."""
        resp = client.post(
            "/bias-audit",
            json={
                "original_text": _RESUME_TEXT,
                "job_description": _JD_TEXT,
                "attribute": "age",  # not in Literal["name","university"]
                "n_pairs": 2,
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

class TestFeedbackEndpoint:
    def test_feedback_success(self):
        """POST /feedback with mocked LLM should return 200 with structured output."""
        with patch(
            "backend.app.routers.feedback.generate_feedback",
            return_value=_MOCK_FEEDBACK_RESULT,
        ):
            resp = client.post(
                "/feedback",
                params={"prompt_version": "v3"},
                json={
                    "job_description": _JD_TEXT,
                    "resume_text": _RESUME_TEXT,
                    "skill_table": [{"skill": "python", "status": "Matched"}],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["prompt_version"] == "v3"
        assert body["format_valid"] is True
        assert "feedback" in body

    def test_feedback_invalid_prompt_version_returns_422(self):
        """POST /feedback with an unsupported prompt_version should return 422."""
        resp = client.post(
            "/feedback",
            params={"prompt_version": "v99"},  # not in Literal["v1","v2","v3"]
            json={
                "job_description": _JD_TEXT,
                "resume_text": _RESUME_TEXT,
                "skill_table": [],
            },
        )
        assert resp.status_code == 422
