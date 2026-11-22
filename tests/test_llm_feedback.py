"""
tests/test_llm_feedback.py

Unit tests for services/llm_feedback.py (Phase 6).

All tests use mocks — no actual HTTP or external LLM API calls occur.
Verifies prompt resolution, string formatting, JSON extraction, schema validation,
and graceful failure without crashing on malformed LLM responses.

Run with:
    pytest tests/test_llm_feedback.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.app.services.llm_feedback import (
    extract_json_block,
    format_prompt,
    generate_feedback,
    load_prompt_template,
    parse_and_validate_feedback,
    resolve_prompt_file,
)


# ===========================================================================
# Prompt Loading & Formatting
# ===========================================================================

class TestPromptLoading:

    def test_resolve_prompt_file_valid_versions(self):
        for version in ["v1", "v2", "v3", "v1_baseline", "v2_criteria", "v3_structured"]:
            path = resolve_prompt_file(version)
            assert path.exists(), f"Prompt file for {version} should exist."

    def test_resolve_prompt_file_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt version"):
            resolve_prompt_file("v99_nonexistent")

    def test_load_prompt_template_contains_placeholders(self):
        for v in ["v1", "v2", "v3"]:
            tmpl = load_prompt_template(v)
            assert "{job_description}" in tmpl
            assert "{resume_text}" in tmpl
            assert "{skill_table}" in tmpl

    def test_format_prompt_injects_values(self):
        jd = "Senior Python Engineer"
        resume = "5 years Python, Docker, SQL"
        skills = [{"skill": "python", "status": "Matched"}]

        formatted = format_prompt("v3", jd, resume, skills)
        assert "Senior Python Engineer" in formatted
        assert "5 years Python, Docker, SQL" in formatted
        assert "python" in formatted


# ===========================================================================
# JSON Extraction & Schema Validation
# ===========================================================================

class TestJsonParsingAndValidation:

    def test_extract_json_pure_json(self):
        raw = '{"matched_skills": ["python"], "gaps": []}'
        assert extract_json_block(raw) == raw

    def test_extract_json_markdown_fenced(self):
        raw = '```json\n{"matched_skills": ["python"]}\n```'
        assert extract_json_block(raw) == '{"matched_skills": ["python"]}'

    def test_extract_json_markdown_no_tag(self):
        raw = '```\n{"matched_skills": ["python"]}\n```'
        assert extract_json_block(raw) == '{"matched_skills": ["python"]}'

    def test_extract_json_surrounded_by_chatter(self):
        raw = 'Here is your critique:\n{"matched_skills": ["python"]}\nHope this helps!'
        assert extract_json_block(raw) == '{"matched_skills": ["python"]}'

    def test_parse_valid_complete_schema(self):
        data = {
            "matched_skills": ["python", "docker"],
            "gaps": ["aws"],
            "rewritten_bullet": "Built REST APIs with Python.",
            "explanation": "Solid engineering profile.",
            "recommendations": ["Gain AWS experience."],
        }
        raw_text = json.dumps(data)
        parsed, valid = parse_and_validate_feedback(raw_text)

        assert valid is True
        assert parsed["matched_skills"] == ["python", "docker"]
        assert parsed["gaps"] == ["aws"]
        assert parsed["rewritten_bullet"] == "Built REST APIs with Python."
        assert parsed["explanation"] == "Solid engineering profile."
        assert parsed["recommendations"] == ["Gain AWS experience."]

    def test_parse_missing_keys_flags_format_invalid(self):
        """Missing required keys should flag format_valid=False without crashing."""
        data = {
            "matched_skills": ["python"],
            # 'gaps', 'rewritten_bullet', 'explanation', 'recommendations' are missing
        }
        raw_text = json.dumps(data)
        parsed, valid = parse_and_validate_feedback(raw_text)

        assert valid is False
        assert isinstance(parsed["gaps"], list)
        assert isinstance(parsed["recommendations"], list)
        assert parsed["rewritten_bullet"] == ""

    def test_parse_invalid_json_does_not_crash(self):
        """Completely malformed non-JSON output flags failure gracefully."""
        raw_text = "I am an AI and I think this resume is great but I forgot to write JSON."
        parsed, valid = parse_and_validate_feedback(raw_text)

        assert valid is False
        assert parsed["matched_skills"] == []
        assert "Format Error" in parsed["explanation"]

    def test_parse_wrong_types_flags_invalid(self):
        """matched_skills as string instead of list must flag format_valid=False."""
        data = {
            "matched_skills": "python, java",  # string, not list!
            "gaps": [],
            "rewritten_bullet": "Rewritten line",
            "explanation": "Good",
            "recommendations": [],
        }
        raw_text = json.dumps(data)
        _, valid = parse_and_validate_feedback(raw_text)
        assert valid is False


# ===========================================================================
# generate_feedback (Service Pipeline)
# ===========================================================================

class TestGenerateFeedback:

    def test_generate_feedback_success(self):
        mock_output = json.dumps({
            "matched_skills": ["python"],
            "gaps": ["kubernetes"],
            "rewritten_bullet": "Optimized Python endpoints.",
            "explanation": "Strong candidate with targeted gap.",
            "recommendations": ["Study Kubernetes."],
        })

        with patch("backend.app.services.llm_feedback.call_llm", return_value=mock_output):
            result = generate_feedback(
                job_description="Python + Kubernetes",
                resume_text="Python developer",
                skill_table=[{"skill": "python", "status": "Matched"}],
                prompt_version="v3",
            )

        assert result["format_valid"] is True
        assert result["error"] is None
        assert result["feedback"]["matched_skills"] == ["python"]
        assert result["feedback"]["gaps"] == ["kubernetes"]

    def test_generate_feedback_llm_exception_handled(self):
        """If call_llm throws an exception, generate_feedback returns format_valid=False and error string."""
        with patch("backend.app.services.llm_feedback.call_llm", side_effect=RuntimeError("Provider unreachable")):
            result = generate_feedback(
                job_description="Python",
                resume_text="Python",
                skill_table=[],
                prompt_version="v2",
            )

        assert result["format_valid"] is False
        assert "Provider unreachable" in result["error"]
        assert result["feedback"]["matched_skills"] == []

    def test_generate_feedback_malformed_llm_output_handled(self):
        """Malformed response is caught, flagged, and returned with empty structured feedback."""
        with patch("backend.app.services.llm_feedback.call_llm", return_value="Not valid JSON at all"):
            result = generate_feedback(
                job_description="Python",
                resume_text="Python",
                skill_table=[],
                prompt_version="v1",
            )

        assert result["format_valid"] is False
        assert "format validation" in result["error"].lower()
        assert result["raw_response"] == "Not valid JSON at all"
