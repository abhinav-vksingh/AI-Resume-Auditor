"""
tests/test_bias.py

Unit tests for services/bias.py.

We test pure logic (swap functions, stat calculations) without calling the
real scoring pipeline or the sentence-transformers model.

Run with:
    pytest tests/test_bias.py -v
"""

from __future__ import annotations

import math
from typing import List
from unittest.mock import patch

import pytest

from backend.app.services.bias import (
    _swap_name,
    _swap_university,
    build_variants,
    run_audit,
    DISCLAIMER,
    _score_variant,
    _score_variant_raw,
)


# ===========================================================================
# _swap_name
# ===========================================================================

class TestSwapName:

    def test_replaces_first_non_empty_line(self):
        text = "John Doe\njohndoe@example.com\nPython Developer"
        result = _swap_name(text, "Emily Walsh")
        lines = result.split("\n")
        assert lines[0] == "Emily Walsh"

    def test_rest_of_text_unchanged(self):
        text = "John Doe\njohndoe@example.com\nPython Developer"
        result = _swap_name(text, "Emily Walsh")
        lines = result.split("\n")
        # Lines after the first must be identical
        assert lines[1] == "johndoe@example.com"
        assert lines[2] == "Python Developer"

    def test_skips_leading_blank_lines(self):
        """If the first line is blank, skip to the next non-empty one."""
        text = "\n\nJohn Doe\nPython Developer"
        result = _swap_name(text, "Lakisha Jones")
        assert "Lakisha Jones" in result
        assert "John Doe" not in result

    def test_only_first_occurrence_swapped(self):
        """Name appears on line 1 only; line 3 (same name) must stay."""
        text = "John Doe\nExperience: 3 years\nJohn Doe again"
        result = _swap_name(text, "Emily Walsh")
        lines = result.split("\n")
        assert lines[0] == "Emily Walsh"
        # Third line untouched
        assert lines[2] == "John Doe again"

    def test_empty_text_fallback(self):
        """Empty text must not crash — name is prepended."""
        result = _swap_name("", "Emily Walsh")
        assert "Emily Walsh" in result

    def test_variant_a_and_b_differ_only_in_name(self):
        text = "John Doe\nPython Developer\njohn@example.com"
        var_a = _swap_name(text, "Emily Walsh")
        var_b = _swap_name(text, "Lakisha Jones")
        lines_a = var_a.split("\n")
        lines_b = var_b.split("\n")
        # First line differs
        assert lines_a[0] != lines_b[0]
        # All other lines identical
        assert lines_a[1:] == lines_b[1:]


# ===========================================================================
# _swap_university
# ===========================================================================

class TestSwapUniversity:

    def test_replaces_well_known_university(self):
        """University full name from universities.json is replaced."""
        text = "Education\nMassachusetts Institute of Technology\nB.Tech Computer Science"
        result = _swap_university(text, "Greenwood State University")
        assert "Greenwood State University" in result
        assert "Massachusetts Institute of Technology" not in result

    def test_replaces_case_insensitively(self):
        text = "Education\nstanford university\nB.S. Computer Science"
        result = _swap_university(text, "Western Plains College")
        assert "Western Plains College" in result

    def test_only_first_occurrence_replaced(self):
        text = "Harvard University graduate. Also applied to Harvard University."
        result = _swap_university(text, "Greenwood State University")
        # Second "Harvard University" must remain
        assert "Harvard University" in result
        assert result.count("Greenwood State University") == 1

    def test_no_known_university_appends_line(self):
        """If no university found, a University: line is appended."""
        text = "John Doe\nPython Developer\n3 years experience"
        result = _swap_university(text, "Stanford University")
        assert "Stanford University" in result
        # Original text preserved
        assert "Python Developer" in result

    def test_rest_of_text_unchanged_when_replaced(self):
        text = "Education\nHarvard University\nGPA: 3.9\nSkills: Python"
        result = _swap_university(text, "Northern Plains University")
        assert "GPA: 3.9" in result
        assert "Skills: Python" in result


# ===========================================================================
# build_variants
# ===========================================================================

class TestBuildVariants:

    def test_returns_tuple_of_variants_and_labels(self):
        text = "Emily Smith\nPython Developer"
        variants, label_a, label_b = build_variants(text, "name", n_pairs=2)
        assert isinstance(variants, list)
        assert isinstance(label_a, str)
        assert isinstance(label_b, str)

    def test_correct_number_of_pairs(self):
        text = "John Doe\nPython Developer"
        variants, _, _ = build_variants(text, "name", n_pairs=3)
        assert len(variants) == 3

    def test_each_variant_has_required_keys(self):
        text = "John Doe\nPython Developer"
        variants, _, _ = build_variants(text, "name", n_pairs=2)
        for v in variants:
            assert "identity_a" in v
            assert "identity_b" in v
            assert "text_a" in v
            assert "text_b" in v

    def test_text_a_and_b_differ(self):
        """The two variants must not be identical."""
        text = "John Doe\nPython Developer"
        variants, _, _ = build_variants(text, "name", n_pairs=1)
        assert variants[0]["text_a"] != variants[0]["text_b"]

    def test_name_labels_correct(self):
        text = "John Doe\nPython Developer"
        _, label_a, label_b = build_variants(text, "name", n_pairs=1)
        assert "american" in label_a.lower() or "white" in label_a.lower()
        assert "american" in label_b.lower() or "black" in label_b.lower()

    def test_university_labels_correct(self):
        text = "John Doe\nMIT\nPython Developer"
        _, label_a, label_b = build_variants(text, "university", n_pairs=1)
        assert "well" in label_a.lower() or "known" in label_a.lower()
        assert "lesser" in label_b.lower() or "known" in label_b.lower()

    def test_invalid_attribute_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported attribute"):
            build_variants("text", "gender", n_pairs=1)

    def test_n_pairs_capped_by_data(self):
        """Requesting more pairs than available data should not crash."""
        text = "John Doe\nPython Developer"
        variants, _, _ = build_variants(text, "name", n_pairs=999)
        # Should return at most as many pairs as we have names
        assert len(variants) <= 20  # our JSON has 20 names per group


# ===========================================================================
# run_audit  (pipeline mocked — no ML model loaded)
# ===========================================================================

class TestRunAudit:
    """
    We mock _score_variant so the ML model is never loaded.
    Tests verify the statistical aggregation and output structure.
    """

    def _mock_scores(self, scores_a: List[float], scores_b: List[float]):
        """Return a side_effect list alternating score_a, score_b."""
        interleaved = []
        for a, b in zip(scores_a, scores_b):
            interleaved.extend([a, b])
        return interleaved

    def test_returns_all_expected_keys(self):
        with patch("backend.app.services.bias._score_variant", return_value=70.0):
            result = run_audit(["John Doe\nPython"], "JD text", "name", n_pairs=2)
        expected = {
            "attribute", "group_a_label", "group_b_label",
            "mode",
            "mean_delta", "std_delta", "p_value", "significant",
            "n_observations", "raw_rows", "disclaimer",
        }
        assert expected.issubset(result.keys())

    def test_delta_is_score_b_minus_score_a(self):
        """delta = score_B - score_A for each pair."""
        # Alternate: score_a=70, score_b=72, score_a=70, score_b=72
        scores = [70.0, 72.0, 70.0, 72.0]
        with patch("backend.app.services.bias._score_variant", side_effect=scores):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=2)
        for row in result["raw_rows"]:
            assert abs(row["delta"] - (row["score_b"] - row["score_a"])) < 1e-6

    def test_mean_delta_calculated_correctly(self):
        """mean_delta = mean of all deltas."""
        # score_a=65, score_b=67 → delta=2 for all 3 pairs
        scores = [65.0, 67.0] * 3
        with patch("backend.app.services.bias._score_variant", side_effect=scores):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=3)
        assert abs(result["mean_delta"] - 2.0) < 0.01

    def test_zero_delta_when_scores_identical(self):
        """If pipeline produces identical scores for A and B, delta = 0."""
        with patch("backend.app.services.bias._score_variant", return_value=72.5):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=4)
        assert abs(result["mean_delta"]) < 1e-6
        assert result["significant"] is False

    def test_n_observations_equals_pairs(self):
        with patch("backend.app.services.bias._score_variant", return_value=70.0):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=5)
        assert result["n_observations"] == 5

    def test_multiple_resumes_multiply_observations(self):
        with patch("backend.app.services.bias._score_variant", return_value=70.0):
            result = run_audit(
                ["Resume One\nPython", "Resume Two\nJava"],
                "JD", "name", n_pairs=3,
            )
        assert result["n_observations"] == 6   # 2 resumes × 3 pairs

    def test_p_value_nan_when_only_one_observation(self):
        """t-test requires ≥ 2 observations; with 1 pair p_value is NaN."""
        with patch("backend.app.services.bias._score_variant", return_value=70.0):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=1)
        assert math.isnan(result["p_value"])

    def test_disclaimer_is_present_and_non_empty(self):
        with patch("backend.app.services.bias._score_variant", return_value=70.0):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=2)
        assert isinstance(result["disclaimer"], str)
        assert len(result["disclaimer"]) > 50
        assert "proof" in result["disclaimer"].lower()

    def test_raw_rows_have_correct_keys(self):
        with patch("backend.app.services.bias._score_variant", return_value=70.0):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=2)
        for row in result["raw_rows"]:
            for key in ("resume_index", "identity_a", "identity_b",
                        "score_a", "score_b", "delta"):
                assert key in row, f"Missing key '{key}' in row: {row}"

    def test_university_audit_runs_without_error(self):
        with patch("backend.app.services.bias._score_variant", return_value=68.0):
            result = run_audit(["John Doe\nMIT\nPython"], "JD", "university", n_pairs=3)
        assert result["attribute"] == "university"
        assert result["n_observations"] == 3


# ===========================================================================
# TestAuditModes — verify raw vs redacted mode behaviour
# ===========================================================================

class TestAuditModes:
    """
    Tests for the mode parameter added to run_audit().

    Key invariants we verify:
      1. Default mode is "redacted".
      2. Result dict always contains "mode" key with the correct value.
      3. Redacted mode produces delta = 0 for name swaps (strip_pii removes name).
      4. Raw mode CAN produce a non-zero delta when the scorer returns different
         values for the two variants (mocked to simulate embedding sensitivity).
    """

    def test_default_mode_is_redacted(self):
        """run_audit() without mode kwarg must default to 'redacted'."""
        with patch("backend.app.services.bias._score_variant", return_value=70.0):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=2)
        assert result["mode"] == "redacted"

    def test_mode_field_present_in_result_redacted(self):
        """'mode' key must be in the returned dict (redacted mode)."""
        with patch("backend.app.services.bias._score_variant", return_value=70.0):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=2, mode="redacted")
        assert "mode" in result
        assert result["mode"] == "redacted"

    def test_mode_field_present_in_result_raw(self):
        """'mode' key must be in the returned dict (raw mode)."""
        with patch("backend.app.services.bias._score_variant_raw", return_value=70.0):
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=2, mode="raw")
        assert "mode" in result
        assert result["mode"] == "raw"

    def test_redacted_name_gives_zero_delta(self):
        """
        In redacted mode the production pipeline runs strip_pii, which removes
        the candidate name before encoding. Swapping names therefore produces
        identical embeddings and delta = 0 by construction.

        We verify this by mocking _score_variant to return a constant value
        (simulating the name-blind pipeline).
        """
        with patch("backend.app.services.bias._score_variant", return_value=72.5):
            result = run_audit(
                ["John Doe\nPython Developer"],
                "Senior Python Developer JD",
                attribute="name",
                n_pairs=5,
                mode="redacted",
            )
        assert abs(result["mean_delta"]) < 1e-6, (
            f"Redacted name audit should give delta ≈ 0, got {result['mean_delta']}"
        )
        assert result["significant"] is False

    def test_raw_mode_can_produce_nonzero_delta(self):
        """
        In raw mode strip_pii is SKIPPED, so the embedding model sees the
        inserted name. We mock _score_variant_raw to return different values
        for group A vs group B to prove the pipeline honours those differences.

        delta = score_B - score_A. We mock: score_A=70, score_B=73 → delta=3.
        """
        # Alternate: for each pair call scorer twice: once for text_a (→70) once for text_b (→73)
        scores = [70.0, 73.0] * 5  # 5 pairs
        with patch("backend.app.services.bias._score_variant_raw", side_effect=scores):
            result = run_audit(
                ["John Doe\nPython Developer"],
                "Senior Python Developer JD",
                attribute="name",
                n_pairs=5,
                mode="raw",
            )
        assert abs(result["mean_delta"] - 3.0) < 0.01, (
            f"Raw mode should propagate mocked delta of 3.0, got {result['mean_delta']}"
        )

    def test_redacted_uses_score_variant_not_raw(self):
        """Confirm that redacted mode calls _score_variant (not _score_variant_raw)."""
        with patch("backend.app.services.bias._score_variant", return_value=68.0) as mock_std, \
             patch("backend.app.services.bias._score_variant_raw", return_value=99.0) as mock_raw:
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=2, mode="redacted")
        mock_std.assert_called()
        mock_raw.assert_not_called()
        # Result should use the value from _score_variant (68.0)
        assert all(r["score_a"] == 68.0 and r["score_b"] == 68.0 for r in result["raw_rows"])

    def test_raw_mode_uses_score_variant_raw_not_redacted(self):
        """Confirm that raw mode calls _score_variant_raw (not _score_variant)."""
        with patch("backend.app.services.bias._score_variant", return_value=99.0) as mock_std, \
             patch("backend.app.services.bias._score_variant_raw", return_value=65.0) as mock_raw:
            result = run_audit(["John Doe\nPython"], "JD", "name", n_pairs=2, mode="raw")
        mock_raw.assert_called()
        mock_std.assert_not_called()
        assert all(r["score_a"] == 65.0 and r["score_b"] == 65.0 for r in result["raw_rows"])


# ===========================================================================
# TestBiasAuditRouter — test the router function and schema validation
# ===========================================================================

from pydantic import ValidationError
from backend.app.schemas import BiasAuditRequest
from backend.app.routers.bias import bias_audit


class TestBiasAuditRouter:

    def test_router_default_mode_is_redacted(self):
        req = BiasAuditRequest(
            original_text="John Doe\nPython developer\njohn@example.com",
            job_description="Looking for Python developer",
            attribute="name",
            n_pairs=2,
        )
        assert req.mode == "redacted"
        with patch("backend.app.routers.bias.run_audit") as mock_audit:
            mock_audit.return_value = {
                "attribute": "name",
                "group_a_label": "white_american",
                "group_b_label": "black_american",
                "mode": "redacted",
                "mean_delta": 0.0,
                "std_delta": 0.0,
                "p_value": None,
                "significant": False,
                "n_observations": 2,
                "raw_rows": [],
                "disclaimer": DISCLAIMER,
            }
            res = bias_audit(req)
        assert res.mode == "redacted"
        mock_audit.assert_called_once()
        assert mock_audit.call_args[1]["mode"] == "redacted"

    def test_router_raw_mode_passed_correctly(self):
        req = BiasAuditRequest(
            original_text="John Doe\nPython developer\njohn@example.com",
            job_description="Looking for Python developer",
            attribute="name",
            n_pairs=2,
            mode="raw",
        )
        assert req.mode == "raw"
        with patch("backend.app.routers.bias.run_audit") as mock_audit:
            mock_audit.return_value = {
                "attribute": "name",
                "group_a_label": "white_american",
                "group_b_label": "black_american",
                "mode": "raw",
                "mean_delta": 0.5,
                "std_delta": 0.2,
                "p_value": 0.03,
                "significant": True,
                "n_observations": 2,
                "raw_rows": [],
                "disclaimer": DISCLAIMER,
            }
            res = bias_audit(req)
        assert res.mode == "raw"
        assert mock_audit.call_args[1]["mode"] == "raw"

    def test_schema_rejects_invalid_mode(self):
        with pytest.raises(ValidationError):
            BiasAuditRequest(
                original_text="John Doe\nPython developer",
                job_description="Looking for Python developer",
                attribute="name",
                n_pairs=2,
                mode="invalid_mode",
            )
