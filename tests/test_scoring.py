"""
tests/test_scoring.py

Unit tests for services/scoring.py.

The critical test is test_final_score_equals_weighted_sum — it proves
the formula is correct without hardcoding 0.6/0.4 (reads from config).

Run with:
    pytest tests/test_scoring.py -v
"""

import pytest
from backend.app.config import settings
from backend.app.services.scoring import compute_final_score


class TestComputeFinalScore:

    def test_final_score_equals_weighted_sum(self):
        """
        Core invariant: final_score == semantic_weight * semantic + skill_weight * skill.
        Weights are read from settings — if config changes, the test still verifies
        the formula rather than a hardcoded number.
        """
        semantic   = 80.0
        skill_ovlp = 60.0
        result     = compute_final_score(semantic, skill_ovlp)

        expected = (
            settings.semantic_weight * semantic
            + settings.skill_weight  * skill_ovlp
        )
        assert abs(result["final_score"] - round(expected, 2)) < 1e-6

    def test_weights_sum_reflected_in_score(self):
        """
        When both components are equal, final_score must equal that common value
        regardless of the weight split (because w1 + w2 == 1.0).
        """
        result = compute_final_score(70.0, 70.0)
        assert abs(result["final_score"] - 70.0) < 1e-6

    def test_zero_inputs_give_zero(self):
        result = compute_final_score(0.0, 0.0)
        assert result["final_score"] == 0.0

    def test_full_inputs_give_100(self):
        result = compute_final_score(100.0, 100.0)
        assert result["final_score"] == 100.0

    def test_clamped_at_100_on_overflow(self):
        """Floating-point overshoot must be clamped to 100."""
        result = compute_final_score(100.0 + 1e-9, 100.0 + 1e-9)
        assert result["final_score"] <= 100.0

    def test_clamped_at_0_on_underflow(self):
        """Negative inputs (edge case) must not produce a negative score."""
        result = compute_final_score(-5.0, -5.0)
        assert result["final_score"] == 0.0

    def test_returns_all_expected_keys(self):
        result = compute_final_score(75.0, 50.0)
        assert "final_score"    in result
        assert "semantic_score" in result
        assert "skill_overlap"  in result
        assert "weights"        in result

    def test_weights_key_has_correct_subkeys(self):
        result = compute_final_score(75.0, 50.0)
        assert "semantic"      in result["weights"]
        assert "skill_overlap" in result["weights"]

    def test_weights_match_config(self):
        """Weights stored in the return value must equal the config values."""
        result = compute_final_score(60.0, 40.0)
        assert result["weights"]["semantic"]      == settings.semantic_weight
        assert result["weights"]["skill_overlap"] == settings.skill_weight

    def test_components_passed_through(self):
        """semantic_score and skill_overlap are echoed back rounded."""
        result = compute_final_score(78.3456, 61.2345)
        assert result["semantic_score"] == round(78.3456, 2)
        assert result["skill_overlap"]  == round(61.2345, 2)

    def test_asymmetric_weights_applied_correctly(self):
        """
        With default weights 0.6 / 0.4:
        final = 0.6 * 100 + 0.4 * 0 = 60
        final = 0.6 * 0   + 0.4 * 100 = 40
        These two must NOT be equal, confirming the weighting is asymmetric.
        """
        high_semantic_only  = compute_final_score(100.0, 0.0)["final_score"]
        high_skill_only     = compute_final_score(0.0, 100.0)["final_score"]
        # They should differ (unless weights happen to be 0.5/0.5)
        if abs(settings.semantic_weight - settings.skill_weight) > 1e-9:
            assert high_semantic_only != high_skill_only
