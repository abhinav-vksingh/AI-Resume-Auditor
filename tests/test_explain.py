"""
tests/test_explain.py

Unit tests for services/explain.py.

Every test verifies a specific coded rule — if the rule changes, only
the test for that rule breaks (not the whole suite).

Run with:
    pytest tests/test_explain.py -v
"""

import pytest
from backend.app.services.explain import generate_explanation, _MAX_MISSING_SHOWN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table(*rows):
    """Build a minimal skill_table from (skill, status) tuples."""
    return [{"skill": s, "status": st} for s, st in rows]


def _explain(**kwargs):
    """Call generate_explanation with safe defaults; override with kwargs."""
    defaults = dict(
        semantic_score=70.0,
        skill_overlap=70.0,
        final_score=70.0,
        skill_table=[],
    )
    defaults.update(kwargs)
    return generate_explanation(**defaults)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_returns_dict_with_correct_keys(self):
        result = _explain()
        assert "positive_factors" in result
        assert "negative_factors" in result

    def test_factors_are_lists(self):
        result = _explain()
        assert isinstance(result["positive_factors"], list)
        assert isinstance(result["negative_factors"], list)

    def test_all_factors_are_strings(self):
        table = _table(("python", "Matched"), ("kubernetes", "Missing"))
        result = _explain(skill_table=table)
        for f in result["positive_factors"] + result["negative_factors"]:
            assert isinstance(f, str), f"Non-string factor: {f!r}"

    def test_empty_skill_table_no_crash(self):
        result = _explain(skill_table=[])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Positive factor rules
# ---------------------------------------------------------------------------

class TestPositiveFactors:

    def test_high_final_score_generates_positive(self):
        """P1: final_score >= 75 → 'Overall strong candidate match …'"""
        result = _explain(final_score=80.0)
        assert any("strong candidate match" in f for f in result["positive_factors"])

    def test_low_final_score_no_strong_match_message(self):
        """P1 must NOT fire when final_score < 75."""
        result = _explain(final_score=60.0)
        assert not any("strong candidate match" in f for f in result["positive_factors"])

    def test_high_semantic_generates_strong_alignment(self):
        """P2: semantic_score >= 75 → 'Strong semantic alignment …'"""
        result = _explain(semantic_score=80.0)
        assert any("Strong semantic alignment" in f for f in result["positive_factors"])

    def test_moderate_semantic_generates_moderate_alignment(self):
        """P2b: 50 <= semantic_score < 75 → 'Moderate semantic alignment …'"""
        result = _explain(semantic_score=60.0)
        assert any("Moderate semantic alignment" in f for f in result["positive_factors"])

    def test_low_semantic_generates_no_alignment_positive(self):
        """semantic_score < 50 → neither 'Strong' nor 'Moderate' alignment positive."""
        result = _explain(semantic_score=40.0)
        positives = result["positive_factors"]
        assert not any("Strong semantic" in f or "Moderate semantic" in f for f in positives)

    def test_matched_skills_listed_in_positive(self):
        """P3: at least one Matched skill → positive factor lists skill name."""
        table = _table(("python", "Matched"), ("docker", "Missing"))
        result = _explain(skill_table=table)
        assert any("python" in f for f in result["positive_factors"])

    def test_no_matched_skills_no_p3_factor(self):
        """P3 must not appear when no skill is Matched."""
        table = _table(("docker", "Missing"), ("kubernetes", "Missing"))
        result = _explain(skill_table=table)
        assert not any("directly evidenced" in f for f in result["positive_factors"])

    def test_high_skill_overlap_generates_good_coverage(self):
        """P4: skill_overlap >= 70 → 'Good skill coverage …'"""
        result = _explain(skill_overlap=75.0, skill_table=_table(("python", "Matched")))
        assert any("Good skill coverage" in f for f in result["positive_factors"])

    def test_low_skill_overlap_no_good_coverage_positive(self):
        """P4 must not fire when skill_overlap < 70."""
        result = _explain(skill_overlap=50.0)
        assert not any("Good skill coverage" in f for f in result["positive_factors"])


# ---------------------------------------------------------------------------
# Negative factor rules
# ---------------------------------------------------------------------------

class TestNegativeFactors:

    def test_low_semantic_generates_negative(self):
        """N1: semantic_score < 50 → negative factor about low similarity."""
        result = _explain(semantic_score=40.0)
        assert any("Low semantic similarity" in f for f in result["negative_factors"])

    def test_high_semantic_no_low_similarity_negative(self):
        """N1 must not fire when semantic_score >= 50."""
        result = _explain(semantic_score=65.0)
        assert not any("Low semantic similarity" in f for f in result["negative_factors"])

    def test_low_skill_overlap_generates_negative(self):
        """N2: skill_overlap < 40 → negative factor about low coverage."""
        table = _table(("python", "Missing"))
        result = _explain(skill_overlap=30.0, skill_table=table)
        assert any("low" in f.lower() and "skill" in f.lower() for f in result["negative_factors"])

    def test_missing_skill_message_exact_format(self):
        """
        N3: missing skill must produce the EXACT format from the spec:
        '"{skill}" is required by the JD but no evidence was found in the resume.'
        """
        table = _table(("docker", "Missing"))
        result = _explain(skill_table=table)
        expected_fragment = '"docker" is required by the JD but no evidence was found in the resume.'
        assert any(expected_fragment in f for f in result["negative_factors"])

    def test_all_missing_skills_up_to_cap_appear(self):
        """N3: up to _MAX_MISSING_SHOWN individual lines are generated."""
        rows = [(f"skill{i}", "Missing") for i in range(_MAX_MISSING_SHOWN)]
        table = _table(*rows)
        result = _explain(skill_table=table)
        # Every missing skill should have its own line
        for i in range(_MAX_MISSING_SHOWN):
            assert any(f"skill{i}" in f for f in result["negative_factors"])

    def test_overflow_missing_skills_collapsed_to_summary(self):
        """N3: when missing > cap, a summary '… and N more …' line appears."""
        rows = [(f"skill{i}", "Missing") for i in range(_MAX_MISSING_SHOWN + 3)]
        table = _table(*rows)
        result = _explain(skill_table=table)
        assert any("more" in f and "absent" in f for f in result["negative_factors"])

    def test_no_missing_skills_no_missing_negative(self):
        """N3 produces zero lines when all skills are Matched."""
        table = _table(("python", "Matched"), ("docker", "Matched"))
        result = _explain(skill_table=table)
        assert not any("no evidence was found" in f for f in result["negative_factors"])

    def test_weak_skill_generates_negative_factor(self):
        """N4: each Weak skill gets its own negative line."""
        table = _table(("sql", "Weak"))
        result = _explain(skill_table=table)
        assert any("sql" in f and "once" in f for f in result["negative_factors"])

    def test_no_weak_skills_no_weak_negative(self):
        """N4 must produce zero lines when no skill is Weak."""
        table = _table(("python", "Matched"))
        result = _explain(skill_table=table)
        assert not any("only once" in f for f in result["negative_factors"])

    def test_multiple_missing_skills_each_has_own_line(self):
        """N3: three missing skills → three individual negative lines."""
        table = _table(
            ("docker", "Missing"),
            ("kubernetes", "Missing"),
            ("aws", "Missing"),
        )
        result = _explain(skill_table=table)
        lines = result["negative_factors"]
        assert any("docker" in f for f in lines)
        assert any("kubernetes" in f for f in lines)
        assert any("aws" in f for f in lines)


# ---------------------------------------------------------------------------
# Integration: mixed skill table
# ---------------------------------------------------------------------------

class TestMixedExplanation:

    def test_mixed_table_all_categories_represented(self):
        """A realistic table produces both positive and negative factors."""
        table = _table(
            ("python",     "Matched"),
            ("docker",     "Matched"),
            ("kubernetes", "Missing"),
            ("sql",        "Weak"),
        )
        result = _explain(
            semantic_score=72.0,
            skill_overlap=55.0,
            final_score=64.8,
            skill_table=table,
        )
        positives = result["positive_factors"]
        negatives = result["negative_factors"]

        # Matched skills → positive
        assert any("python" in f or "docker" in f for f in positives)
        # Missing skill → negative
        assert any("kubernetes" in f for f in negatives)
        # Weak skill → negative
        assert any("sql" in f for f in negatives)
