"""
tests/test_skills.py

Unit tests for services/skills.py.

Tests are pure Python — no HTTP layer, no model loading.
Skill classification uses only regex + dict lookups, so tests are fast.

Run with:
    pytest tests/test_skills.py -v
"""

import pytest

from backend.app.services.skills import (
    extract_required_skills,
    classify_skill,
    build_skill_table,
    compute_skill_overlap,
    run_skill_analysis,
)


# ===========================================================================
# extract_required_skills
# ===========================================================================

class TestExtractRequiredSkills:

    def test_finds_exact_skill_in_jd(self):
        """A skill present verbatim in the JD should be returned."""
        required = extract_required_skills("We need a Python developer.")
        assert "python" in required

    def test_alias_normalised_before_matching(self):
        """JD says 'ml' → aliases expand it → matches 'machine learning' skill."""
        required = extract_required_skills("Experience in ml and dl required.")
        assert "machine learning" in required
        assert "deep learning" in required

    def test_skill_not_in_jd_is_absent(self):
        required = extract_required_skills("Looking for a recruitment specialist.")
        assert "python" not in required

    def test_whole_word_only_no_partial(self):
        """'ml' inside 'html' must NOT trigger a match for 'machine learning'."""
        required = extract_required_skills("Must know html and xml.")
        # 'machine learning' should not appear just because 'ml' is in 'html'
        assert "machine learning" not in required

    def test_returns_list(self):
        result = extract_required_skills("We need python, docker and kubernetes.")
        assert isinstance(result, list)

    def test_empty_jd_returns_empty_list(self):
        result = extract_required_skills("")
        assert result == []

    def test_no_duplicates(self):
        """Skill appearing multiple times in JD should appear once in result."""
        result = extract_required_skills("python python python developer with python skills")
        assert result.count("python") == 1

    def test_multi_word_skill_detected(self):
        result = extract_required_skills("Experience in machine learning required.")
        assert "machine learning" in result


# ===========================================================================
# classify_skill
# ===========================================================================

class TestClassifySkill:

    def test_matched_when_in_skills_section(self):
        """Skill in the 'skills' section → Matched."""
        sections = {"skills": "python, java, docker", "experience": "built web apps"}
        status = classify_skill("python", "python, java, docker built web apps", sections)
        assert status == "Matched"

    def test_matched_when_in_experience_section(self):
        """Skill found only in 'experience' section → Matched."""
        sections = {"experience": "used docker to containerise services"}
        status = classify_skill("docker", "used docker to containerise services", sections)
        assert status == "Matched"

    def test_matched_when_in_projects_section(self):
        """Skill found only in 'projects' section → Matched."""
        sections = {"projects": "deployed kubernetes cluster on aws"}
        status = classify_skill("kubernetes", "deployed kubernetes cluster on aws", sections)
        assert status == "Matched"

    def test_matched_when_mentioned_multiple_times(self):
        """Skill mentioned >1 time anywhere → Matched (even if not in primary section)."""
        sections = {"summary": "python enthusiast", "other": "python experience"}
        core = "python enthusiast python experience"
        status = classify_skill("python", core, sections)
        assert status == "Matched"

    def test_weak_when_mentioned_once_not_in_primary(self):
        """Skill in 'summary' only, mentioned once → Weak."""
        sections = {"summary": "familiar with docker"}
        status = classify_skill("docker", "familiar with docker", sections)
        assert status == "Weak"

    def test_missing_when_not_found(self):
        """Skill absent from resume → Missing."""
        sections = {"skills": "python, java", "experience": "built apis"}
        status = classify_skill("kubernetes", "python, java built apis", sections)
        assert status == "Missing"

    def test_no_partial_match(self):
        """'node' pattern must not fire inside 'nodejs'."""
        sections = {"skills": "nodejs, express"}
        status = classify_skill("node", "nodejs, express", sections)
        assert status == "Missing"

    def test_case_insensitive_match(self):
        """Classification is case-insensitive."""
        sections = {"skills": "Python, Java"}
        status = classify_skill("python", "Python, Java", sections)
        assert status == "Matched"


# ===========================================================================
# compute_skill_overlap
# ===========================================================================

class TestComputeSkillOverlap:

    def test_all_matched_gives_100(self):
        assert compute_skill_overlap(matched=5, weak=0, total=5) == 100.0

    def test_all_missing_gives_0(self):
        assert compute_skill_overlap(matched=0, weak=0, total=5) == 0.0

    def test_zero_total_gives_0(self):
        """No skills required → score is 0, no division-by-zero."""
        assert compute_skill_overlap(matched=0, weak=0, total=0) == 0.0

    def test_weak_counts_half(self):
        """3 matched + 2 weak out of 10 → (3 + 1) / 10 × 100 = 40.0"""
        result = compute_skill_overlap(matched=3, weak=2, total=10)
        assert abs(result - 40.0) < 0.01

    def test_mixed_formula(self):
        """3 matched + 2 weak out of 6 total → (3 + 1) / 6 × 100 ≈ 66.67"""
        result = compute_skill_overlap(matched=3, weak=2, total=6)
        assert abs(result - 66.67) < 0.01

    def test_only_weak_gives_50(self):
        """0 matched + 2 weak out of 2 total → (0 + 1) / 2 × 100 = 50.0"""
        result = compute_skill_overlap(matched=0, weak=2, total=2)
        assert abs(result - 50.0) < 0.01

    def test_result_between_0_and_100(self):
        result = compute_skill_overlap(matched=7, weak=3, total=10)
        assert 0.0 <= result <= 100.0


# ===========================================================================
# build_skill_table
# ===========================================================================

class TestBuildSkillTable:

    def test_returns_list_of_dicts(self):
        required = ["python", "kubernetes"]
        sections = {"skills": "python"}
        table = build_skill_table(required, "python", sections)
        assert isinstance(table, list)
        assert all(isinstance(r, dict) for r in table)
        assert all("skill" in r and "status" in r for r in table)

    def test_one_entry_per_skill(self):
        required = ["python", "java", "docker"]
        sections = {"skills": "python java"}
        table = build_skill_table(required, "python java", sections)
        assert len(table) == 3

    def test_statuses_are_valid(self):
        required = ["python", "kubernetes"]
        table = build_skill_table(required, "python developer", {"skills": "python"})
        for row in table:
            assert row["status"] in {"Matched", "Weak", "Missing"}

    def test_empty_required_gives_empty_table(self):
        table = build_skill_table([], "python developer", {})
        assert table == []


# ===========================================================================
# run_skill_analysis  (integration)
# ===========================================================================

class TestRunSkillAnalysis:

    def test_returns_expected_keys(self):
        result = run_skill_analysis(
            job_description="Python developer with machine learning experience.",
            core_text="python machine learning scikit-learn",
            sections={"skills": "python machine learning scikit-learn"},
        )
        assert "skill_table" in result
        assert "skill_overlap" in result
        assert "matched_count" in result
        assert "weak_count" in result
        assert "missing_count" in result
        assert "total_required" in result

    def test_counts_sum_to_total(self):
        result = run_skill_analysis(
            job_description="Python and docker and kubernetes required.",
            core_text="python experience building apps",
            sections={"skills": "python"},
        )
        assert (
            result["matched_count"] + result["weak_count"] + result["missing_count"]
            == result["total_required"]
        )

    def test_skill_overlap_in_range(self):
        result = run_skill_analysis(
            job_description="Python developer with SQL experience.",
            core_text="python sql postgresql",
            sections={"skills": "python sql postgresql"},
        )
        assert 0.0 <= result["skill_overlap"] <= 100.0


# ===========================================================================
# Newly added skills and next.js fix (Step 2 of Task B)
# ===========================================================================

class TestNewSkillsAndNextjsFix:
    """Verify that next.js, celery, and rabbitmq are correctly extracted from JD text.

    next.js was already in skills.json but alias normalisation expanded the
    '.js' suffix to '.javascript', breaking the match.  The fix makes
    extract_required_skills check both the cleaned and the alias-normalised
    text, so punctuated skill names are never lost.
    """

    def test_nextjs_extracted_from_jd(self):
        """'next.js' should be found even though alias normalisation corrupts '.js'."""
        skills = extract_required_skills(
            "We need a frontend developer with Next.js and React experience."
        )
        assert "next.js" in skills, (
            "next.js was not extracted — check the dual-text fix in extract_required_skills"
        )

    def test_celery_extracted_from_jd(self):
        """'celery' (newly added to skills.json) should be found in a JD."""
        skills = extract_required_skills(
            "Experience with Celery for distributed task queues is required."
        )
        assert "celery" in skills, "celery was not extracted — check skills.json"

    def test_rabbitmq_extracted_from_jd(self):
        """'rabbitmq' (newly added) should be found; alias 'rabbit mq' also maps to it."""
        skills = extract_required_skills(
            "Must have hands-on experience with RabbitMQ message broker."
        )
        assert "rabbitmq" in skills, "rabbitmq was not extracted — check skills.json"
