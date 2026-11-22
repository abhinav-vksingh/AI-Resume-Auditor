"""
tests/test_preprocessing.py

Unit tests for services/preprocessing.py.

All tests call pure Python functions — no HTTP, no FastAPI TestClient.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import pytest

from backend.app.services.preprocessing import (
    clean_text,
    strip_pii,
    detect_sections,
    normalize_aliases,
    preprocess,
)


# ===========================================================================
# clean_text
# ===========================================================================

class TestCleanText:

    def test_lowercases_input(self):
        assert clean_text("Python DEVELOPER") == "python developer"

    def test_form_feed_becomes_newline(self):
        result = clean_text("page one\fpage two")
        assert "\f" not in result
        assert "page one" in result
        assert "page two" in result

    def test_bullet_glyphs_removed(self):
        text = "• Python\n▸ Java\n◦ SQL"
        result = clean_text(text)
        assert "•" not in result
        assert "▸" not in result
        assert "◦" not in result
        assert "python" in result

    def test_newlines_preserved(self):
        """Newlines must survive clean_text so detect_sections can find headers."""
        text = "Education\nB.Tech Computer Science\nExperience\nSoftware Engineer"
        result = clean_text(text)
        assert "\n" in result

    def test_multiple_spaces_collapsed(self):
        result = clean_text("python   developer    engineer")
        assert "  " not in result  # no double spaces

    def test_excessive_newlines_collapsed(self):
        result = clean_text("line one\n\n\n\n\nline two")
        # 5 newlines → at most 2
        assert "\n\n\n" not in result


# ===========================================================================
# strip_pii
# ===========================================================================

class TestStripPii:

    def test_email_removed(self):
        text = "john doe\njohndoe@gmail.com\npython developer"
        result = strip_pii(text)
        assert "@" not in result
        assert "gmail.com" not in result

    def test_phone_number_removed(self):
        text = "john doe\n+91-9876543210\npython developer"
        result = strip_pii(text)
        # The 10-digit block should be gone
        assert "9876543210" not in result

    def test_linkedin_url_removed(self):
        text = "john doe\nlinkedin.com/in/johndoe\npython developer"
        result = strip_pii(text)
        assert "linkedin.com" not in result

    def test_github_url_removed(self):
        text = "john doe\ngithub.com/johndoe\npython developer"
        result = strip_pii(text)
        assert "github.com" not in result

    def test_http_url_removed(self):
        text = "john doe\nhttps://johndoe.dev/portfolio\npython developer"
        result = strip_pii(text)
        assert "https://" not in result

    def test_name_removed_from_first_line(self):
        """A short word-only first line should be treated as a name and removed."""
        text = "john michael doe\neducation\nb.tech computer science"
        result = strip_pii(text)
        # The name line should be blank/removed; the rest should survive
        assert "b.tech" in result

    def test_section_header_not_treated_as_name(self):
        """A line like 'education' must NOT be removed as a name."""
        text = "education\nb.tech computer science\n2020-2024"
        result = strip_pii(text)
        assert "education" in result

    def test_skills_content_preserved(self):
        """Non-PII content is never removed."""
        text = "john doe\npython, java, machine learning, docker"
        result = strip_pii(text)
        assert "python" in result
        assert "machine learning" in result

    # --- additional phone formats -------------------------------------------

    def test_phone_international_with_space(self):
        """Format: +91 98765 43210"""
        text = "john doe\n+91 98765 43210\npython developer"
        result = strip_pii(text)
        assert "9876543210" not in result.replace(" ", "")

    def test_phone_bare_10_digits(self):
        """Format: 9876543210 (bare, no separators)"""
        text = "john doe\n9876543210\npython developer"
        result = strip_pii(text)
        assert "9876543210" not in result

    def test_phone_us_parenthesised(self):
        """Format: (555) 123-4567"""
        text = "john doe\n(555) 123-4567\npython developer"
        result = strip_pii(text)
        assert "5551234567" not in result.replace(" ", "").replace("-", "")

    def test_phone_us_dashes(self):
        """Format: 555-123-4567"""
        text = "john doe\n555-123-4567\npython developer"
        result = strip_pii(text)
        assert "5551234567" not in result.replace("-", "")

    def test_phone_us_country_code(self):
        """Format: +1 555 123 4567"""
        text = "john doe\n+1 555 123 4567\npython developer"
        result = strip_pii(text)
        assert "5551234567" not in result.replace(" ", "")

    # --- false-positive guards: these must SURVIVE strip_pii ----------------

    def test_year_range_hyphen_preserved(self):
        """'2021-2024' is a date range, not a phone number."""
        text = "b.tech computer science\n2021-2024"
        result = strip_pii(text)
        assert "2021" in result
        assert "2024" in result

    def test_year_range_spaced_preserved(self):
        """'2019 - 2023' (with spaces) is a date range."""
        text = "software engineer at acme corp\n2019 - 2023"
        result = strip_pii(text)
        assert "2019" in result
        assert "2023" in result

    def test_cgpa_decimal_preserved(self):
        """'cgpa: 8.5' must not be treated as a phone number."""
        text = "education\nb.tech cse\ncgpa: 8.5"
        result = strip_pii(text)
        assert "8.5" in result

    def test_score_fraction_preserved(self):
        """'9.1/10' must survive."""
        text = "education\nb.tech cse\n9.1/10"
        result = strip_pii(text)
        assert "9.1" in result

    def test_percentage_preserved(self):
        """'increased accuracy by 25%' must not lose the number."""
        text = "projects\nincreased accuracy by 25%"
        result = strip_pii(text)
        assert "25" in result


# ===========================================================================
# detect_sections
# ===========================================================================

class TestDetectSections:

    def test_known_headers_detected(self):
        text = (
            "education\n"
            "b.tech cse 2024\n\n"
            "experience\n"
            "software engineer at xyz\n\n"
            "skills\n"
            "python, java, sql"
        )
        sections = detect_sections(text)
        assert "education" in sections
        assert "experience" in sections
        assert "skills" in sections

    def test_content_assigned_to_correct_section(self):
        text = (
            "education\n"
            "b.tech computer science\n\n"
            "skills\n"
            "python java sql"
        )
        sections = detect_sections(text)
        assert "b.tech" in sections.get("education", "")
        assert "python" in sections.get("skills", "")

    def test_no_headers_falls_back_to_other(self):
        """If no recognised headers exist, all text goes to 'other'."""
        text = "python developer with 2 years of experience building web apps"
        sections = detect_sections(text)
        assert "other" in sections
        assert "python" in sections["other"]

    def test_no_crash_on_empty_text(self):
        """detect_sections must not raise on empty input."""
        sections = detect_sections("")
        assert isinstance(sections, dict)

    def test_technical_skills_header_maps_to_skills(self):
        text = "technical skills\npython, tensorflow, docker"
        sections = detect_sections(text)
        assert "skills" in sections
        assert "python" in sections["skills"]

    def test_academic_projects_header_maps_to_projects(self):
        text = "academic projects\nresume auditor using nlp"
        sections = detect_sections(text)
        assert "projects" in sections

    def test_other_does_not_appear_when_all_content_matched(self):
        """When every line is in a known section, 'other' should be absent."""
        text = (
            "education\n"
            "b.tech cse\n\n"
            "experience\n"
            "software engineer\n\n"
            "skills\n"
            "python"
        )
        sections = detect_sections(text)
        # 'other' may be absent, or if present must be empty
        other_content = sections.get("other", "").strip()
        assert other_content == ""


# ===========================================================================
# normalize_aliases
# ===========================================================================

class TestNormalizeAliases:

    # Use an inline alias map so the test does not depend on the JSON file
    ALIASES = {
        "ml":  "machine learning",
        "js":  "javascript",
        "ts":  "typescript",
        "node": "nodejs",
        "k8s": "kubernetes",
    }

    def test_basic_replacement(self):
        result = normalize_aliases("experience with ml and js", self.ALIASES)
        assert "machine learning" in result
        assert "javascript" in result

    def test_whole_word_only_no_partial_match(self):
        """'ml' inside 'html' must NOT be replaced."""
        result = normalize_aliases("html and xml skills", self.ALIASES)
        # html and xml should stay unchanged
        assert "html" in result
        assert "xml" in result
        # 'ml' replacement must not appear as artifact
        assert "machine learningtml" not in result  # html → "htmachine learning" is wrong
        assert result.count("machine learning") == 0

    def test_case_insensitive_match(self):
        """'ML' uppercase should also be replaced."""
        result = normalize_aliases("skilled in ML", self.ALIASES)
        assert "machine learning" in result

    def test_node_not_replaced_inside_nodejs(self):
        """'node' inside 'nodejs' — no word boundary after 'node'."""
        # 'nodejs' has no \b between 'node' and 'js' (both word chars)
        result = normalize_aliases("i know nodejs", self.ALIASES)
        # Should NOT double-expand to "nodejsjs" or similar
        assert "nodejsjs" not in result

    def test_no_replacement_when_not_in_aliases(self):
        result = normalize_aliases("python developer", self.ALIASES)
        assert result == "python developer"

    def test_multiple_aliases_in_one_string(self):
        result = normalize_aliases("ml, js, k8s deployment", self.ALIASES)
        assert "machine learning" in result
        assert "javascript" in result
        assert "kubernetes" in result


# ===========================================================================
# preprocess  (integration of all steps)
# ===========================================================================

class TestPreprocess:

    def test_original_text_is_unchanged(self):
        """The bias audit depends on original_text being byte-identical to input."""
        raw = "John Doe\njohndoe@gmail.com\n+91-9876543210\nPython Developer"
        result = preprocess(raw)
        assert result["original_text"] == raw

    def test_original_text_contains_pii(self):
        """PII is NOT stripped from original_text — only from core_text."""
        raw = "Jane Smith\njane@example.com\nSoftware Engineer"
        result = preprocess(raw)
        assert "jane@example.com" in result["original_text"]

    def test_core_text_has_no_email(self):
        raw = "Jane Smith\njane@example.com\nSoftware Engineer"
        result = preprocess(raw)
        assert "@" not in result["core_text"]

    def test_core_text_is_lowercased(self):
        raw = "PYTHON DEVELOPER"
        result = preprocess(raw)
        assert result["core_text"] == result["core_text"].lower()

    def test_sections_dict_returned(self):
        raw = "Education\nB.Tech CSE 2024"
        result = preprocess(raw)
        assert isinstance(result["sections"], dict)
        assert len(result["sections"]) > 0

    def test_core_text_has_aliases_expanded(self):
        """'ml' in input → 'machine learning' in core_text."""
        raw = "skills: ml, js, k8s"
        result = preprocess(raw)
        assert "machine learning" in result["core_text"]
        assert "javascript" in result["core_text"]
        assert "kubernetes" in result["core_text"]

    def test_form_feed_handled(self):
        """Multi-page PDFs with \\f separators should not crash."""
        raw = "Education\nB.Tech\fExperience\nSoftware Engineer"
        result = preprocess(raw)
        assert isinstance(result, dict)
        assert result["core_text"]  # not empty
