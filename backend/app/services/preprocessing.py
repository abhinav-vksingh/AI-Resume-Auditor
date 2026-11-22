"""
services/preprocessing.py – All text-cleaning steps before scoring.

Pipeline order (called by preprocess()):
  1. clean_text        – lowercase, kill artifacts, preserve newlines
  2. strip_pii         – remove email / phone / URLs / candidate name
  3. detect_sections   – split into labelled buckets
  4. normalize_aliases – expand abbreviations (ml → machine learning)

Design rule (from spec):
  preprocess() MUST return the original untouched text alongside the cleaned
  copy.  The bias-audit phase will inject names into the original, so it
  must never be mutated here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

# ---------------------------------------------------------------------------
# Path to the aliases dictionary shipped with the backend
# ---------------------------------------------------------------------------

_ALIASES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "aliases.json"


# ---------------------------------------------------------------------------
# 1. clean_text
# ---------------------------------------------------------------------------

# Characters we keep: letters, digits, common punctuation, whitespace
# We also keep  /  +  #  @  .  for tech tokens like C++, C#, .NET, @decorators
_KEEP_PATTERN = re.compile(r"[^\w\s\n\r.,;:()\-+#/@&']")

# Multiple SPACES on the same line (not newlines)
_MULTI_SPACE = re.compile(r"[^\S\n]+")

# Three or more consecutive newlines → two (paragraph break)
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Common PDF bullet / list artifacts that add no information
_BULLET_ARTIFACTS = re.compile(r"[•▸◦➢➤●◉▪▫►◄→←★☆✓✔✗✘]")


def clean_text(text: str) -> str:
    """Return a lowercased, artifact-free version of *text*.

    Newlines are PRESERVED so that detect_sections() can find headers.
    Page-break form-feeds (\\f) are treated as paragraph separators (→ \\n\\n).

    Steps
    -----
    1. Replace form-feeds with double newline (preserve paragraph structure).
    2. Replace bullet glyphs with a space (they add no semantic content).
    3. Lowercase.
    4. Strip characters that are not letters/digits/punctuation/whitespace.
    5. Collapse multiple spaces on the same line.
    6. Collapse 3+ consecutive newlines to 2.
    7. Strip leading/trailing whitespace.
    """
    # Step 1: form-feed → paragraph break
    text = text.replace("\f", "\n\n")

    # Step 2: remove bullet glyphs (replace with space to avoid word merging)
    text = _BULLET_ARTIFACTS.sub(" ", text)

    # Step 3: lowercase
    text = text.lower()

    # Step 4: drop characters we don't want (keeps letters, digits, punctuation)
    text = _KEEP_PATTERN.sub(" ", text)

    # Step 5: collapse runs of spaces (but NOT newlines) on each line
    text = _MULTI_SPACE.sub(" ", text)

    # Step 6: collapse excess blank lines
    text = _MULTI_NEWLINE.sub("\n\n", text)

    # Step 7: strip
    return text.strip()


# ---------------------------------------------------------------------------
# 2. strip_pii
# ---------------------------------------------------------------------------

# Email addresses
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)

# Phone numbers — matches common international and domestic formats found on
# resumes.  Uses alternation so each branch is unambiguous, and digit-boundary
# assertions (?<!\d) / (?!\d) prevent false matches on year ranges or scores.
#
# Branches (tried in order, longest/most-specific first):
#   A: +CC-XXXXXXXXXX  e.g. +91-9876543210, +91 98765 43210
#   B: +CC SEP NXX SEP NXX SEP XXXX  e.g. +1 555 123 4567
#   C: (NXX) NXX-XXXX  e.g. (555) 123-4567
#   D: NXX SEP NXX SEP XXXX  e.g. 555-123-4567
#   E: bare 10-digit block  e.g. 9876543210
#
# NOT matched (by design):
#   "2021-2024"  → branch D needs 3-3-4 digits; this is 4-4 digits
#   "8.5"        → no branch matches a decimal
#   "25%"        → too few digits in any single group
_PHONE_RE = re.compile(
    r"""
    (?<!\d)          # must not be preceded by another digit
    (?:
        # Branch A: country-code + long local number (10 digits, 1 optional sep)
        # e.g. +91-9876543210 | +91 98765 43210 | +919876543210
        \+\d{1,3}[\s\-.]?\d{4,6}[\s\-.]?\d{4,5}
    |
        # Branch B: country-code + 3-part local
        # e.g. +1 555 123 4567
        \+\d{1,3}[\s\-.]\d{3}[\s\-.]\d{3}[\s\-.]\d{4}
    |
        # Branch C: parenthesised area code
        # e.g. (555) 123-4567
        \(\d{3}\)[\s\-.]\d{3}[\s\-.]\d{4}
    |
        # Branch D: NXX-NXX-XXXX without parens (exactly 3-3-4)
        # e.g. 555-123-4567  — 4-4 year ranges do NOT match this branch
        \d{3}[\s\-.]\d{3}[\s\-.]\d{4}
    |
        # Branch E: bare 10-digit block (no separators)
        # e.g. 9876543210
        \d{10}
    )
    (?!\d)           # must not be followed by another digit
    """,
    re.VERBOSE,
)

# URLs — http/https, linkedin, github (most common on resumes)
_URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|linkedin\.com/\S*|github\.com/\S*)",
    re.IGNORECASE,
)

# A "name line" heuristic: first 1–3 non-empty lines that look like a person's
# name — 1 to 5 words, all lowercase word chars (no digits, no punctuation).
# Since clean_text has already lowercased, names look like "john michael doe".
_NAME_LINE_RE = re.compile(r"^[a-z]+(?:\s+[a-z]+){0,4}$")

# Words that appear in section headers — if a short line contains one of these
# it is NOT a name even if it matches _NAME_LINE_RE.
_SECTION_KEYWORDS = {
    "education", "experience", "skills", "projects", "summary",
    "objective", "profile", "certifications", "awards", "publications",
    "interests", "languages", "volunteer", "references", "contact",
    "work", "academic", "professional", "technical", "personal",
}


def strip_pii(text: str) -> str:
    """Remove personally-identifiable information from cleaned resume text.

    Removes
    -------
    - Email addresses
    - Phone numbers (various international formats)
    - LinkedIn / GitHub / HTTP URLs
    - Candidate name (heuristic: first short word-only line ≤ 5 tokens that
      does not look like a section header)

    Parameters
    ----------
    text : str
        Output of clean_text() — lowercased, artifact-free, newlines preserved.

    Returns
    -------
    str
        Text with PII replaced by a single space, then whitespace re-normalised.

    Notes
    -----
    The heuristic name removal works for text-based resumes where the first
    meaningful line is the candidate's name.  It may miss names embedded
    deep in the document — but that is acceptable because:
      (a) the bias audit operates on the ORIGINAL text, not this copy, and
      (b) a missed name in the core copy does not break the similarity score.
    """
    # Remove emails, phone numbers, URLs
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)

    # Heuristic: remove candidate name from the first few lines
    lines = text.split("\n")
    name_removed = False
    cleaned_lines: List[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Only look in the first 5 lines for the name
        if i < 5 and not name_removed and stripped:
            tokens = stripped.split()
            # A name: 1–5 single words, all alphabetic, not a section keyword
            if (
                1 <= len(tokens) <= 5
                and _NAME_LINE_RE.match(stripped)
                and not any(tok in _SECTION_KEYWORDS for tok in tokens)
            ):
                name_removed = True
                cleaned_lines.append("")  # blank line preserves structure
                continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Re-normalise whitespace after replacements
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. detect_sections
# ---------------------------------------------------------------------------

# Maps canonical section names to lists of header keyword patterns.
# Patterns are matched against a lowercased line, so no need for IGNORECASE.
# Order matters: more specific phrases before shorter ones.
_SECTION_PATTERNS: Dict[str, List[str]] = {
    "experience": [
        "work experience", "professional experience", "employment history",
        "work history", "industry experience", "career history",
        "experience",
    ],
    "education": [
        "educational background", "academic background", "academic qualifications",
        "academic history", "education",
    ],
    "skills": [
        "technical skills", "core competencies", "key skills",
        "skills & expertise", "skills and expertise", "technologies",
        "tools & technologies", "tools and technologies",
        "programming languages", "skills",
    ],
    "projects": [
        "academic projects", "personal projects", "side projects",
        "notable projects", "key projects", "projects",
    ],
    "summary": [
        "professional summary", "career objective", "career summary",
        "executive summary", "about me", "profile", "objective", "summary",
    ],
    "certifications": [
        "certifications", "certificates", "accreditations", "licenses",
    ],
    "awards": [
        "awards", "honours", "honors", "achievements", "accomplishments",
    ],
}


def detect_sections(text: str) -> Dict[str, str]:
    """Split resume text into labelled sections using header keywords.

    Parameters
    ----------
    text : str
        Cleaned, lowercased resume text (output of strip_pii).

    Returns
    -------
    Dict[str, str]
        Keys are canonical section names ('education', 'experience', 'skills',
        'projects', 'summary', 'certifications', 'awards', 'other').
        Values are the text belonging to that section.
        If no recognised headers are found, all text goes under 'other'.

    Notes
    -----
    - Matching is on individual lines. A line is a header if, after stripping
      punctuation and extra spaces, it matches one of our keyword phrases.
    - We prefer longer phrases over shorter ones (order in _SECTION_PATTERNS).
    - Unknown/unrecognised content between recognised headers goes to 'other'.
    """

    # Pre-build a flat list: (canonical_name, compiled_regex) sorted by phrase
    # length descending so "work experience" matches before "experience".
    header_rules: List[Tuple[str, Pattern]] = []
    for section_name, phrases in _SECTION_PATTERNS.items():
        # Sort each phrase list by length DESC before compiling
        for phrase in sorted(phrases, key=len, reverse=True):
            # Match the phrase as the entire (stripped) line, allowing trailing
            # punctuation like a colon ("skills:").
            pattern = re.compile(
                r"^\s*" + re.escape(phrase) + r"\s*[:\-–—]?\s*$"
            )
            header_rules.append((section_name, pattern))

    # --- Walk lines and assign to sections ---
    sections: Dict[str, List[str]] = {name: [] for name in _SECTION_PATTERNS}
    sections["other"] = []

    current_section = "other"
    lines = text.split("\n")

    for line in lines:
        # Check if this line is a section header
        matched_section = _match_header(line, header_rules)
        if matched_section:
            current_section = matched_section
        else:
            sections[current_section].append(line)

    # Join and strip each section's content
    result: Dict[str, str] = {}
    for name, content_lines in sections.items():
        joined = "\n".join(content_lines).strip()
        if joined:  # omit empty sections from the result
            result[name] = joined

    # If nothing was detected (no headers found), all content is under 'other'
    if not result:
        result["other"] = text

    return result


def _match_header(line: str, rules: List[Tuple[str, Pattern]]) -> Optional[str]:
    """Return the canonical section name if *line* is a recognised header.

    Returns None if the line is not a header.
    """
    # Strip punctuation from both ends for robust matching
    stripped = line.strip().rstrip(":–—-").strip()
    if not stripped:
        return None
    for section_name, pattern in rules:
        if pattern.match(stripped):
            return section_name
    return None


# ---------------------------------------------------------------------------
# 4. normalize_aliases
# ---------------------------------------------------------------------------

def _load_aliases() -> Dict[str, str]:
    """Load the alias dictionary from aliases.json.

    Returns an empty dict if the file is missing (graceful degradation).
    """
    if not _ALIASES_PATH.exists():
        return {}
    with _ALIASES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def normalize_aliases(text: str, aliases: Optional[Dict[str, str]] = None) -> str:
    """Replace skill abbreviations with their canonical full names.

    Uses WHOLE-WORD matching via \\b boundaries so that short tokens like
    "ml" are not replaced inside "html" or "xml".

    Parameters
    ----------
    text : str
        Cleaned resume text.
    aliases : dict[str, str] | None
        Alias map {abbreviation: canonical}. If None, loaded from aliases.json.

    Returns
    -------
    str
        Text with abbreviations expanded.

    Examples
    --------
    "experience with ml and js"  →  "experience with machine learning and javascript"
    "skilled in html and css"    →  unchanged (html is not in aliases)
    "wrote numpy (np) scripts"   →  unchanged ("np" is not an alias to avoid FP)
    """
    if aliases is None:
        aliases = _load_aliases()

    # Sort by length DESC so longer phrases are replaced first.
    # e.g. "ci cd" before "cd" if both existed.
    sorted_aliases = sorted(aliases.items(), key=lambda kv: len(kv[0]), reverse=True)

    for abbrev, canonical in sorted_aliases:
        # \b = word boundary.  re.escape handles hyphens in multi-word abbrevs.
        pattern = re.compile(r"\b" + re.escape(abbrev) + r"\b", re.IGNORECASE)
        text = pattern.sub(canonical, text)

    return text


# ---------------------------------------------------------------------------
# 5. preprocess  — the public entry point
# ---------------------------------------------------------------------------

def preprocess(text: str) -> dict:
    """Run the full preprocessing pipeline on raw extracted PDF text.

    Parameters
    ----------
    text : str
        Raw text from extraction.py (may contain \\f page separators).

    Returns
    -------
    dict with keys:
        "original_text" : str  – The raw input, NEVER modified. The bias audit
                                 phase will inject names into this copy.
        "core_text"     : str  – Cleaned, PII-stripped, alias-normalised text
                                 used for semantic scoring and skill extraction.
        "sections"      : dict – {section_name: section_text} mapping.

    Pipeline
    --------
    raw → clean_text → strip_pii → normalize_aliases → core_text
                                 ↓
                          detect_sections → sections
    """
    # Keep the raw text unchanged — bias audit depends on it
    original_text = text

    # Step 1: clean (lowercase, kill artifacts, preserve newlines)
    cleaned = clean_text(text)

    # Step 2: remove PII (email, phone, URL, name heuristic)
    pii_stripped = strip_pii(cleaned)

    # Step 3: detect sections from the PII-stripped text
    sections = detect_sections(pii_stripped)

    # Step 4: normalise aliases on the PII-stripped text → final core_text
    core_text = normalize_aliases(pii_stripped)

    return {
        "original_text": original_text,
        "core_text": core_text,
        "sections": sections,
    }
