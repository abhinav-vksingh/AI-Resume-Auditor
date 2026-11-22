"""
tests/test_extraction.py

Unit tests for services/extraction.py.

We test the pure Python function directly — no HTTP, no FastAPI TestClient.
Pure functions are faster and easier to reason about in a viva.

Run with:
    pytest tests/test_extraction.py -v
"""

import io
from typing import List
import pytest
import fitz  # PyMuPDF

from backend.app.services.extraction import extract_text_from_pdf, ExtractionError


# ---------------------------------------------------------------------------
# Helpers — build minimal in-memory PDFs for testing
# ---------------------------------------------------------------------------

def _make_pdf_bytes(pages: List[str]) -> bytes:
    """Create a valid PDF in memory with the given text on successive pages.

    Uses PyMuPDF itself so we don't need an external fixture file.
    """
    doc = fitz.open()  # new empty document
    for page_text in pages:
        page = doc.new_page()
        # Insert text at a fixed position — good enough for extraction tests
        page.insert_text((72, 72), page_text, fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_empty_pdf_bytes() -> bytes:
    """Create a valid PDF with one blank page (no text layer)."""
    doc = fitz.open()
    doc.new_page()  # blank page — no text inserted
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractTextFromPdf:

    def test_single_page_returns_text(self):
        """Basic happy-path: single-page PDF with known text."""
        content = "Software Engineer with 3 years of Python experience."
        pdf_bytes = _make_pdf_bytes([content])

        result = extract_text_from_pdf(pdf_bytes)

        assert isinstance(result, dict)
        assert "text" in result
        assert "page_count" in result
        assert "char_count" in result
        assert result["page_count"] == 1
        # The extracted text should contain our inserted string
        assert "Python" in result["text"]
        assert result["char_count"] > 0

    def test_multi_page_page_count(self):
        """Multi-page PDF should report the correct page count."""
        pages = ["Page one content.", "Page two content.", "Page three content."]
        pdf_bytes = _make_pdf_bytes(pages)

        result = extract_text_from_pdf(pdf_bytes)

        assert result["page_count"] == 3
        # Pages are joined with form-feed
        assert result["text"].count("\f") == 2  # 3 pages → 2 separators

    def test_multi_page_text_contains_all_pages(self):
        """All page content must appear somewhere in the extracted text."""
        pages = ["Alpha section.", "Beta section.", "Gamma section."]
        pdf_bytes = _make_pdf_bytes(pages)

        result = extract_text_from_pdf(pdf_bytes)

        for keyword in ["Alpha", "Beta", "Gamma"]:
            assert keyword in result["text"], f"Expected '{keyword}' in extracted text"

    def test_char_count_matches_stripped_text(self):
        """char_count should equal len(text.strip())."""
        pdf_bytes = _make_pdf_bytes(["Hello world."])
        result = extract_text_from_pdf(pdf_bytes)
        assert result["char_count"] == len(result["text"].strip())

    def test_empty_pdf_raises_extraction_error(self):
        """A blank-page PDF (scanned-like) must raise ExtractionError, not crash."""
        pdf_bytes = _make_empty_pdf_bytes()

        with pytest.raises(ExtractionError) as exc_info:
            extract_text_from_pdf(pdf_bytes)

        assert "No text" in str(exc_info.value) or "scanned" in str(exc_info.value).lower()

    def test_corrupt_bytes_raises_extraction_error(self):
        """Random bytes that aren't a PDF must raise ExtractionError."""
        junk = b"This is definitely not a PDF file \x00\x01\x02"

        with pytest.raises(ExtractionError):
            extract_text_from_pdf(junk)

    def test_empty_bytes_raises_extraction_error(self):
        """Empty byte string must raise ExtractionError."""
        with pytest.raises(ExtractionError):
            extract_text_from_pdf(b"")
