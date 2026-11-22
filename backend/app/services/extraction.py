"""
services/extraction.py – PDF text extraction using PyMuPDF (fitz).

Why PyMuPDF?
  It's the fastest pure-Python PDF library and handles multi-column layouts
  better than pdfplumber for our use-case. We preserve line breaks so that
  later preprocessing can detect section headers.

Design rule (from project spec):
  This module has ONE job: bytes -> text. No scoring, no NLP.
"""

from typing import Any, Dict, List

import fitz  # PyMuPDF


class ExtractionError(Exception):
    """Raised when we cannot extract usable text from a PDF.

    Callers should convert this to an HTTP 422 response.
    Possible causes:
      - Scanned PDF (image-only, no text layer)
      - Password-protected / encrypted PDF
      - Corrupted file
    """


def extract_text_from_pdf(file_bytes: bytes) -> Dict[str, Any]:
    """Extract all text from a PDF given its raw bytes.

    Parameters
    ----------
    file_bytes : bytes
        The raw binary content of a PDF file.

    Returns
    -------
    dict with keys:
        "text"       – Full extracted text, pages separated by form-feed '\\f'.
                       Line breaks within pages are preserved.
        "page_count" – Number of pages in the document.
        "char_count" – Total character count of extracted text.

    Raises
    ------
    ExtractionError
        If the PDF is unreadable or yields no extractable text at all.
    """

    # Open PDF from bytes (no temp file needed — fitz supports memory streams)
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ExtractionError(
            "Could not open the file as a PDF. It may be corrupted or "
            f"password-protected. Detail: {exc}"
        ) from exc

    page_count = len(doc)
    pages_text: List[str] = []

    for page_num, page in enumerate(doc, start=1):
        # get_text("text") returns plain text with newlines preserved.
        # "blocks" mode is better for multi-column but harder to join cleanly;
        # "text" is good enough for single-column resumes.
        raw = page.get_text("text")
        pages_text.append(raw)

    doc.close()

    # Join pages with form-feed so downstream code can split them if needed
    full_text = "\f".join(pages_text)
    char_count = len(full_text.strip())

    if char_count == 0:
        raise ExtractionError(
            f"No text could be extracted from this PDF ({page_count} page(s) found). "
            "This is usually because the resume is a scanned image. "
            "Please upload a text-based PDF or convert it first."
        )

    return {
        "text": full_text,
        "page_count": page_count,
        "char_count": char_count,
    }
