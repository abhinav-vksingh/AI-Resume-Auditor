"""
routers/preprocess.py – HTTP routes for text preprocessing.

Accepts the raw text produced by POST /extract and returns the cleaned,
PII-stripped, section-detected, alias-normalised result.

Design note: we accept TEXT (JSON body), not a PDF here.
  Caller flow: upload PDF → /extract → pipe text into /preprocess.
  This separation keeps each endpoint single-responsibility.
"""

from fastapi import APIRouter, HTTPException

from backend.app.schemas import PreprocessRequest, PreprocessResponse
from backend.app.services.preprocessing import preprocess

router = APIRouter(prefix="/preprocess", tags=["Preprocessing"])


@router.post(
    "",
    response_model=PreprocessResponse,
    summary="Clean, PII-strip, section-detect and alias-normalise raw resume text",
)
def preprocess_text(body: PreprocessRequest) -> PreprocessResponse:
    """
    Accepts raw resume text (the 'text' field from POST /extract) and runs
    the full preprocessing pipeline:

    1. **clean_text** – lowercase, remove PDF artifacts, preserve newlines.
    2. **strip_pii** – remove email, phone, URLs, candidate name (first line heuristic).
    3. **detect_sections** – split into education / experience / skills / projects / …
    4. **normalize_aliases** – expand abbreviations (ml → machine learning, etc.)

    Returns `original_text` (untouched), `core_text` (processed), and `sections`.

    Errors:
    - 400 if the text is empty after preprocessing (all content was PII / artifacts).
    """
    try:
        result = preprocess(body.text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing failed unexpectedly: {exc}",
        ) from exc

    if not result["core_text"].strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "After preprocessing, no usable text remains. "
                "The document may consist entirely of PII or PDF artifacts."
            ),
        )

    return PreprocessResponse(
        original_text=result["original_text"],
        core_text=result["core_text"],
        sections=result["sections"],
    )
