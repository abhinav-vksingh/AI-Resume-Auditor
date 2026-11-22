"""
routers/extract.py – HTTP routes for PDF text extraction.

Keeps routing logic separate from business logic (which lives in services/).
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas import ExtractionResponse
from backend.app.services.extraction import extract_text_from_pdf, ExtractionError

router = APIRouter(prefix="/extract", tags=["Extraction"])


@router.post(
    "",
    response_model=ExtractionResponse,
    summary="Upload a PDF resume and get back raw extracted text",
)
async def extract_pdf(file: UploadFile = File(..., description="PDF file to parse")):
    """
    Accepts a multipart PDF upload, runs PyMuPDF extraction, and returns the
    raw text. Downstream preprocessing/scoring endpoints expect this text as
    their input.

    Errors:
    - 400 if the file is not a PDF or is empty.
    - 422 if PyMuPDF cannot read it (scanned, corrupted, password-protected).
    """

    # Basic content-type guard — browsers often send 'application/pdf'
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a PDF file, got content-type '{file.content_type}'. "
                   "Please upload a .pdf file.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = extract_text_from_pdf(file_bytes)
    except ExtractionError as exc:
        # 422 Unprocessable Entity: file is a valid PDF structurally but we
        # cannot get useful text from it (scanned image, encrypted, etc.)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ExtractionResponse(
        filename=file.filename or "unknown.pdf",
        page_count=result["page_count"],
        char_count=result["char_count"],
        text=result["text"],
    )
