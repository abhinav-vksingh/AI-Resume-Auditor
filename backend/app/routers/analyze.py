"""
routers/analyze.py – POST /analyze: the full end-to-end pipeline in one call.

Accepts a PDF resume (multipart file) + a job description (form field).
Internally chains all prior services:
  extraction → preprocessing → semantic + skill → scoring → explain

Returns a single AnalyzeResponse with final_score, components, skill
breakdown, and rule-based explanation.

Why one endpoint instead of chaining the smaller ones?
  Convenience for the frontend: one round-trip instead of four.
  The smaller endpoints (/extract, /preprocess, /match) remain available
  for debugging individual steps.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.schemas import (
    AnalyzeResponse,
    Explanation,
    ScoreComponents,
    SkillBreakdown,
)
from backend.app.services.explain import generate_explanation
from backend.app.services.extraction import ExtractionError
from backend.app.services.extraction import extract_text_from_pdf
from backend.app.services.preprocessing import preprocess
from backend.app.services.scoring import compute_final_score
from backend.app.services.semantic import compute_semantic_similarity
from backend.app.services.skills import run_skill_analysis

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post(
    "",
    response_model=AnalyzeResponse,
    summary="Full pipeline: upload a PDF resume + JD text → scored, explained result",
)
async def analyze(
    file: UploadFile = File(..., description="PDF resume to audit."),
    job_description: str = Form(
        ..., description="Raw job description text.", min_length=1
    ),
) -> AnalyzeResponse:
    """
    End-to-end analysis in a single request:

    1. **Extract** text from the uploaded PDF (PyMuPDF).
    2. **Preprocess** — clean, strip PII, detect sections, normalise aliases.
    3. **Semantic similarity** — embed resume + JD, cosine similarity → 0–100.
    4. **Skill analysis** — extract required skills from JD, classify each
       against resume as Matched / Weak / Missing.
    5. **Score fusion** — final_score = 0.6 × semantic + 0.4 × skill_overlap
       (weights from config, not hardcoded).
    6. **Explain** — rule-based positive/negative factors from the numbers.
       No LLM is involved in any step.

    Errors:
    - 400  empty or non-PDF file / no usable text after preprocessing.
    - 422  scanned / encrypted PDF or no recognisable skills in the JD.
    - 500  unexpected internal error.
    """

    # ------------------------------------------------------------------
    # Step 1 — PDF extraction
    # ------------------------------------------------------------------
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected a PDF file, got content-type '{file.content_type}'. "
                "Please upload a .pdf file."
            ),
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        extracted = extract_text_from_pdf(file_bytes)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # Step 2 — Preprocessing
    # Preserve the raw extracted text before PII stripping so the
    # AnalyzeResponse can return it for downstream bias auditing.
    # ------------------------------------------------------------------
    raw_text  = extracted["text"]
    preprocessed = preprocess(raw_text)
    core_text = preprocessed["core_text"]
    sections  = preprocessed["sections"]

    if not core_text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable text remains after preprocessing. "
                "The document may consist entirely of PII or PDF artifacts."
            ),
        )

    # ------------------------------------------------------------------
    # Step 3 — Semantic similarity
    # ------------------------------------------------------------------
    try:
        semantic_score = compute_semantic_similarity(
            resume_text=core_text,
            job_description=job_description,
            sections=sections,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic similarity computation failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # Step 4 — Skill analysis
    # ------------------------------------------------------------------
    try:
        skill_result = run_skill_analysis(
            job_description=job_description,
            core_text=core_text,
            sections=sections,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Skill analysis failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # Step 5 — Score fusion
    # ------------------------------------------------------------------
    score_result = compute_final_score(
        semantic_score=semantic_score,
        skill_overlap=skill_result["skill_overlap"],
    )

    # ------------------------------------------------------------------
    # Step 6 — Rule-based explanation
    # ------------------------------------------------------------------
    explanation = generate_explanation(
        semantic_score=semantic_score,
        skill_overlap=skill_result["skill_overlap"],
        final_score=score_result["final_score"],
        skill_table=skill_result["skill_table"],
    )

    # ------------------------------------------------------------------
    # Assemble response
    # ------------------------------------------------------------------
    matched_skills = [r["skill"] for r in skill_result["skill_table"] if r["status"] == "Matched"]
    weak_skills    = [r["skill"] for r in skill_result["skill_table"] if r["status"] == "Weak"]
    missing_skills = [r["skill"] for r in skill_result["skill_table"] if r["status"] == "Missing"]

    return AnalyzeResponse(
        original_text=raw_text,
        final_score=score_result["final_score"],
        components=ScoreComponents(
            semantic=score_result["semantic_score"],
            skill_overlap=score_result["skill_overlap"],
            weights=score_result["weights"],
        ),
        skills=SkillBreakdown(
            matched=matched_skills,
            weak=weak_skills,
            missing=missing_skills,
        ),
        explanation=Explanation(
            positive_factors=explanation["positive_factors"],
            negative_factors=explanation["negative_factors"],
        ),
    )
