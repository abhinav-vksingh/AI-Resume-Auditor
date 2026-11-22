"""
routers/match.py – POST /match endpoint.

Orchestrates the two Phase-3 services:
  1. semantic.compute_semantic_similarity  → semantic_score (float, 0–100)
  2. skills.run_skill_analysis             → skill table + skill_overlap

The final_score fusion (0.6 × semantic + 0.4 × skill) lives in Phase 4
(scoring service).  This router returns BOTH components separately so the
caller can see the breakdown.
"""

from fastapi import APIRouter, HTTPException

from backend.app.schemas import MatchRequest, MatchResponse, SkillResult
from backend.app.services.semantic import compute_semantic_similarity
from backend.app.services.skills import run_skill_analysis

router = APIRouter(prefix="/match", tags=["Matching"])


@router.post(
    "",
    response_model=MatchResponse,
    summary="Compute semantic similarity and skill overlap between resume and JD",
)
def match(body: MatchRequest) -> MatchResponse:
    """
    Accepts the preprocessed resume (`core_text` + `sections` from POST /preprocess)
    and a raw job description, then returns:

    - **semantic_score** – embedding cosine similarity (0–100). Computed by our
      pipeline, NOT by an LLM.
    - **skill_overlap** – (matched + 0.5×weak) / total × 100. Each skill is
      labelled Matched / Weak / Missing.
    - **skill_table** – transparent per-skill breakdown.

    Errors:
    - 422 if the job description contains no recognisable skills from our
      dictionary (usually means the JD is too generic or in a different domain).
    """
    # --- Semantic similarity ---
    try:
        semantic_score = compute_semantic_similarity(
            resume_text=body.core_text,
            job_description=body.job_description,
            sections=body.sections or None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic similarity computation failed: {exc}",
        ) from exc

    # --- Skill analysis ---
    try:
        skill_result = run_skill_analysis(
            job_description=body.job_description,
            core_text=body.core_text,
            sections=body.sections,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Skill analysis failed: {exc}",
        ) from exc

    if skill_result["total_required"] == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "No recognisable skills were found in the job description. "
                "The JD may be too generic or outside the supported skill categories. "
                "skill_overlap will be 0."
            ),
        )

    skill_table = [
        SkillResult(skill=row["skill"], status=row["status"])
        for row in skill_result["skill_table"]
    ]

    return MatchResponse(
        semantic_score=semantic_score,
        skill_overlap=skill_result["skill_overlap"],
        skill_table=skill_table,
        matched_count=skill_result["matched_count"],
        weak_count=skill_result["weak_count"],
        missing_count=skill_result["missing_count"],
        total_required=skill_result["total_required"],
    )
