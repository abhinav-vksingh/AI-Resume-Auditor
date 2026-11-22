"""
routers/feedback.py – HTTP routes for prompt-engineered LLM feedback (Phase 6).

Accepts job description, resume text, and skill table from /analyze,
invoking the provider-agnostic LLM with one of the prompt templates (v1, v2, v3).
Returns structured JSON with format validity flag.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas import FeedbackRequest, FeedbackResponse, StructuredFeedback
from backend.app.services.llm_feedback import generate_feedback

router = APIRouter(prefix="/feedback", tags=["LLM Feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    summary="Generate evidence-grounded resume review using LLM prompts (v1, v2, v3)",
)
def get_feedback(
    body: FeedbackRequest,
    prompt_version: Literal["v1", "v2", "v3"] = Query(
        "v3",
        description="Prompt version: v1 (baseline role), v2 (criteria-based), v3 (structured schema)",
    ),
) -> FeedbackResponse:
    """
    Generate candidate-facing qualitative feedback without altering the numeric match score:

    - **v1 (baseline)**: General review instructions.
    - **v2 (criteria)**: Explicit evaluation of relevance, skill evidence, clarity, and grounding.
    - **v3 (structured)**: Rigid schema enforcing matched skills, gaps, bullet rewrite, and next steps.

    Hard rules embedded in all prompts:
    - Never invent skills, metrics, or experiences.
    - Explicitly state when information is missing.
    - Never calculate or override the numeric match score.

    Output format validation:
    - If the LLM generates non-JSON or violates schema keys, `format_valid` will be `false`
      with diagnostics rather than an unhandled 500 crash.
    """
    skill_table_dicts = [s.model_dump() for s in body.skill_table]

    result = generate_feedback(
        job_description=body.job_description,
        resume_text=body.resume_text,
        skill_table=skill_table_dicts,
        prompt_version=prompt_version,
    )

    feedback_obj = StructuredFeedback(
        matched_skills=result["feedback"].get("matched_skills", []),
        gaps=result["feedback"].get("gaps", []),
        rewritten_bullet=result["feedback"].get("rewritten_bullet", ""),
        explanation=result["feedback"].get("explanation", ""),
        recommendations=result["feedback"].get("recommendations", []),
    )

    return FeedbackResponse(
        prompt_version=result["prompt_version"],
        format_valid=result["format_valid"],
        feedback=feedback_obj,
        raw_response=result["raw_response"],
        error=result["error"],
    )
