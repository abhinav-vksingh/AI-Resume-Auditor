"""
schemas.py – Pydantic request/response models for every endpoint.

Why Pydantic?
  FastAPI uses these for automatic validation AND OpenAPI docs generation.
  Every field has a description so the /docs page is self-explanatory.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="Always 'ok' when the server is up")
    version: str = Field(..., description="API version string")


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

class ExtractionResponse(BaseModel):
    """Returned by POST /extract after a PDF is uploaded."""

    filename: str = Field(..., description="Original uploaded filename")
    page_count: int = Field(..., description="Number of pages found in the PDF")
    char_count: int = Field(..., description="Total characters extracted")
    text: str = Field(..., description="Raw extracted text, line-breaks preserved")


class ExtractionError(BaseModel):
    """Returned when extraction fails (e.g. scanned PDF, password-protected)."""

    filename: str
    error: str = Field(..., description="Human-readable reason for the failure")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class PreprocessRequest(BaseModel):
    """Body for POST /preprocess — accepts the raw text from /extract."""

    text: str = Field(
        ...,
        description=(
            "Raw text exactly as returned by POST /extract. "
            "May contain \\f page separators."
        ),
        min_length=1,
    )


class PreprocessResponse(BaseModel):
    """Returned by POST /preprocess."""

    original_text: str = Field(
        ...,
        description=(
            "The raw input text, completely unmodified. "
            "Preserved for the bias-audit phase, which injects names into this copy."
        ),
    )
    core_text: str = Field(
        ...,
        description=(
            "Cleaned, PII-stripped, alias-normalised text. "
            "This is the version fed into semantic scoring and skill extraction."
        ),
    )
    sections: Dict[str, str] = Field(
        ...,
        description=(
            "Resume split into labelled sections "
            "(education, experience, skills, projects, summary, other, …). "
            "Each key maps to the text belonging to that section. "
            "If no headers are found, all text appears under 'other'."
        ),
    )


# ---------------------------------------------------------------------------
# Matching (Phase 3)
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    """Body for POST /match."""

    core_text: str = Field(
        ...,
        description="Preprocessed resume text — the core_text field from POST /preprocess.",
        min_length=1,
    )
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Section dict from POST /preprocess. "
            "Used for chunked embedding and primary-section skill classification. "
            "Pass an empty dict {} if sections are unavailable."
        ),
    )
    job_description: str = Field(
        ...,
        description="Raw job description text. The pipeline cleans it internally.",
        min_length=1,
    )


class SkillResult(BaseModel):
    """One row in the skill table."""

    skill: str = Field(..., description="Canonical skill name from skills.json")
    status: Literal["Matched", "Weak", "Missing"] = Field(
        ...,
        description=(
            "Matched  – found in skills/experience/projects section, or mentioned >1 time. "
            "Weak     – mentioned exactly once, not in a primary section. "
            "Missing  – not found anywhere in the resume."
        ),
    )


class MatchResponse(BaseModel):
    """Returned by POST /match."""

    semantic_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Cosine similarity between resume and JD embeddings, scaled 0–100.",
    )
    skill_overlap: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Skill overlap score 0–100. "
            "Formula: (matched + 0.5 × weak) / total_required × 100."
        ),
    )
    skill_table: List[SkillResult] = Field(
        ...,
        description="Per-skill classification: Matched / Weak / Missing.",
    )
    matched_count: int = Field(..., description="Number of Matched skills.")
    weak_count: int = Field(..., description="Number of Weak skills.")
    missing_count: int = Field(..., description="Number of Missing skills.")
    total_required: int = Field(
        ..., description="Total skills extracted from the JD."
    )


# ---------------------------------------------------------------------------
# Full Analysis — POST /analyze  (Phase 4)
# ---------------------------------------------------------------------------

class ScoreComponents(BaseModel):
    """Transparent breakdown of how final_score was computed."""

    semantic: float = Field(
        ..., ge=0, le=100,
        description="Semantic similarity component (cosine embedding, 0–100).",
    )
    skill_overlap: float = Field(
        ..., ge=0, le=100,
        description="Skill overlap component (matched+0.5×weak / total, 0–100).",
    )
    weights: Dict[str, float] = Field(
        ...,
        description=(
            'Fusion weights read from config: {"semantic": 0.6, "skill_overlap": 0.4}. '
            "final_score = semantic × weights.semantic + skill_overlap × weights.skill_overlap."
        ),
    )


class SkillBreakdown(BaseModel):
    """Grouped skill lists for the UI table."""

    matched: List[str] = Field(..., description="Skills found in primary resume sections.")
    weak: List[str] = Field(
        ...,
        description="Skills mentioned once, not in a primary section.",
    )
    missing: List[str] = Field(..., description="Required skills absent from the resume.")


class Explanation(BaseModel):
    """Rule-based textual explanation.  No LLM — every sentence is a coded rule."""

    positive_factors: List[str] = Field(
        ...,
        description="Factors working in the candidate's favour, derived from scores.",
    )
    negative_factors: List[str] = Field(
        ...,
        description="Gaps or weaknesses, e.g. missing skills or low alignment.",
    )


class AnalyzeResponse(BaseModel):
    """Returned by POST /analyze — the complete pipeline output."""

    original_text: str = Field(
        default="",
        description="Raw extracted resume text before preprocessing, preserved for downstream auditing and feedback.",
    )
    final_score: float = Field(
        ..., ge=0, le=100,
        description=(
            "Weighted final score. "
            "Formula: semantic_weight × semantic + skill_weight × skill_overlap."
        ),
    )
    components: ScoreComponents = Field(
        ..., description="Individual component scores and weights for full transparency."
    )
    skills: SkillBreakdown = Field(
        ..., description="Matched / Weak / Missing skill lists."
    )
    explanation: Explanation = Field(
        ..., description="Rule-based positive and negative factors."
    )


# ---------------------------------------------------------------------------
# Bias Audit — POST /bias-audit  (Phase 5)
# ---------------------------------------------------------------------------

class BiasAuditRequest(BaseModel):
    """Body for POST /bias-audit."""

    original_text: str = Field(
        ...,
        description=(
            "Raw resume text from POST /extract — BEFORE preprocessing. "
            "The audit injects names/universities into this text, then runs "
            "each variant through the full pipeline independently."
        ),
        min_length=1,
    )
    job_description: str = Field(
        ...,
        description="Job description to score against (held constant for all variants).",
        min_length=1,
    )
    attribute: Literal["name", "university"] = Field(
        ...,
        description=(
            "'name'       – swap candidate name between white_american / black_american groups. "
            "'university' – swap university between well_known / lesser_known groups."
        ),
    )
    n_pairs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of identity pairs to test (capped at available data, max 20).",
    )
    mode: Literal["raw", "redacted"] = Field(
        default="redacted",
        description=(
            "'redacted' – score through the production pipeline (strip_pii runs before encoding). "
            "For name audits, delta ≈ 0 by design — proves name-blindness. "
            "'raw' – score WITHOUT strip_pii so the embedding model sees the name or university. "
            "Reveals whether the model's representations associate identity signals with scores."
        ),
    )


class AuditRow(BaseModel):
    """One comparison pair in the bias audit."""

    resume_index: int = Field(..., description="0-based index of the source resume.")
    identity_a: str   = Field(..., description="Identity value used for group A (e.g. 'Emily Walsh').")
    identity_b: str   = Field(..., description="Identity value used for group B (e.g. 'Lakisha Jones').")
    score_a: float    = Field(..., description="Final score for the group-A variant.")
    score_b: float    = Field(..., description="Final score for the group-B variant.")
    delta: float      = Field(..., description="score_B − score_A. Positive means group B scored higher.")


class BiasAuditResponse(BaseModel):
    """Returned by POST /bias-audit."""

    attribute: str                 = Field(..., description="Attribute that was swapped ('name' or 'university').")
    group_a_label: str             = Field(..., description="Label for group A (e.g. 'white_american').")
    group_b_label: str             = Field(..., description="Label for group B (e.g. 'black_american').")
    mode: str                      = Field(..., description="Scoring mode used: 'raw' or 'redacted'.")
    mean_delta: float              = Field(..., description="Mean of (score_B − score_A) across all pairs.")
    std_delta: Optional[float]     = Field(None, description="Standard deviation of deltas (None if < 2 observations or undefined).")
    p_value: Optional[float]       = Field(
        None,
        description="Two-tailed paired t-test p-value. None if fewer than 2 observations or identical scores.",
    )
    significant: bool              = Field(..., description="True if p_value is not None and p_value < 0.05.")
    n_observations: int            = Field(..., description="Total number of (resume × pair) comparisons made.")
    raw_rows: List[AuditRow]       = Field(..., description="Per-pair detail for full transparency.")
    disclaimer: str                = Field(
        ...,
        description=(
            "Mandatory disclaimer: this is a small project-level audit, "
            "not proof of fairness or unfairness."
        ),
    )


# ---------------------------------------------------------------------------
# LLM Feedback — POST /feedback  (Phase 6)
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """Body for POST /feedback."""

    job_description: str = Field(
        ...,
        description="The target job description text.",
        min_length=1,
    )
    resume_text: str = Field(
        ...,
        description="The candidate's resume text (typically core_text from /preprocess).",
        min_length=1,
    )
    skill_table: List[SkillResult] = Field(
        default_factory=list,
        description="Skill analysis classification table from /match or /analyze.",
    )


class StructuredFeedback(BaseModel):
    """Validated structured output produced by the LLM."""

    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills verified in the resume that match the job description.",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Required qualifications or skills lacking direct evidence in the resume.",
    )
    rewritten_bullet: str = Field(
        default="",
        description="A rewritten version of an existing resume bullet point to improve technical clarity.",
    )
    explanation: str = Field(
        default="",
        description="Objective synthesis of candidate qualifications against requirements.",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable next steps grounded strictly in existing facts.",
    )


class FeedbackResponse(BaseModel):
    """Returned by POST /feedback."""

    prompt_version: str = Field(..., description="Prompt version executed (v1, v2, or v3).")
    format_valid: bool = Field(
        ...,
        description="True if the LLM output matched the required JSON schema structure.",
    )
    feedback: StructuredFeedback = Field(
        ...,
        description="Parsed structured feedback content (with safe defaults if validation failed).",
    )
    raw_response: str = Field(
        ...,
        description="Verbatim text response emitted by the LLM.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details or format validation diagnostic message, if any.",
    )
