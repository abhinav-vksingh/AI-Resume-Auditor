"""
routers/bias.py – POST /bias-audit endpoint.

Accepts the raw resume text (original_text from /extract, before preprocessing)
and a job description, then runs the counterfactual bias audit:
  - Swaps names (white_american vs black_american) OR
  - Swaps universities (well_known vs lesser_known)
  - Scores each variant with the same pipeline as /analyze
  - Returns mean delta, std, paired t-test p-value, and all raw rows

IMPORTANT: works on ORIGINAL text so the name/university is present before
preprocessing strips it.  For name audits this means delta ≈ 0 (name-blind
pipeline), which is the intended result — not a bug.
"""

import math
from fastapi import APIRouter, HTTPException

from backend.app.schemas import AuditRow, BiasAuditRequest, BiasAuditResponse
from backend.app.services.bias import run_audit

router = APIRouter(prefix="/bias-audit", tags=["Bias Audit"])


@router.post(
    "",
    response_model=BiasAuditResponse,
    summary="Counterfactual bias audit: swap name or university, compare scores",
)
def bias_audit(body: BiasAuditRequest) -> BiasAuditResponse:
    """
    Runs a Bertrand & Mullainathan-style counterfactual audit:

    1. Takes the **original** (pre-preprocessing) resume text.
    2. Builds N pairs of variants — only the chosen attribute differs.
    3. Scores each variant with the full pipeline (same as POST /analyze).
    4. Returns delta = score_B − score_A, mean, std, and paired t-test.

    **Expected result for `attribute=name`**: delta ≈ 0 — our pipeline strips
    candidate names before scoring, so it is demonstrably name-blind.

    **For `attribute=university`**: delta may be non-zero because university
    names survive preprocessing and can influence semantic similarity.

    See the `disclaimer` field in the response for the mandatory caveat.

    Errors:
    - 400 if the resume text is empty or no pairs could be generated.
    - 500 for unexpected pipeline errors.
    """
    try:
        result = run_audit(
            original_texts=[body.original_text],
            job_description=body.job_description,
            attribute=body.attribute,
            n_pairs=body.n_pairs,
            mode=body.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Bias audit failed: {exc}",
        ) from exc

    if result["n_observations"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No identity pairs could be generated. Check names.json / universities.json.",
        )

    rows = [
        AuditRow(
            resume_index=r["resume_index"],
            identity_a=r["identity_a"],
            identity_b=r["identity_b"],
            score_a=r["score_a"],
            score_b=r["score_b"],
            delta=r["delta"],
        )
        for r in result["raw_rows"]
    ]

    p_val = result["p_value"]
    if p_val is not None and (math.isnan(p_val) or math.isinf(p_val)):
        p_val = None

    std_val = result["std_delta"]
    if std_val is not None and (math.isnan(std_val) or math.isinf(std_val)):
        std_val = None

    return BiasAuditResponse(
        attribute=result["attribute"],
        group_a_label=result["group_a_label"],
        group_b_label=result["group_b_label"],
        mode=result["mode"],
        mean_delta=result["mean_delta"],
        std_delta=std_val,
        p_value=p_val,
        significant=result["significant"],
        n_observations=result["n_observations"],
        raw_rows=rows,
        disclaimer=result["disclaimer"],
    )
