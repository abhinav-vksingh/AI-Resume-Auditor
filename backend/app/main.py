"""
main.py – FastAPI application entry point.

Registers all routers and exposes the health endpoint.
Run with:
    uvicorn backend.app.main:app --reload
"""

from fastapi import FastAPI

from backend.app.config import settings
from backend.app.schemas import HealthResponse
from backend.app.routers import extract as extract_router
from backend.app.routers import preprocess as preprocess_router
from backend.app.routers import match as match_router
from backend.app.routers import analyze as analyze_router
from backend.app.routers import bias as bias_router
from backend.app.routers import feedback as feedback_router

# ---------------------------------------------------------------------------
# App instantiation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Resume Auditor",
    description=(
        "Transparent resume–job-description scoring pipeline. "
        "Score = 0.6 × semantic similarity + 0.4 × skill overlap. "
        "No LLM calculates the score."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(extract_router.router)
app.include_router(preprocess_router.router)
app.include_router(match_router.router)
app.include_router(analyze_router.router)
app.include_router(bias_router.router)
app.include_router(feedback_router.router)

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Meta"],
    summary="Liveness check",
)
def health() -> HealthResponse:
    """Returns 200 OK when the server is running.

    Useful for scripted readiness checks before hitting other endpoints.
    """
    return HealthResponse(status="ok", version=app.version)
