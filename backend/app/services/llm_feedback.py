"""
services/llm_feedback.py – Evidence-grounded LLM feedback generation.

PRINCIPLES (Phase 6):
  1. The numeric score is calculated ONLY by our own pipeline (scoring.py).
     The LLM never calculates, overrides, or alters the score.
  2. Provider-agnostic: Configured completely via environment variables:
     LLM_API_KEY, LLM_MODEL, and LLM_BASE_URL.
  3. Prompts reside in `prompts/` (v1_baseline.txt, v2_criteria.txt, v3_structured.txt).
  4. Robust format validation: If the LLM generates malformed JSON or omits required
     keys, flag a format failure instead of crashing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx

from backend.app.config import settings

# ---------------------------------------------------------------------------
# Paths and aliases
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"

_PROMPT_VERSION_MAP = {
    "v1": "v1_baseline.txt",
    "v1_baseline": "v1_baseline.txt",
    "v2": "v2_criteria.txt",
    "v2_criteria": "v2_criteria.txt",
    "v3": "v3_structured.txt",
    "v3_structured": "v3_structured.txt",
}

_REQUIRED_KEYS = {
    "matched_skills",
    "gaps",
    "rewritten_bullet",
    "explanation",
    "recommendations",
}


# ---------------------------------------------------------------------------
# Prompt loader & formatter
# ---------------------------------------------------------------------------

def resolve_prompt_file(version: str) -> Path:
    """Resolve a prompt version string (e.g. 'v1', 'v3_structured') to a file path."""
    filename = _PROMPT_VERSION_MAP.get(version.lower().strip())
    if not filename:
        valid_opts = list(_PROMPT_VERSION_MAP.keys())
        raise ValueError(f"Unknown prompt version '{version}'. Supported options: {valid_opts}")
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found at {path}")
    return path


def load_prompt_template(version: str) -> str:
    """Load the raw text of the chosen prompt version."""
    path = resolve_prompt_file(version)
    with path.open(encoding="utf-8") as fh:
        return fh.read()


def format_prompt(
    version: str,
    job_description: str,
    resume_text: str,
    skill_table: Union[List[Dict[str, Any]], Dict[str, Any], List[Any]],
) -> str:
    """Inject inputs into the chosen prompt template.

    Parameters
    ----------
    version : str
        Prompt version ('v1', 'v2', 'v3', etc.)
    job_description : str
        Raw or preprocessed JD text.
    resume_text : str
        Resume text (e.g., core_text from preprocessing).
    skill_table : list[dict] | dict
        Skill analysis table from /analyze or /match.
    """
    template = load_prompt_template(version)

    # Format skill table nicely as human-readable JSON if it's a list/dict
    if isinstance(skill_table, (list, dict)):
        skills_str = json.dumps(skill_table, indent=2)
    else:
        skills_str = str(skill_table)

    return template.format(
        job_description=job_description.strip(),
        resume_text=resume_text.strip(),
        skill_table=skills_str.strip(),
    )


# ---------------------------------------------------------------------------
# LLM Client (OpenAI-compatible HTTP chat completions)
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: float = 30.0,
) -> str:
    """Send prompt to the configured LLM endpoint via OpenAI-compatible Chat Completions API.

    Can connect to any provider exposing an OpenAI-compatible /chat/completions endpoint
    (e.g., OpenAI, Groq, Ollama, OpenRouter, Together, vLLM, DeepSeek).
    """
    key = api_key if api_key is not None else settings.llm_api_key
    target_model = model if model is not None else settings.llm_model
    target_base = (base_url if base_url is not None else settings.llm_base_url).rstrip("/")

    if not key:
        raise ValueError(
            "LLM_API_KEY is not set. Please set LLM_API_KEY in your environment or .env file."
        )

    endpoint = f"{target_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": target_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
    }

    try:
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"LLM API returned HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to communicate with LLM provider: {exc}") from exc


# ---------------------------------------------------------------------------
# Output Parsing and Format Validation
# ---------------------------------------------------------------------------

def extract_json_block(text: str) -> str:
    """Extract raw JSON text if enclosed in markdown code fences or surrounded by prose."""
    text = text.strip()

    # Case 1: Markdown codeblock ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Case 2: Extract between the outermost curly braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    return text


def parse_and_validate_feedback(raw_text: str) -> Tuple[Dict[str, Any], bool]:
    """Parse the LLM response into structured JSON and validate required schema keys.

    Returns
    -------
    tuple of (feedback_dict, format_valid: bool)
    If format validation fails, format_valid is False and default values are populated.
    """
    cleaned = extract_json_block(raw_text)

    try:
        parsed = json.loads(cleaned)
    except Exception:
        # JSON syntax failure
        return (
            {
                "matched_skills": [],
                "gaps": [],
                "rewritten_bullet": "",
                "explanation": "Format Error: Output could not be parsed as valid JSON.",
                "recommendations": [],
            },
            False,
        )

    if not isinstance(parsed, dict):
        return (
            {
                "matched_skills": [],
                "gaps": [],
                "rewritten_bullet": "",
                "explanation": "Format Error: Root JSON entity must be an object/dict.",
                "recommendations": [],
            },
            False,
        )

    # Validate required keys
    has_all_keys = _REQUIRED_KEYS.issubset(parsed.keys())

    # Validate types
    valid_types = (
        isinstance(parsed.get("matched_skills"), list)
        and isinstance(parsed.get("gaps"), list)
        and isinstance(parsed.get("rewritten_bullet"), str)
        and isinstance(parsed.get("explanation"), str)
        and isinstance(parsed.get("recommendations"), list)
    )

    format_valid = has_all_keys and valid_types

    # Ensure non-None fallback values
    normalized = {
        "matched_skills": [str(s) for s in parsed.get("matched_skills", []) if s]
        if isinstance(parsed.get("matched_skills"), list)
        else [],
        "gaps": [str(g) for g in parsed.get("gaps", []) if g]
        if isinstance(parsed.get("gaps"), list)
        else [],
        "rewritten_bullet": str(parsed.get("rewritten_bullet") or ""),
        "explanation": str(parsed.get("explanation") or ""),
        "recommendations": [str(r) for r in parsed.get("recommendations", []) if r]
        if isinstance(parsed.get("recommendations"), list)
        else [],
    }

    return normalized, format_valid


# ---------------------------------------------------------------------------
# Public Service API
# ---------------------------------------------------------------------------

def generate_feedback(
    job_description: str,
    resume_text: str,
    skill_table: Union[List[Dict[str, Any]], Dict[str, Any], List[Any]],
    prompt_version: str = "v3",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate structured, rule-checked LLM feedback for a resume against a JD.

    Returns
    -------
    dict with keys:
        "prompt_version": canonical prompt identifier
        "format_valid": bool (True if the LLM output obeyed the JSON structure)
        "feedback": dict with matched_skills, gaps, rewritten_bullet, explanation, recommendations
        "raw_response": str (exact text returned by the model)
        "error": str | None (error message if LLM call or formatting failed)
    """
    canonical_version = "v3"
    for k, v in _PROMPT_VERSION_MAP.items():
        if prompt_version.lower().strip() == k:
            canonical_version = k.split("_")[0]
            break

    prompt = format_prompt(canonical_version, job_description, resume_text, skill_table)

    try:
        raw_response = call_llm(
            prompt=prompt,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
    except Exception as exc:
        return {
            "prompt_version": canonical_version,
            "format_valid": False,
            "feedback": {
                "matched_skills": [],
                "gaps": [],
                "rewritten_bullet": "",
                "explanation": "",
                "recommendations": [],
            },
            "raw_response": "",
            "error": str(exc),
        }

    feedback, format_valid = parse_and_validate_feedback(raw_response)

    return {
        "prompt_version": canonical_version,
        "format_valid": format_valid,
        "feedback": feedback,
        "raw_response": raw_response,
        "error": None if format_valid else "Response failed format validation.",
    }
