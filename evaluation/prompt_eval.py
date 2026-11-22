"""
evaluation/prompt_eval.py – Comparative Prompt Engineering Evaluation.

Runs 10 realistic benchmark test cases across all three prompt versions:
  - v1_baseline: simple role instruction
  - v2_criteria: explicit criteria for relevance, evidence, clarity, grounding
  - v3_structured: rigid, type-enforced schema

Saves results to `evaluation/prompt_eval_results.csv` with columns:
  - test_id
  - scenario
  - prompt_version
  - format_pass (PASS / FAIL)
  - grounding_unsupported_claims (count of unsupported claims)
  - relevance (1-5 rating, editable)
  - consistency (1-5 rating, editable)
  - clarity (1-5 rating, editable)
  - skill_gap_accuracy (1-5 rating, editable)
  - matched_skills_count
  - gaps_count
  - raw_output

USAGE
-----
  # 1. Run with mock LLM outputs (no API key needed, tests the evaluation harness):
  python -m evaluation.prompt_eval --mock

  # 2. Run with live configured LLM provider:
  python -m evaluation.prompt_eval --live
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.config import settings
from backend.app.services.llm_feedback import (
    call_llm,
    format_prompt,
    parse_and_validate_feedback,
)

_OUT_CSV = Path(__file__).resolve().parent / "prompt_eval_results.csv"
_OUT_CSV_MOCK = Path(__file__).resolve().parent / "prompt_eval_results_MOCK.csv"

# ---------------------------------------------------------------------------
# 10 Diverse Benchmark Test Cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "TC-01",
        "scenario": "Python ML Engineer (Strong Match)",
        "job_description": "Senior ML Engineer needed. Must have strong Python, PyTorch, Scikit-learn, and experience deploying Docker containers on AWS.",
        "resume_text": "Machine Learning Engineer with 4 years experience. Built computer vision pipelines using Python and PyTorch. Trained classification models with scikit-learn. Containerized services using Docker and deployed on AWS EC2.",
        "skill_table": [
            {"skill": "python", "status": "Matched"},
            {"skill": "pytorch", "status": "Matched"},
            {"skill": "scikit-learn", "status": "Matched"},
            {"skill": "docker", "status": "Matched"},
            {"skill": "aws", "status": "Matched"},
        ],
    },
    {
        "id": "TC-02",
        "scenario": "Frontend Dev Applying for Fullstack (Missing Backend)",
        "job_description": "Fullstack Web Developer required. Experience in React, TypeScript, Node.js, Express, and PostgreSQL database design.",
        "resume_text": "Frontend Developer with 3 years building responsive UIs in React and TypeScript. Managed global state with Redux. Styled applications using Tailwind CSS. Basic exposure to SQL queries.",
        "skill_table": [
            {"skill": "react", "status": "Matched"},
            {"skill": "typescript", "status": "Matched"},
            {"skill": "nodejs", "status": "Missing"},
            {"skill": "express", "status": "Missing"},
            {"skill": "postgresql", "status": "Weak"},
        ],
    },
    {
        "id": "TC-03",
        "scenario": "Data Analyst Seeking Data Scientist Role (Lacks DL)",
        "job_description": "Data Scientist required for predictive analytics. Must have Python, SQL, Statistics, Deep Learning, and TensorFlow.",
        "resume_text": "Data Analyst with 2 years conducting EDA using Python and Pandas. Wrote complex SQL queries for business intelligence dashboards. Applied linear regression and hypothesis testing.",
        "skill_table": [
            {"skill": "python", "status": "Matched"},
            {"skill": "sql", "status": "Matched"},
            {"skill": "statistics", "status": "Matched"},
            {"skill": "deep learning", "status": "Missing"},
            {"skill": "tensorflow", "status": "Missing"},
        ],
    },
    {
        "id": "TC-04",
        "scenario": "DevOps Engineer with Missing Orchestration",
        "job_description": "DevOps Specialist. Requirements: Linux administration, Bash, Docker, Kubernetes, CI/CD pipelines, and Terraform.",
        "resume_text": "Systems Administrator maintaining Linux servers. Wrote Bash scripts for automated log rotation. Packaged microservices in Docker and set up GitHub Actions CI/CD workflows.",
        "skill_table": [
            {"skill": "linux", "status": "Matched"},
            {"skill": "bash", "status": "Matched"},
            {"skill": "docker", "status": "Matched"},
            {"skill": "ci/cd", "status": "Matched"},
            {"skill": "kubernetes", "status": "Missing"},
        ],
    },
    {
        "id": "TC-05",
        "scenario": "University Student with Academic Projects Only",
        "job_description": "Junior Software Developer. Looking for foundational knowledge of Java, OOP, Data Structures, Git, and Unit Testing.",
        "resume_text": "Computer Science student (Graduating 2024). Completed coursework in Data Structures, Algorithms, and Object-Oriented Programming using Java. Maintained academic projects on Git repository.",
        "skill_table": [
            {"skill": "java", "status": "Matched"},
            {"skill": "object-oriented programming", "status": "Matched"},
            {"skill": "data structures", "status": "Matched"},
            {"skill": "git", "status": "Matched"},
            {"skill": "unit testing", "status": "Missing"},
        ],
    },
    {
        "id": "TC-06",
        "scenario": "Career Switcher: QA Engineer Transitioning to Dev",
        "job_description": "Backend Python Engineer. Responsibilities include building REST APIs with FastAPI, PostgreSQL, and Redis caching.",
        "resume_text": "QA Automation Engineer with 3 years writing automated integration tests in Python. Familiar with REST API protocols and PostgreSQL databases. Used Redis in test environments.",
        "skill_table": [
            {"skill": "python", "status": "Matched"},
            {"skill": "rest api", "status": "Matched"},
            {"skill": "postgresql", "status": "Matched"},
            {"skill": "fastapi", "status": "Missing"},
            {"skill": "redis", "status": "Weak"},
        ],
    },
    {
        "id": "TC-07",
        "scenario": "Senior Architect Missing Specific Niche Framework",
        "job_description": "Lead Golang Developer. Deep expertise in Go, Microservices, Kafka event streaming, and Kubernetes cluster administration.",
        "resume_text": "Senior Distributed Systems Engineer with 8 years building high-throughput microservices in Go and C++. Architected event-driven architectures and managed production Kubernetes clusters.",
        "skill_table": [
            {"skill": "go", "status": "Matched"},
            {"skill": "microservices", "status": "Matched"},
            {"skill": "kubernetes", "status": "Matched"},
            {"skill": "kafka", "status": "Missing"},
        ],
    },
    {
        "id": "TC-08",
        "scenario": "Keyword Stuffed Resume (Lacks Depth)",
        "job_description": "Cloud Data Engineer. Required: Python, SQL, Apache Spark, Snowflake, AWS, and ETL pipeline development.",
        "resume_text": "Technical Skills: Python, SQL, Apache Spark, Snowflake, AWS, ETL. Experience: Worked on various projects utilizing cloud data tools and SQL queries.",
        "skill_table": [
            {"skill": "python", "status": "Weak"},
            {"skill": "sql", "status": "Weak"},
            {"skill": "apache spark", "status": "Weak"},
            {"skill": "aws", "status": "Weak"},
        ],
    },
    {
        "id": "TC-09",
        "scenario": "Minimal Ambiguous Resume",
        "job_description": "Data Scientist with experience in Machine Learning, Pandas, and Model Evaluation.",
        "resume_text": "Tech enthusiast who completed online analytics modules.",
        "skill_table": [
            {"skill": "machine learning", "status": "Missing"},
            {"skill": "pandas", "status": "Missing"},
            {"skill": "model evaluation", "status": "Missing"},
        ],
    },
    {
        "id": "TC-10",
        "scenario": "Completely Irrelevant Role Applicant",
        "job_description": "Senior Backend Developer. Experience in Java, Spring Boot, Microservices, and PostgreSQL.",
        "resume_text": "Sales Executive with 5 years achieving revenue targets, client negotiations, cold calling, and CRM management.",
        "skill_table": [
            {"skill": "java", "status": "Missing"},
            {"skill": "microservices", "status": "Missing"},
            {"skill": "postgresql", "status": "Missing"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Mock Response Generator (for offline evaluation testing)
# ---------------------------------------------------------------------------

def _mock_llm_response(prompt_version: str, tc: Dict[str, Any]) -> str:
    """Generate deterministic mock responses conforming or diverging according to version."""
    matched = [s["skill"] for s in tc["skill_table"] if s["status"] == "Matched"]
    missing = [s["skill"] for s in tc["skill_table"] if s["status"] == "Missing"]

    if prompt_version == "v1":
        # Baseline prompt might sometimes include conversational prefix or markdown
        resp = {
            "matched_skills": matched,
            "gaps": missing,
            "rewritten_bullet": f"Applied verified skills in {', '.join(matched) if matched else 'core tasks'}.",
            "explanation": f"Review for {tc['scenario']}. Found {len(matched)} matched skills.",
            "recommendations": ["Clarify project details and evidence missing competencies."],
        }
        return f"Here is the evaluation review:\n```json\n{json.dumps(resp, indent=2)}\n```"

    elif prompt_version == "v2":
        resp = {
            "matched_skills": matched,
            "gaps": missing,
            "rewritten_bullet": f"Executed responsibilities using {matched[0] if matched else 'documented methods'} without adding ungrounded metrics.",
            "explanation": f"Criteria assessment: Relevance matches candidate profile. Evidence verified for {len(matched)} skills.",
            "recommendations": [
                f"Address evident gap in {missing[0]}" if missing else "Continue strengthening technical portfolio.",
                "Maintain clear focus on verifiable impact.",
            ],
        }
        return json.dumps(resp, indent=2)

    else:  # v3
        resp = {
            "matched_skills": matched,
            "gaps": missing,
            "rewritten_bullet": f"Strengthened bullet focusing on demonstrated usage of {', '.join(matched[:2]) if matched else 'stated tasks'}.",
            "explanation": f"Structured audit: The candidate demonstrates {len(matched)} required competencies with {len(missing)} unfilled requirement gaps.",
            "recommendations": [
                f"Gain practical project exposure in {missing[0]}" if missing else "Document advanced architectures.",
                "Ensure all resume bullets contain specific technical actions.",
            ],
        }
        return json.dumps(resp)


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def run_evaluation(live: bool = False) -> None:
    """Execute all test cases across prompt versions and write comparative CSV."""
    versions = ["v1", "v2", "v3"]
    rows: List[Dict[str, Any]] = []

    # Print a prominent banner when running in mock mode so the output cannot
    # be confused with real LLM results.
    _BANNER = "=" * 60
    if not live:
        print(_BANNER)
        print("  MOCK OUTPUT — NO LLM WAS CALLED")
        print("  Results are deterministic mock responses, not real LLM output.")
        print("  CSV will be saved as prompt_eval_results_MOCK.csv.")
        print(_BANNER)

    print(f"Starting prompt evaluation ({'LIVE LLM' if live else 'MOCK HARNESS'})...")

    for tc in TEST_CASES:
        for v in versions:
            prompt = format_prompt(v, tc["job_description"], tc["resume_text"], tc["skill_table"])

            if live:
                try:
                    raw_output = call_llm(prompt)
                except Exception as exc:
                    raw_output = f"ERROR: {exc}"
            else:
                raw_output = _mock_llm_response(v, tc)

            parsed, format_valid = parse_and_validate_feedback(raw_output)

            # Heuristic grounding check: count hallucinated digits/percentages in rewritten bullet
            unsupported_claims_count = len(re.findall(r"\b\d+%\b|\$\d+", parsed["rewritten_bullet"]))

            # Default ratings for user review
            rows.append({
                "test_id": tc["id"],
                "scenario": tc["scenario"],
                "prompt_version": v,
                "format_pass": "PASS" if format_valid else "FAIL",
                "grounding_unsupported_claims": unsupported_claims_count,
                "relevance": 5 if format_valid else 1,
                "consistency": 5 if format_valid else 2,
                "clarity": 5 if format_valid else 2,
                "skill_gap_accuracy": 5 if len(parsed["gaps"]) > 0 or "TC-01" in tc["id"] else 3,
                "matched_skills_count": len(parsed["matched_skills"]),
                "gaps_count": len(parsed["gaps"]),
                "raw_output": raw_output.replace("\n", " ")[:300],
            })

    # Save to CSV
    fieldnames = [
        "test_id",
        "scenario",
        "prompt_version",
        "format_pass",
        "grounding_unsupported_claims",
        "relevance",
        "consistency",
        "clarity",
        "skill_gap_accuracy",
        "matched_skills_count",
        "gaps_count",
        "raw_output",
    ]

    # Choose output CSV filename: mock runs get a distinct name so they cannot
    # be accidentally treated as results from a real LLM run.
    out_csv = _OUT_CSV if live else _OUT_CSV_MOCK

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not live:
        _BANNER = "=" * 60
        print(_BANNER)
        print("  MOCK OUTPUT — NO LLM WAS CALLED")
        print(_BANNER)
    print(f"Evaluation complete! Saved {len(rows)} evaluations across {len(TEST_CASES)} scenarios to:\n{out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prompt versions v1, v2, v3.")
    parser.add_argument("--live", action="store_true", help="Call live LLM provider using settings.")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock evaluation harness (default).")
    args = parser.parse_args()

    use_live = args.live and bool(settings.llm_api_key)
    run_evaluation(live=use_live)
