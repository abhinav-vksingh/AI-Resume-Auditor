"""
frontend/app.py – Streamlit User Interface for AI Resume Auditor.

A clean, friendly interface designed for students and recruiters to:
  1. Check how well a resume matches a job description in plain language.
  2. Test whether the scoring process treats candidates fairly.
  3. Get concrete, grounded advice on how to improve the resume for this job.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------------------------

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"

st.set_page_config(
    page_title="AI Resume Auditor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom Styling for Chips, Cards & Visual Polish
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Metric card styling */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-subtext {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Score callout */
    .score-banner {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .score-number {
        font-size: 3.2rem;
        font-weight: 800;
        color: #1D4ED8;
        line-height: 1;
    }
    .score-label {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E40AF;
        margin-top: 6px;
    }
    .score-footnote {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Skill chips */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    .chip {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .chip-matched {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .chip-weak {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .chip-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }

    /* Generic content card */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Backend API Client Helpers
# ---------------------------------------------------------------------------

def check_backend_health() -> Tuple[bool, str]:
    """Check if the backend is running and healthy."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=3.0)
        if resp.status_code == 200:
            return True, resp.json().get("version", "unknown")
        return False, f"Server returned status {resp.status_code}"
    except Exception:
        return False, "Could not reach the server"


def call_analyze_api(file_bytes: bytes, filename: str, job_description: str) -> Dict[str, Any]:
    """Call POST /analyze to run the full scoring pipeline."""
    files = {"file": (filename, file_bytes, "application/pdf")}
    data = {"job_description": job_description}
    resp = httpx.post(f"{BACKEND_URL}/analyze", files=files, data=data, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def call_bias_audit_api(
    original_text: str, job_description: str, attribute: str, n_pairs: int, mode: str = "redacted"
) -> Dict[str, Any]:
    """Call POST /bias-audit to test whether swapped identities change scores."""
    payload = {
        "original_text": original_text,
        "job_description": job_description,
        "attribute": attribute,
        "n_pairs": n_pairs,
        "mode": mode,
    }
    resp = httpx.post(f"{BACKEND_URL}/bias-audit", json=payload, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def call_feedback_api(
    job_description: str,
    resume_text: str,
    skill_table: List[Dict[str, str]],
    prompt_version: str,
) -> Dict[str, Any]:
    """Call POST /feedback for qualitative suggestions."""
    payload = {
        "job_description": job_description,
        "resume_text": resume_text,
        "skill_table": skill_table,
    }
    resp = httpx.post(
        f"{BACKEND_URL}/feedback",
        params={"prompt_version": prompt_version},
        json=payload,
        timeout=45.0,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Header & Backend Status Check
# ---------------------------------------------------------------------------

backend_online, backend_msg = check_backend_health()

st.title("📄 AI Resume Auditor")
st.markdown(
    "Upload your resume and a job description. We show how well they match, "
    "what is missing, and whether the scoring is fair."
)

if not backend_online:
    st.error(
        "**The server is not currently running.**\n\n"
        "**Next step:** Start the backend in your terminal with:\n"
        "```bash\nuvicorn backend.app.main:app --reload\n```\n"
        "Then refresh this page."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Step 1: Input Card (Side by Side)
# ---------------------------------------------------------------------------

st.markdown("### Step 1: Add Your Documents")

with st.container():
    col_resume, col_jd = st.columns(2, gap="medium")

    with col_resume:
        st.markdown("**Your Resume (PDF)**")
        uploaded_file = st.file_uploader(
            "Choose a PDF resume",
            type=["pdf"],
            label_visibility="collapsed",
            help="Upload your resume as a standard PDF file.",
        )
        if uploaded_file:
            st.caption(f"Selected: **{uploaded_file.name}**")
        else:
            st.caption("Please upload a PDF document.")

    with col_jd:
        st.markdown("**Target Job Description**")
        default_jd = (
            "We are looking for a Senior Python Developer with strong Machine Learning experience. "
            "Must be proficient in Python, PyTorch, Scikit-learn, Docker, and REST APIs. "
            "Familiarity with Kubernetes and AWS is a plus."
        )
        job_description = st.text_area(
            "Job requirements text",
            value=default_jd,
            height=180,
            label_visibility="collapsed",
            placeholder="Paste the requirements and responsibilities from the job posting here...",
        )

    check_button = st.button("🔍 Check my resume", type="primary", use_container_width=True)

if check_button:
    if not uploaded_file:
        st.warning(
            "**No resume uploaded.**\n\n"
            "**Next step:** Choose a PDF resume on the left, then click 'Check my resume'."
        )
    elif not job_description.strip():
        st.warning(
            "**The job description is empty.**\n\n"
            "**Next step:** Paste the job requirements into the box on the right."
        )
    else:
        with st.spinner("Reading your resume..."):
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            try:
                analyze_res = call_analyze_api(file_bytes, filename, job_description)
                st.session_state["original_text"] = analyze_res["original_text"]
                st.session_state["analyze_result"] = analyze_res
                st.session_state["job_description"] = job_description
                st.session_state["resume_filename"] = filename
                st.session_state["analyzed"] = True
                # Reset previous audit & feedback when a new resume is analyzed
                st.session_state.pop("audit_result", None)
                st.session_state.pop("feedback_result", None)
                st.success("Resume checked successfully! View your results below.")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400:
                    st.error(
                        "**We could not read usable text from this PDF.**\n\n"
                        "**Next step:** Please ensure your resume contains searchable text, not scanned images."
                    )
                elif exc.response.status_code == 422:
                    st.error(
                        "**The job description does not contain recognisable skills.**\n\n"
                        "**Next step:** Add specific tools, skills, or programming languages to the job description."
                    )
                else:
                    st.error(
                        "**An unexpected problem occurred while reading the resume.**\n\n"
                        "**Next step:** Please verify the backend is healthy and try again."
                    )
            except Exception:
                st.error(
                    "**Connection lost while talking to the server.**\n\n"
                    "**Next step:** Check that the backend server is still running in your terminal."
                )

st.markdown("---")


# ---------------------------------------------------------------------------
# Results Tabs
# ---------------------------------------------------------------------------

tab_score, tab_fair, tab_improve = st.tabs([
    "Your Match Score",
    "Is the Scoring Fair?",
    "How to Improve",
])


# ===========================================================================
# TAB 1: Your Match Score
# ===========================================================================

with tab_score:
    if not st.session_state.get("analyzed"):
        st.info("Check your resume above to see how well it matches this job.")
    else:
        res = st.session_state["analyze_result"]
        final_score = float(res["final_score"])
        comps = res["components"]
        sem_score = float(comps["semantic"])
        skill_score = float(comps["skill_overlap"])
        weights = comps.get("weights", {})
        w_sem = weights.get("semantic", 0.6)
        w_skill = weights.get("skill_overlap", 0.4)

        # Plain language match label
        if final_score >= 75:
            score_label = "Strong match"
        elif final_score >= 50:
            score_label = "Good match, room to improve"
        else:
            score_label = "Weak match"

        # Large Score Banner
        st.markdown(
            f"""
            <div class="score-banner">
                <div class="score-number">{final_score:.0f} / 100</div>
                <div class="score-label">{score_label}</div>
                <div class="score-footnote">A guide, not a hiring decision.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Two friendly sub-cards
        col_c1, col_c2 = st.columns(2, gap="medium")
        with col_c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">How closely the meaning matches the job</div>
                    <div class="metric-value">{sem_score:.0f} / 100</div>
                    <div class="metric-subtext">Compares your experience and projects against the requirements.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(sem_score / 100.0, 0.0), 1.0))

        with col_c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">How many required skills you have</div>
                    <div class="metric-value">{skill_score:.0f} / 100</div>
                    <div class="metric-subtext">Direct evidence of specific tools and technologies mentioned in the job.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(skill_score / 100.0, 0.0), 1.0))

        st.markdown("<br>", unsafe_allow_html=True)

        # Skills as Colored Chips
        st.subheader("Skills Overview")
        skills = res.get("skills", {})
        matched_list = skills.get("matched", [])
        weak_list = skills.get("weak", [])
        missing_list = skills.get("missing", [])

        col_s1, col_s2, col_s3 = st.columns(3, gap="medium")

        with col_s1:
            st.markdown(f"**You have this ({len(matched_list)})**")
            if matched_list:
                chips_html = "".join(f'<span class="chip chip-matched">{s}</span>' for s in matched_list)
                st.markdown(f'<div class="chip-container">{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.caption("No directly verified skills found.")

        with col_s2:
            st.markdown(f"**Mentioned only briefly ({len(weak_list)})**")
            if weak_list:
                chips_html = "".join(f'<span class="chip chip-weak">{s}</span>' for s in weak_list)
                st.markdown(f'<div class="chip-container">{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.caption("None mentioned only once.")

        with col_s3:
            st.markdown(f"**Missing from your resume ({len(missing_list)})**")
            if missing_list:
                chips_html = "".join(f'<span class="chip chip-missing">{s}</span>' for s in missing_list)
                st.markdown(f'<div class="chip-container">{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.caption("None missing — all required skills are present!")

        st.markdown("---")

        # Why you got this score
        st.subheader("Why you got this score")
        expl = res.get("explanation", {})
        pos_factors = expl.get("positive_factors", [])
        neg_factors = expl.get("negative_factors", [])

        col_e1, col_e2 = st.columns(2, gap="medium")
        with col_e1:
            st.markdown("#### What helped")
            if pos_factors:
                for f in pos_factors:
                    st.markdown(f"✔️ {f}")
            else:
                st.caption("No strong positive points recorded.")

        with col_e2:
            st.markdown("#### What held it back")
            if neg_factors:
                for f in neg_factors:
                    st.markdown(f"⚠️ {f}")
            else:
                st.caption("No significant gaps found.")

        # Technical Expander
        with st.expander("🔬 For technical readers (formula, weights, and component details)"):
            st.markdown(
                f"""
                - **Formula:** `Final Score = ({w_sem:.2f} × Semantic Similarity) + ({w_skill:.2f} × Skill Overlap)`
                - **Semantic Component:** `{sem_score:.2f} / 100` (all-MiniLM-L6-v2 dense embedding cosine similarity)
                - **Skill Overlap Component:** `{skill_score:.2f} / 100` (formula: `(Matched + 0.5 × Weak) / Total Required × 100`)
                - **Scoring Architecture:** Purely mathematical arithmetic. The language model never computes or adjusts scores.
                """
            )


# ===========================================================================
# TAB 2: Is the Scoring Fair?
# ===========================================================================

with tab_fair:
    st.subheader("Is the Scoring Fair?")
    st.markdown(
        "We test whether changing only a name or a university changes the score for the same resume."
    )

    if not st.session_state.get("analyzed"):
        st.info("Check your resume first to run a fairness test on it.")
    else:
        col_f1, col_f2 = st.columns([1, 1], gap="medium")

        with col_f1:
            audit_attr = st.selectbox(
                "What should we change?",
                options=["name", "university"],
                format_func=lambda x: "The candidate's name" if x == "name" else "The university",
            )

        with col_f2:
            n_pairs = st.slider(
                "How many examples?",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of different identity pairs to swap and test.",
            )

        run_fairness_btn = st.button("🧪 Test for fairness", type="secondary", use_container_width=True)

        if run_fairness_btn:
            with st.spinner("Running the fairness test across both modes..."):
                try:
                    res_redacted = call_bias_audit_api(
                        original_text=st.session_state["original_text"],
                        job_description=st.session_state["job_description"],
                        attribute=audit_attr,
                        n_pairs=n_pairs,
                        mode="redacted",
                    )
                    res_raw = call_bias_audit_api(
                        original_text=st.session_state["original_text"],
                        job_description=st.session_state["job_description"],
                        attribute=audit_attr,
                        n_pairs=n_pairs,
                        mode="raw",
                    )
                    st.session_state["audit_redacted"] = res_redacted
                    st.session_state["audit_raw"] = res_raw
                    st.success("Fairness test complete!")
                except Exception as exc:
                    st.error(f"Could not complete the fairness test: {exc}")

        if "audit_redacted" in st.session_state and "audit_raw" in st.session_state:
            red_res = st.session_state["audit_redacted"]
            raw_res = st.session_state["audit_raw"]

            red_delta = abs(float(red_res.get("mean_delta", 0.0)))
            raw_delta = abs(float(raw_res.get("mean_delta", 0.0)))
            red_sig = red_res.get("significant", False)
            raw_sig = raw_res.get("significant", False)

            col_left, col_right = st.columns(2, gap="medium")

            # Helper to build pair dataframe
            def _build_friendly_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows)
                rename_map = {
                    "identity_a": "Name or university A",
                    "identity_b": "Name or university B",
                    "score_a": "Score A",
                    "score_b": "Score B",
                    "delta": "Difference",
                }
                df_friendly = df.rename(columns=rename_map)
                wanted = ["Name or university A", "Name or university B", "Score A", "Score B", "Difference"]
                cols = [c for c in wanted if c in df_friendly.columns]
                return df_friendly[cols] if cols else df_friendly

            # LEFT COLUMN: Names hidden first
            with col_left:
                st.markdown("### Names hidden first (how real screening works)")
                st.markdown(
                    f"""
                    <div class="card">
                        <p style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-bottom: 4px;">
                            Average change: {red_delta:.2f} points
                        </p>
                        <p style="font-size: 0.95rem; color: #334155; margin-bottom: 4px;">
                            {'The model reacted to this change. Treat it as a possible bias.' if red_sig else 'No reliable difference found in this test.'}
                        </p>
                        <p style="font-size: 0.82rem; color: #64748B; margin-bottom: 0;">
                            This is a small test of {n_pairs} examples. Different examples can give slightly different results. See the full study below for the main result.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                df_red_tbl = _build_friendly_table(red_res.get("raw_rows", []))
                if not df_red_tbl.empty:
                    st.dataframe(df_red_tbl, use_container_width=True)
                st.caption("ℹ️ The system removes names before scoring, so nothing can change.")

            # RIGHT COLUMN: Names shown to the model
            with col_right:
                st.markdown("### Names shown to the model (stress test)")
                st.markdown(
                    f"""
                    <div class="card">
                        <p style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-bottom: 4px;">
                            Average change: {raw_delta:.2f} points
                        </p>
                        <p style="font-size: 0.95rem; color: #334155; margin-bottom: 4px;">
                            {'The model reacted to this change. Treat it as a possible bias.' if raw_sig else 'No reliable difference found in this test.'}
                        </p>
                        <p style="font-size: 0.82rem; color: #64748B; margin-bottom: 0;">
                            This is a small test of {n_pairs} examples. Different examples can give slightly different results. See the full study below for the main result.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                df_raw_tbl = _build_friendly_table(raw_res.get("raw_rows", []))
                if not df_raw_tbl.empty:
                    st.dataframe(df_raw_tbl, use_container_width=True)
                st.caption("ℹ️ Here we let the model see the names to check whether it reacts to them.")

            # Summary sentence under both columns
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                f"**Overall comparison:** With names visible the score moved by **{raw_delta:.2f}** points on average; "
                f"with names hidden it moved by **{red_delta:.2f}**."
            )

            # Technical Expander
            with st.expander("🔬 For technical readers (statistics, p-values, and charts)"):
                p_red = red_res.get("p_value")
                p_red_str = f"{p_red:.4f}" if (p_red is not None and pd.notna(p_red)) else "N/A"
                std_red = red_res.get("std_delta")
                std_red_str = f"{std_red:.4f}" if (std_red is not None and pd.notna(std_red)) else "0.0000"

                p_raw = raw_res.get("p_value")
                p_raw_str = f"{p_raw:.4f}" if (p_raw is not None and pd.notna(p_raw)) else "N/A"
                std_raw = raw_res.get("std_delta")
                std_raw_str = f"{std_raw:.4f}" if (std_raw is not None and pd.notna(std_raw)) else "0.0000"

                st.markdown(
                    f"""
                    **Redacted Mode (Production):**
                    - Mean Delta: `{red_res.get('mean_delta'):+.4f}`, Std: `{std_red_str}`, p-value: `{p_red_str}` (Significant: `{red_sig}`)
                    
                    **Raw Mode (Stress Test):**
                    - Mean Delta: `{raw_res.get('mean_delta'):+.4f}`, Std: `{std_raw_str}`, p-value: `{p_raw_str}` (Significant: `{raw_sig}`)

                    - **Pairs Tested:** `{red_res.get('n_observations')}`
                    - **Disclaimer:** {red_res.get('disclaimer')}
                    """
                )

                # Show distribution charts if available
                bar_chart_path = EVAL_DIR / "results" / "bias_bar.png"
                if not bar_chart_path.exists():
                    bar_chart_path = EVAL_DIR / "bias_bar.png"

                box_chart_path = EVAL_DIR / "results" / "bias_box.png"
                if not box_chart_path.exists():
                    box_chart_path = EVAL_DIR / "bias_box.png"

                if bar_chart_path.exists() or box_chart_path.exists():
                    st.markdown("**Benchmark Distribution Charts:**")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        if bar_chart_path.exists():
                            st.image(str(bar_chart_path), caption="Mean Delta across 30 Resumes")
                    with col_p2:
                        if box_chart_path.exists():
                            st.image(str(box_chart_path), caption="Delta Distribution")

    # FULL STUDY SECTION
    st.markdown("---")
    st.markdown("### Full study on 30 real resumes")
    st.caption("Saved results from the offline benchmark evaluation across 30 deduplicated resumes from the Kaggle dataset.")

    results_csv_path = EVAL_DIR / "results" / "bias_results.csv"
    if results_csv_path.exists():
        try:
            df_study = pd.read_csv(results_csv_path)
            # Map columns to simple friendly names
            display_rows = []
            for _, r in df_study.iterrows():
                attr_name = "Candidate name" if r["attribute"] == "name" else "University prestige"
                mode_name = "Names shown" if r["mode"] == "raw" else "Names hidden"
                mean_d = abs(float(r["mean_delta"]))
                n_res = int(r["n_resumes"]) if "n_resumes" in r and pd.notna(r["n_resumes"]) else 30
                is_sig = bool(r["significant"]) if "significant" in r else False

                display_rows.append({
                    "Name or university": attr_name,
                    "Names hidden or shown": mode_name,
                    "Average change in points": f"{mean_d:.2f}",
                    "Number of resumes": n_res,
                    "Reliable difference?": "Yes" if is_sig else "No",
                })
            df_study_friendly = pd.DataFrame(display_rows)
            st.dataframe(df_study_friendly, use_container_width=True)
            st.caption("📌 *Note: These are pre-computed, saved results from the full 30-resume evaluation study, not the live test above.*")
        except Exception as e:
            st.error(f"Could not display saved study results: {e}")
    else:
        st.info("The saved study results file (`evaluation/results/bias_results.csv`) was not found.")


# ===========================================================================
# TAB 3: How to Improve
# ===========================================================================

with tab_improve:
    st.subheader("Suggestions for this job")
    st.markdown("Get concrete ideas on how to strengthen your resume specifically for these requirements.")

    if not st.session_state.get("analyzed"):
        st.info("Check your resume first to unlock personalized suggestions.")
    else:
        # Prompt version selector hidden in expander
        with st.expander("⚙️ Advanced options"):
            prompt_choice = st.selectbox(
                "Feedback prompt strategy",
                options=["v3", "v2", "v1"],
                index=0,
                format_func=lambda x: {
                    "v3": "v3 (Structured, recommended)",
                    "v2": "v2 (Criteria-focused)",
                    "v1": "v1 (Simple review)",
                }[x],
                help="Controls the prompt structure sent to the language model. v3 provides the most rigid format.",
            )

        get_suggestions_btn = st.button("💡 Get suggestions", type="primary", use_container_width=True)

        if get_suggestions_btn:
            with st.spinner("Generating suggestions for your resume..."):
                res = st.session_state["analyze_result"]
                skills_table_flat = []
                for s in res["skills"].get("matched", []):
                    skills_table_flat.append({"skill": s, "status": "Matched"})
                for s in res["skills"].get("weak", []):
                    skills_table_flat.append({"skill": s, "status": "Weak"})
                for s in res["skills"].get("missing", []):
                    skills_table_flat.append({"skill": s, "status": "Missing"})

                try:
                    fb_res = call_feedback_api(
                        job_description=st.session_state["job_description"],
                        resume_text=st.session_state["original_text"],
                        skill_table=skills_table_flat,
                        prompt_version=prompt_choice,
                    )
                    st.session_state["feedback_result"] = fb_res
                    st.success("Suggestions ready!")
                except httpx.HTTPStatusError as exc:
                    st.error(
                        "**The feedback service could not complete the request.**\n\n"
                        "**Next step:** Check your LLM API key in the `.env` configuration file."
                    )
                except Exception:
                    st.error(
                        "**Could not connect to the feedback service.**\n\n"
                        "**Next step:** Ensure the backend is running and reachable."
                    )

        if "feedback_result" in st.session_state:
            fb = st.session_state["feedback_result"]
            content = fb.get("feedback", {})
            format_valid = fb.get("format_valid", True)

            if not format_valid:
                st.warning("We could not format the suggestions cleanly, please try again.")
                with st.expander("Raw response from the assistant"):
                    st.text(fb.get("raw_response", ""))

            # Result Cards
            st.markdown("### Tailored Recommendations")

            # 1. Why / Explanation
            st.markdown(
                f"""
                <div class="card">
                    <div style="font-weight: 600; color: #1E3A8A; margin-bottom: 6px;">Why</div>
                    <div>{content.get('explanation', 'No overview provided.')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Stronger way to write one of your lines
            rewritten = content.get("rewritten_bullet", "")
            if rewritten:
                st.markdown(
                    f"""
                    <div class="card" style="border-left: 4px solid #2563EB;">
                        <div style="font-weight: 600; color: #1D4ED8; margin-bottom: 6px;">
                            A stronger way to write one of your lines
                        </div>
                        <blockquote style="margin: 0; font-style: italic; color: #1E293B;">
                            "{rewritten}"
                        </blockquote>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 3. Skills that match & Skills to add
            col_f1, col_f2 = st.columns(2, gap="medium")
            with col_f1:
                st.markdown(
                    """
                    <div class="card">
                        <div style="font-weight: 600; color: #166534; margin-bottom: 6px;">Skills that match</div>
                    """,
                    unsafe_allow_html=True,
                )
                m_skills = content.get("matched_skills", [])
                if m_skills:
                    for ms in m_skills:
                        st.markdown(f"- **{ms}**")
                else:
                    st.caption("No matching skills listed.")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_f2:
                st.markdown(
                    """
                    <div class="card">
                        <div style="font-weight: 600; color: #991B1B; margin-bottom: 6px;">Skills to add or show</div>
                    """,
                    unsafe_allow_html=True,
                )
                g_skills = content.get("gaps", [])
                if g_skills:
                    for gs in g_skills:
                        st.markdown(f"- **{gs}**")
                else:
                    st.caption("No missing skills identified.")
                st.markdown("</div>", unsafe_allow_html=True)

            # 4. Next steps
            recs = content.get("recommendations", [])
            if recs:
                st.markdown(
                    """
                    <div class="card">
                        <div style="font-weight: 600; color: #0F172A; margin-bottom: 6px;">Next steps</div>
                    """,
                    unsafe_allow_html=True,
                )
                for idx, r in enumerate(recs, 1):
                    st.markdown(f"**{idx}.** {r}")
                st.markdown("</div>", unsafe_allow_html=True)

            # Mandatory Note
            st.info(
                "💡 **Note:** Suggestions are written by an AI assistant. "
                "The score above is calculated separately and is not changed by it."
            )
