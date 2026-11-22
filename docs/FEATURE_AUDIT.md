# Feature Audit Report: AI Resume Auditor

**Audit Date:** 2026-11-19  
**Execution Environment:** Windows (Python 3.9.13)  
**Verification Method:** Static code inspection, test suite execution (173 tests), and live runtime invocation of FastAPI services.

---

## 1. Feature Table

| Feature | Status | Implementing File & Function | Exposed Endpoint | UI Tab / Element | Tests Present? | Notes / Missing Elements |
|---|---|---|---|---|---|---|
| **PDF Text Extraction** | Implemented | `backend/app/services/extraction.py`<br>`extract_text_from_pdf` | `POST /extract` | Sidebar: "Upload Resume (PDF)" | Yes (`test_extraction.py`, 7 tests) | Uses PyMuPDF (`fitz`), preserves newlines, captures page count and char count. |
| **Preprocessing: Text Cleaning** | Implemented | `backend/app/services/preprocessing.py`<br>`clean_text` | `POST /preprocess` | Used internally by `POST /analyze` | Yes (`test_preprocessing.py`) | Lowercases text, replaces form feeds, strips control characters, normalizes whitespace. |
| **Preprocessing: PII Stripping** | Implemented | `backend/app/services/preprocessing.py`<br>`strip_pii` | `POST /preprocess` | Used internally by `POST /analyze` | Yes (`test_preprocessing.py`) | Removes email, phone formats, URLs, and first-line candidate name heuristic. Survives date ranges/percentages. |
| **Preprocessing: Section Detection** | Implemented | `backend/app/services/preprocessing.py`<br>`detect_sections` | `POST /preprocess` | Used internally by `POST /analyze` | Yes (`test_preprocessing.py`) | Detects 6 canonical sections (`skills`, `experience`, `projects`, `education`, `summary`, `other`). |
| **Preprocessing: Alias Normalization** | Implemented | `backend/app/services/preprocessing.py`<br>`normalize_aliases` | `POST /preprocess` | Used internally by `POST /analyze` | Yes (`test_preprocessing.py`) | Replaces abbreviations with canonical terms from `aliases.json` using whole-word regex boundaries. |
| **Semantic Matching** | Implemented | `backend/app/services/semantic.py`<br>`compute_semantic_similarity` | `POST /match` | Tab 1: Component Bars & Metric Card | Yes (`test_semantic.py`, 13 tests) | Loads `all-MiniLM-L6-v2` singleton with local cache check; encodes resume section chunks, averages vectors, computes cosine similarity scaled 0–100. |
| **Skill Extraction & Classification** | Implemented | `backend/app/services/skills.py`<br>`run_skill_analysis` | `POST /match` | Tab 1: Matched / Weak / Missing Breakdown Table | Yes (`test_skills.py`, 30 tests) | Curated 154 skills dictionary; classifies into `Matched` (primary section or >1 mention), `Weak` (single generic mention), or `Missing`. Formula: $(M + 0.5W) / Total \times 100$. |
| **Score Fusion** | Implemented | `backend/app/services/scoring.py`<br>`compute_final_score` | `POST /analyze` | Tab 1: Final Score Gauge & Weight Formula | Yes (`test_scoring.py`, 11 tests) | Strictly algorithmic: $0.6 \times \text{semantic} + 0.4 \times \text{skill\_overlap}$, configurable via `config.py` / `.env`. |
| **Rule-Based Explainability** | Implemented | `backend/app/services/explain.py`<br>`generate_explanation` | `POST /analyze` | Tab 1: "Why this score" Positive & Negative Drivers | Yes (`test_explain.py`, 24 tests) | Purely rule-based conditionals based on scores and skill statuses. No LLM touches this logic. |
| **Counterfactual Bias Audit** | Implemented | `backend/app/services/bias.py`<br>`run_audit` | `POST /bias-audit` | Tab 2: "Counterfactual Bias Audit" | Yes (`test_bias.py`, 29 tests) | Generates $N$ identity variants by swapping strictly name or university into original text; computes $\Delta = \text{score}_B - \text{score}_A$, mean, std, and paired t-test. |
| **Prompt-Engineered LLM Feedback** | Implemented | `backend/app/services/llm_feedback.py`<br>`generate_feedback` | `POST /feedback` | Tab 3: "LLM Feedback" | Yes (`test_llm_feedback.py`, 15 tests) | Provider-agnostic HTTP client (supports Groq/OpenAI/Ollama). Formats prompts `v1`, `v2`, or `v3`. Validates JSON output and enforces schema without crashing. |
| **Streamlit Thin-Client Frontend** | Implemented | `frontend/app.py` | N/A (Frontend Consumer) | Complete UI (Sidebar + 3 Tabs) | Not verified (No automated UI test suite) | Calls FastAPI backend via `httpx`. Displays score gauge, component metrics, skill table, bias metrics, and feedback rewrite. |

---

## 2. Endpoints

| HTTP Method | Route Path | Expected Input | Output Schema | Frontend Status |
|---|---|---|---|---|
| `GET` | `/health` | None | `HealthResponse` (`status`, `model`, `weights`, `version`) | **Used**: Called on load by `check_backend_health()` to ensure backend is online. |
| `POST` | `/extract` | Multipart `file: UploadFile` (PDF) | `ExtractionResponse` (`text`, `page_count`, `char_count`) | **Used**: Called when clicking "Analyze Resume" to store `original_text` for Bias Audit & Feedback tabs. |
| `POST` | `/preprocess` | `PreprocessRequest` (`text: str`) | `PreprocessResponse` (`original_text`, `core_text`, `sections`) | **Unused by Frontend**: Handled internally by `/analyze`. Available only for direct API consumers or debugging. |
| `POST` | `/match` | `MatchRequest` (`core_text`, `job_description`, `sections`) | `MatchResponse` (`semantic_score`, `skill_overlap`, `skill_table`) | **Unused by Frontend**: Handled internally by `/analyze`. Available only for direct API consumers or debugging. |
| `POST` | `/analyze` | Multipart `file: UploadFile` + Form field `job_description: str` | `AnalyzeResponse` (`final_score`, `components`, `skills`, `explanation`) | **Used**: Called when clicking "Analyze Resume" in the sidebar to populate Tab 1. |
| `POST` | `/bias-audit` | `BiasAuditRequest` (`original_text`, `job_description`, `attribute`, `n_pairs`) | `BiasAuditResponse` (`mean_delta`, `std_delta`, `p_value`, `significant`, `raw_rows`, `disclaimer`) | **Used**: Called when clicking "Run Bias Audit" in Tab 2. |
| `POST` | `/feedback` | Query param `prompt_version: Literal['v1','v2','v3']` + `FeedbackRequest` (`job_description`, `resume_text`, `skill_table`) | `FeedbackResponse` (`prompt_version`, `format_valid`, `feedback`, `raw_response`, `error`) | **Used**: Called when clicking "Generate Feedback" in Tab 3. |

### Observations on Endpoints:
1. **Redundant Round-Trip on Analysis:** The frontend calls both `POST /extract` and `POST /analyze` upon clicking "Analyze Resume". This happens because `AnalyzeResponse` does not include `original_text`, requiring the frontend to upload the PDF twice to retain the raw text for tabs 2 and 3.
2. **Unused Standalone Endpoints:** `POST /preprocess` and `POST /match` are fully functional and tested in unit tests, but are never called by the Streamlit application.

---

## 3. Data Files

| File Path | Entry Count | Structure | Adequacy for Real Resumes |
|---|---|---|---|
| `backend/data/skills.json` | **154 skills** across 5 categories:<br>- `software_engineering`: 40<br>- `data_science`: 35<br>- `web_development`: 29<br>- `human_resources`: 25<br>- `sales`: 25 | Key-value dictionary mapping domain names to lists of lowercase strings. | **Sufficient for common tech & business roles**, but missing emerging modern tooling (e.g., `Polars`, `Next.js`, `Celery`, `GCP`, `Tailwind`). If a JD contains unlisted skills, they cannot be matched. |
| `backend/data/aliases.json` | **42 mapping pairs** (e.g., `"ml": "machine learning"`, `"k8s": "kubernetes"`, `"postgres": "postgresql"`) | JSON dictionary mapping abbreviation to expanded lowercase canonical term. | **Sufficient for general software/data science**, covers major acronyms. Specialized domain acronyms (e.g., in finance or medicine) are absent. |
| `backend/data/names.json` | **40 names** across 2 demographic groups:<br>- `white_american`: 20<br>- `black_american`: 20 | JSON dictionary with bibliographic citation of Bertrand & Mullainathan (2004). | **Sufficient for audit methodology demonstration**. Grounded directly in established labor audit literature. |
| `backend/data/universities.json` | **20 universities** across 2 categories:<br>- `well_known`: 10 real universities (MIT, Stanford, Harvard, IIT Delhi, IIT Bombay, CMU, UC Berkeley, Oxford, Georgia Tech, Michigan)<br>- `lesser_known`: 10 synthetic/fictional names (e.g., Greenwood State University, Western Plains College) | JSON dictionary mapping prestige labels to lists of university names. | **Partial / Synthetic limitation**: The "lesser-known" universities are fictitious names rather than actual regional accredited colleges. SentenceTransformer embeddings for synthetic names lack real-world cultural prestige embeddings. |

---

## 4. Hardcoding and Fake Data

1. **Unverified Baseline Claim in `evaluation/keyword_baseline.py` (Line 120):**
   ```python
   print("Note: our embedding pipeline typically scores 10–20 points HIGHER than these baselines on the same pair, because it understands synonyms and context.")
   ```
   *Finding:* This text is printed verbatim to stdout without computing the embedding score at runtime on the evaluated text pair.
2. **Synthetic Data in `evaluation/bias_chart.py` (Lines 83–105):**
   *Finding:* The script contains a `--demo` flag implementing `_synthetic_demo_data()` that generates hardcoded deltas (`mean_delta: 0.05`, `std_delta: 0.42`) for visualization if no real audit output JSON is provided.
3. **Simulated Evaluation in `evaluation/prompt_eval.py` (Lines 150–220):**
   *Finding:* Contains a `--mock` flag that produces simulated evaluation ratings across 10 scenarios without invoking an LLM.
4. **Fictional Institutions in `backend/data/universities.json`:**
   *Finding:* `Greenwood State University`, `Western Plains College`, `Tri-State University`, `Midwest Institute of Technology`, `Valley Community College`, `Northern Plains University`, `Eastside Technical College`, `Southern Regional University`, `Lakewood State College`, `Riverside Community College` are made-up names, not real-world colleges.

---

## 5. Runtime Check

The backend services were invoked in-process using FastAPI's application context on Python 3.9.13 with a real synthetic PDF resume and real Job Description text.

### Executed Request Payloads:
- **Resume text:** Senior Python Developer with 5 years experience, listing Python, PyTorch, Scikit-learn, Docker, FastAPI, Kubernetes, AWS, and IIT Delhi education.
- **Job Description:** Senior Python Developer requiring Python, PyTorch, Scikit-learn, Docker, REST APIs, Kubernetes, AWS.

### Actual Output Results:

#### 1. `POST /analyze` (Status: 200 OK)
```json
{
  "final_score": 84.11,
  "components": {
    "semantic": 73.51,
    "skill_overlap": 100.0,
    "weights": {
      "semantic": 0.6,
      "skill_overlap": 0.4
    }
  },
  "skills": {
    "matched": [
      "python",
      "docker",
      "kubernetes",
      "machine learning",
      "pytorch",
      "scikit-learn"
    ],
    "weak": [],
    "missing": []
  },
  "explanation": {
    "positive_factors": [
      "Overall strong candidate match (final score: 84.11/100).",
      "Moderate semantic alignment between resume and job description (73.51/100) — the resume is broadly relevant to the role.",
      "Skills directly evidenced in the resume: python, docker, kubernetes, machine learning, pytorch and 1 more.",
      "Good skill coverage (100.0/100) — 6 of 6 required skills are matched."
    ],
    "negative_factors": []
  }
}
```

#### 2. `POST /bias-audit` (Status: 200 OK)
```json
{
  "attribute": "name",
  "group_a_label": "white_american",
  "group_b_label": "black_american",
  "mean_delta": 0.0,
  "std_delta": 0.0,
  "p_value": null,
  "significant": false,
  "n_observations": 3,
  "raw_rows": [
    {
      "resume_index": 0,
      "identity_a": "Emily Walsh",
      "identity_b": "Lakisha Washington",
      "score_a": 83.78,
      "score_b": 83.78,
      "delta": 0.0
    },
    {
      "resume_index": 0,
      "identity_a": "Anne Schmidt",
      "identity_b": "Jamal Jones",
      "score_a": 83.78,
      "score_b": 83.78,
      "delta": 0.0
    },
    {
      "resume_index": 0,
      "identity_a": "Jill Taylor",
      "identity_b": "Tamika Williams",
      "score_a": 83.78,
      "score_b": 83.78,
      "delta": 0.0
    }
  ],
  "disclaimer": "DISCLAIMER: This is a small project-level audit conducted on synthetic resume variants. The results do NOT constitute proof of fairness or unfairness in any real-world hiring context..."
}
```

#### 3. `POST /feedback?prompt_version=v3` (Status: 200 OK - Groq Live Provider)
```json
{
  "prompt_version": "v3",
  "format_valid": true,
  "feedback": {
    "matched_skills": [
      "Python",
      "PyTorch",
      "Scikit-learn",
      "Docker",
      "REST APIs (FastAPI)",
      "Kubernetes",
      "AWS",
      "Machine Learning"
    ],
    "gaps": [],
    "rewritten_bullet": "Built and deployed machine learning pipelines using PyTorch and Scikit-learn, directly supporting the company's AI-driven product features.",
    "explanation": "The resume provides clear evidence of all core requirements: Python development, REST API design (FastAPI), machine-learning pipelines with PyTorch and Scikit-learn, containerization with Docker, and orchestration on Kubernetes, plus listed AWS familiarity. No required skill is missing.",
    "recommendations": [
      "Quantify impact by providing business or system metrics for the deployed ML pipelines."
    ]
  },
  "raw_response": "{...}",
  "error": null
}
```

### Runtime Evaluation:
All 3 calls executed successfully with HTTP 200. No failures or unexpected schema shapes occurred. In the bias audit, `p_value` was correctly serialized as `null` due to zero delta variance across name variants.

---

## 6. LLM Layer

1. **Configuration Status:**
   - Provider is configured via environment variables in `.env`:
     - `LLM_API_KEY`: Set to Groq API key (`gsk_...`).
     - `LLM_MODEL`: `openai/gpt-oss-120b`.
     - `LLM_BASE_URL`: `https://api.groq.com/openai/v1`.
   - Verified live in runtime: Calls Groq's OpenAI-compatible completions API using `httpx`.
2. **Prompt Templates:**
   - Three prompt template text files exist under `prompts/`:
     - `prompts/v1_baseline.txt` (1,268 chars): Role instruction baseline.
     - `prompts/v2_criteria.txt` (1,750 chars): Criteria-focused on relevance, skill evidence, clarity, and grounding.
     - `prompts/v3_structured.txt` (1,799 chars): Schema-enforcing prompt requiring matched skills, gaps, rewritten bullet, explanation, and recommendations.
3. **Output Format Validation:**
   - `backend/app/services/llm_feedback.py` implements `parse_and_validate_feedback(raw_text)`.
   - Validates that root element is a JSON dict containing all required keys (`matched_skills`, `gaps`, `rewritten_bullet`, `explanation`, `recommendations`) and correct types.
   - If malformed JSON or markdown fences are returned, it attempts regex extraction; if invalid, sets `format_valid = False` with diagnostics instead of raising a 500 error.
4. **Code Confirmation: LLM Never Touches Score:**
   - Verified in `backend/app/services/scoring.py` (Line 22) and `backend/app/routers/analyze.py` (Lines 110–135): `final_score` is computed solely by mathematical arithmetic (`0.6 * semantic_score + 0.4 * skill_overlap`).
   - The LLM service (`llm_feedback.py`) is never imported by `scoring.py` or `analyze.py`.
   - Prompts explicitly state: *"You must NEVER calculate, suggest, or alter any numeric score."*

---

## 7. Tests

- **Total Test Count:** **173 tests** across 8 test suites.
- **Execution Result:** All 173 tests passed in **~10.92 seconds** (`pytest tests/ -v`).

### Breakdown of Test Coverage:
- `tests/test_preprocessing.py` (44 tests): Text cleaning, PII regex (emails, international phone numbers, URLs, name heuristic), section header detection, alias normalization.
- `tests/test_skills.py` (30 tests): Extraction of required skills from JD, section-aware classification (Matched vs Weak vs Missing), overlap scoring formula.
- `tests/test_bias.py` (29 tests): Name and university swapping, attribute isolation, paired t-test statistics, edge cases (single observation NaN).
- `tests/test_explain.py` (24 tests): Return structure, positive factors generation, negative factors generation, cap overflow handling.
- `tests/test_llm_feedback.py` (15 tests): Prompt formatting across v1/v2/v3, JSON parser and markdown fence stripper, missing key validation, fallback generation.
- `tests/test_semantic.py` (13 tests): Section and paragraph chunking logic, cosine similarity arithmetic, lazy singleton model loading.
- `tests/test_scoring.py` (11 tests): Formula verification ($0.6 \times \text{semantic} + 0.4 \times \text{skills}$), weight sum validation, score clamping [0, 100].
- `tests/test_extraction.py` (7 tests): PyMuPDF extraction, empty PDF error, corrupted PDF handling, page count tracking.

### Core Logic With NO Tests:
1. **FastAPI HTTP Routers:** Zero tests use `TestClient(app)` in `tests/`. Endpoint status codes, validation handlers, form-data parsing, and query parameter handling are not tested in pytest.
2. **Streamlit Frontend:** `frontend/app.py` has no automated tests.
3. **Evaluation Scripts:** `evaluation/keyword_baseline.py`, `evaluation/bias_chart.py`, and `evaluation/prompt_eval.py` have no unit tests.
4. **Pretrained Model Weights:** In `test_semantic.py`, `SentenceTransformer` is intentionally mocked for speed; real embedding inference is not tested in pytest.

---

## 8. User Flow (Streamlit App)

### Step-by-Step Experience:
1. **Sidebar Input:**
   - User sees a backend connection badge (`🟢 Online (v0.1.0)`).
   - User drags and drops a PDF resume file into the file uploader.
   - User pastes a job description into the text area.
   - User clicks the prominent red **"Analyze Resume"** button.
2. **Analysis & Tab 1 ("Analysis & Scoring"):**
   - The page displays a large metric card with the **Final Score (0–100)** alongside component metric cards (**Semantic Similarity** and **Skill Overlap**).
   - Below, horizontal progress bars visualize component contributions against configured weights ($0.6 \times \text{Semantic} + 0.4 \times \text{Skill Overlap}$).
   - A colored skill table lists each required skill from the JD classified as Matched (green), Weak (orange), or Missing (red).
   - "Why this score?" section displays rule-generated bullet factors explaining positive drivers and negative gaps.
3. **Tab 2 ("Counterfactual Bias Audit"):**
   - If user navigates here without analyzing a resume, a warning instructs them to analyze first.
   - User selects an attribute (`Candidate Name` or `University Prestige`) and adjusts a slider for number of identity pairs (2 to 10).
   - User clicks **"🔬 Run Bias Audit"**.
   - Displays 4 metrics: Mean Delta ($\Delta$), Standard Deviation, Paired t-test p-value, and Significance flag.
   - Shows a "Viva Insight" explaining why $\Delta \approx 0$ is expected for name-auditing.
   - Shows a detailed comparison table of each identity pair with Variant A score, Variant B score, and individual delta.
4. **Tab 3 ("LLM Feedback"):**
   - User selects a prompt strategy from a dropdown: `v3_structured`, `v2_criteria`, or `v1_baseline`.
   - User clicks **"✨ Generate Feedback"**.
   - Shows a green banner confirming schema validation pass.
   - Displays Qualitative Analysis: Synthesis & Explanation, a Strengthened Bullet Point Rewrite based strictly on verified resume facts, evidenced skills, and identified skill gaps.

### Evaluation of Flow Cohesion:
The tabs feel **mostly like one connected flow**: Tab 2 and Tab 3 depend on state generated by Tab 1 (`st.session_state["analyzed"]` and `st.session_state["original_text"]`). However, Tab 2 requires re-running a separate button click to trigger the audit, and Tab 3 requires a third button click. They function as progressive analytical views on the uploaded candidate document.

---

## 9. Gaps and Inconsistencies

Ranked list of the top 10 discrepancies between the specification and current code:

| Rank | Severity | Issue / Discrepancy | Description |
|---|---|---|---|
| **1** | **High** | **Synthetic / Fictitious Universities in Audit Dataset** | In `backend/data/universities.json`, the 10 "lesser-known" universities are made-up names (e.g., `Greenwood State University`, `Western Plains College`). SentenceTransformer embeddings for synthetic names cannot reflect genuine real-world institutional bias. |
| **2** | **High** | **Zero Integration Tests for HTTP Endpoints** | The 173 pytest unit tests solely test internal service functions. None of the FastAPI routers (`/extract`, `/preprocess`, `/match`, `/analyze`, `/bias-audit`, `/feedback`) or their HTTP status codes/error handlers are exercised by automated tests. |
| **3** | **Medium** | **Redundant Double PDF Extraction in UI** | Clicking "Analyze Resume" in `frontend/app.py` triggers `call_extract_api` (uploading PDF to `/extract`) and immediately triggers `call_analyze_api` (uploading the PDF to `/analyze`, which repeats extraction internally). This occurs because `AnalyzeResponse` does not include `original_text`. |
| **4** | **Medium** | **Unused Standalone Modular Endpoints** | The architecture provides `POST /preprocess` and `POST /match`, but the frontend completely bypasses them in favor of the monolithic `POST /analyze`. They exist as orphaned endpoints for the UI. |
| **5** | **Medium** | **Degenerate Paired T-Test for Single-Resume Name Audits** | For single-resume name audits, every pair yields $\Delta = 0.0$ because `strip_pii` purges names. This zero-variance sample causes `scipy.stats.ttest_rel` to produce $t = \frac{0}{0} = \text{NaN}$. While the API converts this to `null` and the UI shows `N/A`, students must be prepared to explain why the t-test has no p-value. |
| **6** | **Medium** | **Unverified Print Claim in Baseline Script** | In `evaluation/keyword_baseline.py` (Line 120), the script prints: *"our embedding pipeline typically scores 10–20 points HIGHER than these baselines..."* without executing the embedding model on that pair at runtime. |
| **7** | **Medium** | **Chunking Mechanism Sensitive to Header Formatting** | `backend/app/services/semantic.py` chunks resumes based on section headers detected during preprocessing. Resumes without standard headers ("Experience", "Skills") collapse into a single `"other"` block, falling back to double-newline chunking or truncation to 256 word-pieces. |
| **8** | **Low** | **Dictionary Scope Limited to 154 Skills** | In `backend/data/skills.json`, only 154 skills across 5 categories exist. Modern frameworks (`FastAPI` was recently added, but tools like `Polars`, `Celery`, `GCP`, `Tailwind`) will be missed if present in JDs. |
| **9** | **Low** | **Unautomated Rubric Columns in Prompt Evaluation** | `evaluation/prompt_eval.py` generates output rows for prompt evaluation, but leaves qualitative scoring columns (`relevance`, `consistency`, `clarity`, `skill_gap_accuracy`) blank for manual human evaluation rather than running an LLM-as-a-judge eval. |
| **10** | **Low** | **No Automated Tests for Streamlit Frontend** | There are no UI test suites (e.g. Playwright or `streamlit.testing.v1.AppTest`) ensuring widgets, buttons, and session state bindings remain intact across dependency updates. |

---

## 5-Line Viva Summary

> "Our project is an interpretable resume-scoring system that computes candidate match purely through an algorithmic formula—combining all-MiniLM-L6-v2 dense semantic embeddings with rule-based skill coverage—guaranteeing that an LLM never calculates or alters the score. It includes an explainability engine that generates auditable bullet points detailing why a score was awarded. We implement a counterfactual audit framework that proves the pipeline is mathematically name-blind while detecting potential institutional prestige bias. Finally, an LLM layer is restricted strictly to qualitative feedback and bullet-point rewriting using explicit criteria and rigid schema validation."
