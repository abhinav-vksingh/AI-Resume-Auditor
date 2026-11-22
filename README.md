# AI Resume Auditor

A transparent, explainable resume–job-description scoring tool.  
University project for **Intelligent Model Design** and **Prompt Engineering**.

## How the score is computed

```
final_score = 0.6 × semantic_similarity + 0.4 × skill_overlap
```

Both components are scaled 0–100. **No LLM calculates or overrides this score.**

## Tech stack

| Layer | Library |
|---|---|
| Backend | FastAPI + Uvicorn |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF parsing | PyMuPDF |
| ML / stats | scikit-learn, scipy, pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Frontend | Streamlit |

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env      # Windows
# cp .env.example .env       # macOS/Linux
# Edit .env if needed

# 4. Run the backend
uvicorn backend.app.main:app --reload
```

## Project structure

```
resume-auditor/
  backend/
    app/
      main.py          # FastAPI app + health endpoint
      config.py        # Weights, model name (loaded from .env)
      schemas.py       # Pydantic request/response models
      routers/
        extract.py     # POST /extract
      services/
        extraction.py  # PDF → raw text (PyMuPDF)
    data/
      skills.json      # Curated skill list (Phase 2)
      aliases.json     # Skill normalisation map (Phase 2)
      names.json       # Identity names for bias audit (Phase 5)
      universities.json
  frontend/
    app.py             # Streamlit UI (Phase 7)
  evaluation/          # Scoring scripts + sample data
  tests/
    test_extraction.py
  requirements.txt
  .env.example
  README.md
```

## Phases

| Phase | What gets built |
|---|---|
| 1 | Scaffold, config, schemas, PDF extraction, `/health`, `/extract` |
| 2 | Preprocessing (lowercase, clean, section detection, alias normalisation) |
| 3 | Semantic matching (embeddings + cosine similarity) |
| 4 | Skill extraction (dictionary + regex, Matched/Weak/Missing) |
| 5 | Score fusion + explainability breakdown |
| 6 | Counterfactual bias audit |
| 7 | Streamlit frontend + LLM feedback (optional) |
