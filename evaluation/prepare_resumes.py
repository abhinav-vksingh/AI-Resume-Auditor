import re
import sys
from pathlib import Path
import pandas as pd

# Usage: python evaluation/prepare_resumes.py path\to\UpdatedResumeDataSet.csv
csv_path = sys.argv[1] if len(sys.argv) > 1 else "UpdatedResumeDataSet.csv"
out_dir = Path("evaluation/sample_resumes")
out_dir.mkdir(parents=True, exist_ok=True)
PER_GROUP = 6  # 5 groups x 6 = 30 resumes

df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")

# Deduplicate strictly by normalized resume text to eliminate all identical rows
df["clean_resume"] = df["Resume"].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
u = df.drop_duplicates(subset=["clean_resume"]).copy()

# Map Kaggle categories to our 5 target groups
# To ensure 6 unique resumes per group:
# - data_science: 'Data Science' (10 unique available)
# - human_resources: 'HR' (10 unique available)
# - software_engineering: 'Java Developer' (13) + 'Python Developer' (6)
# - sales: 'Sales' (5) + 'Business Analyst' (6)
# - web_development: 'Web Designing' (4) + 'DotNet Developer' (7)
MAP = {
    "Data Science": "data_science",
    "HR": "human_resources",
    "Java Developer": "software_engineering",
    "Python Developer": "software_engineering",
    "Sales": "sales",
    "Business Analyst": "sales",
    "Web Designing": "web_development",
    "DotNet Developer": "web_development",
}

u["group"] = u["Category"].map(MAP)

# Delete old placeholder / synthetic files
for old in out_dir.glob("*.txt"):
    old.unlink()

# Sample 6 distinct resumes per group with fixed seed
for group, g in u.dropna(subset=["group"]).groupby("group"):
    # For sales and web_development, prioritize core categories first
    if group == "sales":
        core = g[g["Category"] == "Sales"]
        extra = g[g["Category"] != "Sales"]
        sample = pd.concat([core, extra]).head(PER_GROUP)["Resume"]
    elif group == "web_development":
        core = g[g["Category"] == "Web Designing"]
        extra = g[g["Category"] != "Web Designing"]
        sample = pd.concat([core, extra]).head(PER_GROUP)["Resume"]
    else:
        sample = g["Resume"].sample(min(PER_GROUP, len(g)), random_state=42)

    for i, text in enumerate(sample, 1):
        # Clean PII patterns: emails & phone numbers
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"\+?\d[\d\s().-]{8,}\d", " ", text)
        (out_dir / f"{group}_{i:02d}.txt").write_text(text, encoding="utf-8")
        print("wrote", f"{group}_{i:02d}.txt")
