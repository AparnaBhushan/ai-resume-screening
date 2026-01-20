from flask import Flask, render_template, request
import pandas as pd
from pathlib import Path

# Import your modules
from predict import predict_role
from skill_gap_analyzer import analyze_skill_gap

# -----------------------------
# App & Path Setup
# -----------------------------
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# Load Job Dataset SAFELY
# -----------------------------
JOB_DATA_PATH = BASE_DIR / "linkdin_Job_data.csv"

if not JOB_DATA_PATH.exists():
    raise FileNotFoundError(f"Job data file not found at: {JOB_DATA_PATH}")

jobs_df = pd.read_csv(JOB_DATA_PATH)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        resume_text = request.form.get("resume", "")

        if resume_text.strip() == "":
            result = {"error": "Resume text cannot be empty"}
            return render_template("index.html", result=result)

        # 1. Predict role
        predicted_role = predict_role(resume_text)

        # 2. Find matching job description
        job_row = jobs_df[
            jobs_df["job"].astype(str).str.contains(
                predicted_role, case=False, na=False
            )
        ].head(1)

        if not job_row.empty:
            job_text = str(job_row.iloc[0]["job_details"])
        else:
            job_text = ""

        # 3. Skill gap analysis
        skill_gap = analyze_skill_gap(resume_text, job_text)

        result = {
            "predicted_role": predicted_role,
            "match_percentage": skill_gap["match_percentage"],
            "matched_skills": skill_gap["matched_skills"],
            "missing_skills": skill_gap["missing_skills"],
        }

    return render_template("index.html", result=result)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
