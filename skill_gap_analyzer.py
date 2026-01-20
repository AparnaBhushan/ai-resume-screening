from skill_extractor import extract_skills

def analyze_skill_gap(resume_text, job_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills.difference(resume_skills)

    match_percentage = (
        len(matched_skills) / len(job_skills) * 100
        if job_skills else 0
    )

    return {
        "matched_skills": sorted(list(matched_skills)),
        "missing_skills": sorted(list(missing_skills)),
        "match_percentage": round(match_percentage, 2)
    }

# 👇 THIS PART WAS MISSING
if __name__ == "__main__":
    resume_text = "I have experience in Python, machine learning, pandas and numpy"
    job_text = "Looking for a Data Scientist with Python, SQL, AWS, machine learning"

    result = analyze_skill_gap(resume_text, job_text)
    print("Skill Gap Analysis Result:")
    print(result)
