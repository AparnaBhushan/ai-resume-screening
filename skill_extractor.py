# skill_extractor.py

SKILLS = {
    "python", "java", "sql", "machine learning", "deep learning",
    "data analysis", "data science", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "flask", "django",
    "aws", "azure", "gcp",
    "docker", "kubernetes",
    "html", "css", "javascript", "react",
    "git", "linux"
}

def extract_skills(text: str):
    text = text.lower()
    extracted = set()
    for skill in SKILLS:
        if skill in text:
            extracted.add(skill)
    return extracted
