import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load trained model and vectorizer
model = joblib.load(BASE_DIR / "models" / "resume_rf_model.joblib")
tfidf = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.joblib")

def predict_role(resume_text):
    vector = tfidf.transform([resume_text])
    prediction = model.predict(vector)
    return prediction[0]

# Test prediction
if __name__ == "__main__":
    sample_text = "Experienced Python developer with machine learning and data analysis skills"
    print("Predicted Role:", predict_role(sample_text))
