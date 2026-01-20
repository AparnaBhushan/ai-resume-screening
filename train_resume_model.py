import pandas as pd
import numpy as np
import re
import string
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# 1. Load dataset
# ---------------------------
# Adjust the path if needed
from pathlib import Path

data_path = Path("UpdatedResumeDataSet.csv")
df = pd.read_csv(data_path)


print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# The dataset usually has columns: 'Category', 'Resume_str'
df = df.rename(columns={"Category": "category", "Resume_str": "resume"})

# ---------------------------
# 2. Preprocessing function
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove urls
    text = re.sub(r"\d+", " ", text)             # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuations
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_resume"] = df["Resume"].apply(clean_text)

# ---------------------------
# 3. Train/Test Split
# ---------------------------
X = df["clean_resume"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 4. Vectorization (TF-IDF)
# ---------------------------
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words="english",
    max_features=5000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# 5. Train Model (Random Forest)
# ---------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_vec, y_train)

# ---------------------------
# 6. Evaluate
# ---------------------------
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# 7. Save model + vectorizer
# ---------------------------
Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/resume_rf_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("✅ Model and vectorizer saved in 'models/' folder")
