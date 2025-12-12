import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------
# Load model + vectorizer
# --------------------------
model = joblib.load("../models/logreg_model.joblib")
vectorizer = joblib.load("../models/tfidf_vectorizer.joblib")

def predict(text: str):
    text_clean = text.lower()
    X_vec = vectorizer.transform([text_clean])
    pred = model.predict(X_vec)[0]
    return pred

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'Your text here'")
        sys.exit(1)
    input_text = " ".join(sys.argv[1:])
    prediction = predict(input_text)
    print(f"Prediction: {prediction}")