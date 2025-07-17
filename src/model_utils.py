import joblib
vec = joblib.load("model/vectorizer.joblib")
clf = joblib.load("model/advice_clf.joblib")

def predict_advice_type(text: str) -> str:
    """Returns 'Advice' or 'Neutral'."""
    label = clf.predict(vec.transform([text]))[0]
    return "Advice" if label else "Neutral"
