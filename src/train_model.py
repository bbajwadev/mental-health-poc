import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train():
    # 1. Load
    df = pd.read_csv("data/raw/train.csv")
    print(f"Loaded {len(df)} raw records")

    # 2. Clean up missing responses
    #    - Drop rows where Response is NaN
    df = df.dropna(subset=["Response"])
    #    - Or if you prefer to keep them, fill with empty string:
    # df["Response"] = df["Response"].fillna("")
    print(f"{len(df)} records remain after dropping null responses")

    # 3. Create weak labels: Advice vs. Neutral
    df["label"] = df["Response"].str.contains(
        r"\b(should|try|consider)\b", case=False, regex=True
    )
    print("Label distribution:")
    print(df["label"].value_counts())

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["Response"], df["label"], test_size=0.2, random_state=42
    )

    # 5. Vectorize
    vec = TfidfVectorizer(max_features=5_000)
    Xtr = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1_000)
    clf.fit(Xtr, y_train)

    # 6. Evaluate
    Xte = vec.transform(X_test)
    preds = clf.predict(Xte)
    print(classification_report(y_test, preds))

    # 7. Save
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/advice_clf.joblib")
    joblib.dump(vec, "model/vectorizer.joblib")
    print("Model and vectorizer saved to ./model/")

if __name__ == "__main__":
    train()
