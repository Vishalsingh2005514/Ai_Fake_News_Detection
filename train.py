"""
train.py
--------
Trains a Logistic Regression fake news classifier using True.csv and Fake.csv.
Saves model.pkl, vectorizer.pkl, and metrics.json for use in the app.
"""

import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train():
    print("Loading dataset...")

    if not os.path.exists("True.csv") or not os.path.exists("Fake.csv"):
        print("ERROR: True.csv or Fake.csv not found.")
        print("Run generate_dataset.py first to create training data.")
        return

    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")

    # Label: 1 = Real, 0 = Fake
    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True).sample(frac=1, random_state=42)

    # Combine title + text for richer features
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    X = df["content"]
    y = df["label"]

    print(f"Total articles: {len(df)} ({y.sum()} real, {(y==0).sum()} fake)")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred), 4),
        "recall":           round(recall_score(y_test, y_pred), 4),
        "f1":               round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1']*100:.2f}%")

    print("Saving model and vectorizer...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete! Files saved: model.pkl, vectorizer.pkl, metrics.json")


if __name__ == "__main__":
    train()
