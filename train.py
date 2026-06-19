import logging

logging.basicConfig(level=logging.INFO)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import json
import os

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    """Load and return the Iris dataset as a DataFrame."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y


def train_model(X, y):
    """Train a Random Forest classifier and return the model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    logging.info(f"Accuracy: {acc:.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump({"accuracy": round(float(acc), 2)}, f)
    logging.info("Metrics saved to results/metrics.json")

    return model


if __name__ == "__main__":
    X, y = load_data()
    model = train_model(X, y)
