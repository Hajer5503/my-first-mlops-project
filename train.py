import json
import logging
import os

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    """Load and return the Iris dataset as a DataFrame."""
    df = pd.read_csv("data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def train_model(X, y, n_estimators=100, max_depth=None):
    """Train a Random Forest classifier and return the model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        logging.info(
            f"n_estimators={n_estimators}, max_depth={max_depth} -> Accuracy: {acc:.2f}"
        )

        os.makedirs("results", exist_ok=True)
        with open("results/metrics.json", "w") as f:
            json.dump({"accuracy": round(float(acc), 2)}, f)

    return model


if __name__ == "__main__":
    X, y = load_data()

    configs = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 300, "max_depth": 10},
    ]

    for config in configs:
        train_model(X, y, **config)
