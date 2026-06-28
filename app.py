from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Set up logging to both console and file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler (rotates every 10MB, keeps 5 backups)
file_handler = RotatingFileHandler(
    "logs/predictions.log", maxBytes=10_000_000, backupCount=5
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)
app = FastAPI(title="Iris Classifier", version="1.0")

# Load the production model once at startup
model = mlflow.pyfunc.load_model("models:/iris-classifier@production")


# Define the input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(input_data: IrisInput):
    """Predict iris class from measurements."""
    # Convert input to DataFrame with correct column order
    df = pd.DataFrame(
        [
            {
                "sepal length (cm)": input_data.sepal_length,
                "sepal width (cm)": input_data.sepal_width,
                "petal length (cm)": input_data.petal_length,
                "petal width (cm)": input_data.petal_width,
            }
        ]
    )

    # Get prediction
    prediction = model.predict(df)[0]
    class_name = {0: "setosa", 1: "versicolor", 2: "virginica"}[int(prediction)]

    # Log the prediction
    logger.info(
        f"Prediction made | Time: {datetime.utcnow().isoformat()} | "
        f"Input: sepal_length={input_data.sepal_length}, sepal_width={input_data.sepal_width}, "
        f"petal_length={input_data.petal_length}, petal_width={input_data.petal_width} | "
        f"Prediction: {class_name} (class {int(prediction)})"
    )

    # Return result
    return {"prediction": int(prediction), "class_name": class_name}


@app.get("/metrics")
def metrics():
    """Return prediction statistics."""
    # Count predictions per class from logs
    try:
        with open("logs/predictions.log", "r") as f:
            lines = f.readlines()

        setosa_count = sum(1 for line in lines if "setosa" in line)
        versicolor_count = sum(1 for line in lines if "versicolor" in line)
        virginica_count = sum(1 for line in lines if "virginica" in line)
        total_predictions = len(lines)

        return {
            "total_predictions": total_predictions,
            "class_distribution": {
                "setosa": setosa_count,
                "versicolor": versicolor_count,
                "virginica": virginica_count,
            },
        }
    except FileNotFoundError:
        return {"error": "No predictions made yet"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
