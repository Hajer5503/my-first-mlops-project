import mlflow.pyfunc
from train import load_data

model = mlflow.pyfunc.load_model("models:/iris-classifier@production")

X, y = load_data()
predictions = model.predict(X.head(5))
print("Predictions:", predictions)