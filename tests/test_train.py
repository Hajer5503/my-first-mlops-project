import pandas as pd
from train import load_data, train_model


def test_load_data_returns_dataframe():
    X, y = load_data()
    assert isinstance(X, pd.DataFrame)
    # Checks that your function actually returns a DataFrame and not something else — a numpy array, a list, None. Sounds obvious, but if someone refactors load_data() and accidentally returns the wrong type, every downstream function breaks in confusing ways. This catches it immediately.


def test_load_data_correct_shape():
    # This test checks that the DataFrame has the expected number of rows and columns, and that the target variable has the correct length. If load_data() is modified to return a different dataset or if the dataset structure changes, this test will fail, alerting you to the issue right away.
    X, y = load_data()
    assert X.shape == (150, 4)
    assert len(y) == 150


def test_load_data_no_nulls():
    # This test ensures that the dataset is complete and doesn't contain any missing values. If load_data() is changed to load a different dataset or if the dataset is modified in a way that introduces null values, this test will fail, prompting you to investigate and fix the issue before it causes problems in model training.
    X, y = load_data()
    assert X.isnull().sum().sum() == 0


def test_model_output_shape():
    # This test checks that the model's predictions have the correct shape, which should match the number of samples in the dataset. If train_model() is modified to return a model that doesn't predict the correct number of samples (for example, if it accidentally returns a model trained on a different dataset), this test will fail, alerting you to the issue immediately.
    X, y = load_data()
    model = train_model(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_model_accuracy_above_threshold():
    # This test checks that the model's accuracy on the training data is above a certain threshold (in this case, 80%). If train_model() is modified in a way that degrades the model's performance (for example, by changing hyperparameters or using a different algorithm), this test will fail, prompting you to investigate and improve the model before it is used in production.
    X, y = load_data()
    model = train_model(X, y)
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    assert accuracy > 0.8


def test_load_data_correct_columns():
    X, y = load_data()
    expected_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    assert list(X.columns) == expected_columns
