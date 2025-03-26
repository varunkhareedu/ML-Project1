import csv
import numpy as np
import pytest
from model.LassoHomotopy import LassoHomotopyModel
from sklearn.linear_model import Lasso as SklearnLasso


def test_predict():
    model = LassoHomotopyModel(lambda_val=5.0)

    with open("./small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        data = list(reader)

    feature_keys = [k for k in data[0] if k.lower().startswith("x")]
    target_key = list(data[0].keys())[-1]  # Assume last column is the target

    print("Detected target column:", target_key)
    assert target_key.lower() in {"y", "target"}, "Unexpected target column name"

    X = np.array([[float(row[k]) for k in feature_keys] for row in data])
    y = np.array([float(row[target_key]) for row in data])

    results = model.fit(X, y)
    preds = results.predict(X)

    assert preds.shape == y.shape
    assert isinstance(preds, np.ndarray)

    print("Coefficients:", results.coef_)
    print("Non-zero coefficients (>|1e-4|):", np.sum(np.abs(results.coef_) > 1e-4), "/", X.shape[1])


def test_collinear_features():
    X = np.random.randn(100, 1)
    X = np.hstack([X, X, np.random.randn(100, 1)])  # X0 == X1 (perfectly collinear)
    y = 3 * X[:, 0] + 2 * X[:, 2] + np.random.randn(100) * 0.1

    model = LassoHomotopyModel(lambda_val=0.1)
    results = model.fit(X, y)
    coef = results.coef_

    print("Collinear Test Coefficients:", coef)
    assert (abs(coef[0]) < 1e-3) or (abs(coef[1]) < 1e-3)  # One of the collinear vars dropped
    assert abs(coef[2]) > 1e-3


def test_lambda_sparsity_effect():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    true_coef = np.zeros(10)
    true_coef[:3] = [3, 0, -2]
    y = X @ true_coef + np.random.randn(100) * 0.1

    model_low_lambda = LassoHomotopyModel(lambda_val=0.01)
    model_high_lambda = LassoHomotopyModel(lambda_val=10)

    coef_low = model_low_lambda.fit(X, y).coef_
    coef_high = model_high_lambda.fit(X, y).coef_

    print("Low λ Non-zero:", np.sum(np.abs(coef_low) > 1e-4))
    print("High λ Non-zero:", np.sum(np.abs(coef_high) > 1e-4))

    assert np.sum(np.abs(coef_low) > 1e-4) >= np.sum(np.abs(coef_high) > 1e-4)


def test_against_sklearn_lasso():
    np.random.seed(0)
    X = np.random.randn(100, 5)
    true_coef = np.array([1.5, -2.0, 0.0, 0.0, 3.0])
    y = X @ true_coef + np.random.randn(100) * 0.1

    lambda_val = 0.1

    my_model = LassoHomotopyModel(lambda_val=lambda_val)
    my_results = my_model.fit(X, y)
    my_coef = my_results.coef_

    sklearn_model = SklearnLasso(alpha=lambda_val, fit_intercept=True, max_iter=10000)
    sklearn_model.fit(X, y)
    skl_coef = sklearn_model.coef_

    print("Your Coefs:", my_coef)
    print("Sklearn Coefs:", skl_coef)

    np.testing.assert_allclose(my_coef, skl_coef, atol=0.5)