"""Integration test for the end-to-end loan approval pipeline."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for headless test runs
import matplotlib
matplotlib.use("Agg")

from member1_data_preprocessing import prepare_data
from member2_model_training import train_models
from member3_model_evaluation import evaluate_models
from member4_interpretability import analyze_models
from main import run_pipeline


@pytest.fixture
def synthetic_csv_path():
    """Create a temporary synthetic CSV with the exact schema expected by the pipeline."""
    np.random.seed(42)
    n_samples = 120

    data = pd.DataFrame({
        "loan_id": np.arange(1, n_samples + 1),
        "no_of_dependents": np.random.randint(0, 6, size=n_samples),
        "education": np.random.choice(["Graduate", "Not Graduate"], size=n_samples),
        "self_employed": np.random.choice(["Yes", "No"], size=n_samples),
        "income_annum": np.random.randint(200000, 10000000, size=n_samples),
        "loan_amount": np.random.randint(100000, 40000000, size=n_samples),
        "loan_term": np.random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], size=n_samples),
        "cibil_score": np.random.randint(300, 900, size=n_samples),
        "residential_assets_value": np.random.randint(0, 30000000, size=n_samples),
        "commercial_assets_value": np.random.randint(0, 20000000, size=n_samples),
        "luxury_assets_value": np.random.randint(0, 40000000, size=n_samples),
        "bank_asset_value": np.random.randint(0, 15000000, size=n_samples),
        "loan_status": np.random.choice(["Approved", "Rejected"], size=n_samples),
    })

    temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    data.to_csv(temp_file.name, index=False)
    temp_file.close()

    yield temp_file.name

    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


def test_prepare_data_shapes(synthetic_csv_path):
    (
        X, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler
    ) = prepare_data(synthetic_csv_path)

    assert len(X_train) == 96  # 80% of 120
    assert len(X_test) == 24   # 20% of 120
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert set(y_train.unique()).issubset({0, 1})


def test_train_models_keys(synthetic_csv_path):
    data_tuple = prepare_data(synthetic_csv_path)
    model_data = train_models(data_tuple)

    required_keys = [
        "X", "X_train", "X_test", "y_train", "y_test",
        "X_train_scaled", "X_test_scaled", "log_model", "knn_model", "dt_model",
        "log_pred", "knn_pred", "dt_pred",
        "log_train_time", "knn_train_time", "dt_train_time"
    ]
    for key in required_keys:
        assert key in model_data, f"Missing key: {key}"


def test_evaluate_models_output(synthetic_csv_path):
    data_tuple = prepare_data(synthetic_csv_path)
    model_data = train_models(data_tuple)
    results = evaluate_models(model_data, save_figures=False)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 3
    assert "Accuracy" in results.columns
    assert "F1 Score" in results.columns


def test_end_to_end_pipeline(synthetic_csv_path):
    comparison, coefficients = run_pipeline(synthetic_csv_path)

    assert isinstance(comparison, pd.DataFrame)
    assert isinstance(coefficients, pd.DataFrame)
    assert len(comparison) == 3
    assert set(comparison["Model"]) == {"Logistic Regression", "KNN", "Decision Tree"}
    assert "Coefficient" in coefficients.columns
