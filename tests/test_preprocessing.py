"""Unit tests for Member 1 preprocessing module."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from member1_data_preprocessing import prepare_data, get_data_filepath


@pytest.fixture
def sample_csv():
    """Create a temporary dataset fixture."""
    np.random.seed(10)
    n = 50
    df = pd.DataFrame({
        "loan_id": range(1, n + 1),
        "no_of_dependents": [2] * n,
        "education": ["Graduate"] * (n // 2) + ["Not Graduate"] * (n // 2),
        "self_employed": ["No"] * (n // 2) + ["Yes"] * (n // 2),
        "income_annum": [5000000] * n,
        "loan_amount": [15000000] * n,
        "loan_term": [10] * n,
        "cibil_score": [750] * n,
        "residential_assets_value": [2000000] * n,
        "commercial_assets_value": [1000000] * n,
        "luxury_assets_value": [3000000] * n,
        "bank_asset_value": [500000] * n,
        "loan_status": ["Approved"] * (n // 2) + ["Rejected"] * (n // 2),
    })
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    tmp.close()

    yield tmp.name

    if os.path.exists(tmp.name):
        os.remove(tmp.name)


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        prepare_data("non_existent_dataset_path_12345.csv")


def test_encoded_features(sample_csv):
    X, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(sample_csv)

    assert "loan_id" not in X.columns
    assert "loan_status" not in X.columns
    # Check dummy columns created
    assert any("education_" in col for col in X.columns)
    assert any("self_employed_" in col for col in X.columns)
