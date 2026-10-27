"""
Loan Approval Prediction - Complete Team ML Pipeline Runner

Integrates contributions from all four team members:
- Member 1: Data preprocessing and feature scaling
- Member 2: Model training (Logistic Regression, KNN, Decision Tree)
- Member 3: Model evaluation (metrics, confusion matrices, charts)
- Member 4: Model interpretability and complexity benchmarking
"""

import sys
import os

from member1_data_preprocessing import prepare_data
from member2_model_training import train_models
from member3_model_evaluation import evaluate_models
from member4_interpretability import analyze_models


def run_pipeline(data_path=None):
    """
    Execute the end-to-end Machine Learning pipeline.

    Parameters
    ----------
    data_path : str, optional
        Path to the dataset CSV file. If None, resolves from default candidate paths.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Final comparison summary DataFrame and Logistic Regression coefficients DataFrame.
    """
    print("=" * 70)
    print(" LOAN APPROVAL PREDICTION - MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # Phase 1: Data Preprocessing
    print("\n[PHASE 1] Preprocessing Data (Member 1)...")
    data_tuple = prepare_data(data_path)

    # Phase 2: Model Training
    print("\n[PHASE 2] Training Models (Member 2)...")
    model_data = train_models(data_tuple)

    # Phase 3: Model Evaluation
    print("\n[PHASE 3] Evaluating Model Performance (Member 3)...")
    results = evaluate_models(model_data)

    # Phase 4: Interpretability & Final Comparison
    print("\n[PHASE 4] Analyzing Interpretability & Model Complexity (Member 4)...")
    comparison, coefficients = analyze_models(model_data, results)

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print(" Figures saved in: reports/")
    print("=" * 70)

    return comparison, coefficients


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--multi-split", "--experiment", "train.csv"):
        from experiment_10_runs import run_10_experiments
        run_10_experiments()
    else:
        cli_path = sys.argv[1] if len(sys.argv) > 1 else None
        run_pipeline(cli_path)

