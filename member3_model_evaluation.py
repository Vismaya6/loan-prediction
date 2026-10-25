"""Member 3: Evaluate model performance and plot comparison charts."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from member2_model_training import train_models


def evaluate_models(model_data, save_figures=True):
    """
    Evaluate classification models and produce performance summaries and plots.

    Parameters
    ----------
    model_data : dict
        Model training output dictionary containing true labels and predictions.
    save_figures : bool, default=True
        Whether to save visualization figures to the reports/ directory.

    Returns
    -------
    pd.DataFrame
        DataFrame of Accuracy, Precision, Recall, and F1 scores per model.
    """
    y_test = model_data["y_test"]
    log_pred = model_data["log_pred"]
    knn_pred = model_data["knn_pred"]
    dt_pred = model_data["dt_pred"]

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "Decision Tree"],
        "Accuracy": [
            accuracy_score(y_test, log_pred),
            accuracy_score(y_test, knn_pred),
            accuracy_score(y_test, dt_pred)
        ],
        "Precision": [
            precision_score(y_test, log_pred, zero_division=0),
            precision_score(y_test, knn_pred, zero_division=0),
            precision_score(y_test, dt_pred, zero_division=0)
        ],
        "Recall": [
            recall_score(y_test, log_pred, zero_division=0),
            recall_score(y_test, knn_pred, zero_division=0),
            recall_score(y_test, dt_pred, zero_division=0)
        ],
        "F1 Score": [
            f1_score(y_test, log_pred, zero_division=0),
            f1_score(y_test, knn_pred, zero_division=0),
            f1_score(y_test, dt_pred, zero_division=0)
        ]
    })

    print("\nModel Evaluation Results:")
    print(results.round(4))

    reports_dir = "reports"
    if save_figures:
        os.makedirs(reports_dir, exist_ok=True)

    # Accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.bar(results["Model"], results["Accuracy"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    min_acc = max(0.0, float(results["Accuracy"].min()) - 0.05)
    plt.ylim(min_acc, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(reports_dir, "accuracy_comparison.png"), dpi=300)
    plt.show()

    # F1 score comparison
    plt.figure(figsize=(8, 5))
    plt.bar(results["Model"], results["F1 Score"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("F1 Score Comparison")
    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    min_f1 = max(0.0, float(results["F1 Score"].min()) - 0.05)
    plt.ylim(min_f1, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(reports_dir, "f1_score_comparison.png"), dpi=300)
    plt.show()

    # Confusion matrices
    predictions = {
        "Logistic Regression": log_pred,
        "KNN": knn_pred,
        "Decision Tree": dt_pred
    }
    print("\nConfusion Matrices:")
    for name, prediction in predictions.items():
        print(f"\n{name}")
        cm = confusion_matrix(y_test, prediction)
        print(cm)

    return results


if __name__ == "__main__":
    data = train_models()
    evaluate_models(data)
