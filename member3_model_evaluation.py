"""Member 3: Evaluate model performance and plot comparison charts."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from member2_model_training import train_models


def evaluate_models(model_data):
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
            precision_score(y_test, log_pred),
            precision_score(y_test, knn_pred),
            precision_score(y_test, dt_pred)
        ],
        "Recall": [
            recall_score(y_test, log_pred),
            recall_score(y_test, knn_pred),
            recall_score(y_test, dt_pred)
        ],
        "F1 Score": [
            f1_score(y_test, log_pred),
            f1_score(y_test, knn_pred),
            f1_score(y_test, dt_pred)
        ]
    })

    print("\nModel Evaluation Results:")
    print(results.round(4))

    # Accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.bar(results["Model"], results["Accuracy"])
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    # F1 score comparison
    plt.figure(figsize=(8, 5))
    plt.bar(results["Model"], results["F1 Score"])
    plt.title("F1 Score Comparison")
    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
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
        print(confusion_matrix(y_test, prediction))

    return results


if __name__ == "__main__":
    data = train_models()
    evaluate_models(data)
