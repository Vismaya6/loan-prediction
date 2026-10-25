"""
Member 4: Explain model decisions and compare complexity/training time.
Author: Joshwin James (Member 4)
Project: Loan Approval Prediction
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from member2_model_training import train_models
from member3_model_evaluation import evaluate_models


def analyze_models(model_data, results):
    """
    Analyze model interpretability, complexity, and performance metrics.

    Parameters
    ----------
    model_data : dict
        Dictionary containing trained models, training times, and feature data.
    results : dict
        Evaluation metrics dictionary containing Model names, Accuracy, and F1 Score.
    """
    X = model_data["X"]
    log_model = model_data["log_model"]
    dt_model = model_data["dt_model"]

    # Logistic Regression feature coefficients.
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": log_model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print("\nLogistic Regression Coefficients:")
    print(coefficients)

    # Decision Tree complexity.
    print("\nDecision Tree Complexity:")
    print("Decision Tree Depth:", dt_model.get_depth())
    print("Number of Nodes:", dt_model.tree_.node_count)
    print("Number of Leaves:", dt_model.get_n_leaves())

    # Visualize the first three levels of the Decision Tree.
    plt.figure(figsize=(18, 10))
    plot_tree(
        dt_model,
        feature_names=list(X.columns),
        class_names=["Rejected", "Approved"],
        filled=True,
        rounded=True,
        max_depth=3
    )
    plt.tight_layout()
    plt.show()

    # Training time comparison.
    print("\nTraining Time")
    print("-------------------")
    print("Logistic Regression:", model_data["log_train_time"])
    print("KNN:", model_data["knn_train_time"])
    print("Decision Tree:", model_data["dt_train_time"])

    # Final comparison table.
    comparison = pd.DataFrame({
        "Model": results["Model"],
        "Accuracy": results["Accuracy"],
        "F1 Score": results["F1 Score"],
        "Interpretability": ["High", "Low", "High"],
        "Complexity": ["Low", "Low", "Medium"],
        "Training Time": [
            model_data["log_train_time"],
            model_data["knn_train_time"],
            model_data["dt_train_time"]
        ]
    })

    print("\nFinal Model Comparison:")
    print(comparison.round(4))


if __name__ == "__main__":
    model_data = train_models()
    results = evaluate_models(model_data)
    analyze_models(model_data, results)
