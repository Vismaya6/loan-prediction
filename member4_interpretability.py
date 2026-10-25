"""Member 4: Explain model decisions and compare complexity/training time."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from member2_model_training import train_models
from member3_model_evaluation import evaluate_models


def analyze_models(model_data, results, save_figures=True):
    """
    Analyze model interpretability, complexity metrics, and overall performance.

    Parameters
    ----------
    model_data : dict
        Model training output dictionary containing models and training times.
    results : pd.DataFrame
        Evaluation results DataFrame from evaluate_models().
    save_figures : bool, default=True
        Whether to save visualization figures to reports/.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Final comparison summary DataFrame and Logistic Regression coefficients.
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

    reports_dir = "reports"
    if save_figures:
        os.makedirs(reports_dir, exist_ok=True)

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
    if save_figures:
        plt.savefig(os.path.join(reports_dir, "decision_tree_visualization.png"), dpi=300)
    plt.show()

    # Training time comparison.
    print("\nTraining Time")
    print("-------------------")
    print("Logistic Regression:", f"{model_data['log_train_time']:.6f}s")
    print("KNN:", f"{model_data['knn_train_time']:.6f}s")
    print("Decision Tree:", f"{model_data['dt_train_time']:.6f}s")

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

    return comparison, coefficients


if __name__ == "__main__":
    model_data = train_models()
    results = evaluate_models(model_data)
    analyze_models(model_data, results)
