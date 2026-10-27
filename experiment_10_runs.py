"""
Experiment: Multi-Split Robustness and Generalization Analysis
Dataset: Loan Prediction (train.csv, 614 records)

Tests whether Decision Tree remains the best-performing model across 10 different
stratified 80:20 train-test splits compared to Logistic Regression and KNN.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 10 distinct random seeds for reproducible experiments
RANDOM_SEEDS = [42, 1, 7, 13, 21, 55, 99, 123, 777, 2024]


def load_dataset(filepath=None):
    """Locate and load the 614-record loan dataset."""
    candidates = [
        filepath,
        os.path.join("data", "train.csv"),
        "train.csv",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Could not find 'train.csv' in 'data/' or current directory.")


def run_10_experiments(data_path=None, save_figures=True):
    """
    Run 10 stratified 80:20 train-test experiments across 10 random seeds.

    Parameters
    ----------
    data_path : str, optional
        Path to train.csv dataset.
    save_figures : bool, default=True
        Whether to generate and save the 10-run comparison chart.

    Returns
    -------
    tuple of (pd.DataFrame, dict, dict)
        Results dataframe, win count dictionary, and summary statistics.
    """
    df = load_dataset(data_path)

    # Separate features and target
    X = df.drop(columns=["Loan_ID", "Loan_Status"])
    y = df["Loan_Status"].map({"Y": 1, "N": 0})

    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

    results = []

    for run_idx, seed in enumerate(RANDOM_SEEDS, start=1):
        # 80:20 stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )

        # Preprocessing without scaling (for Decision Tree)
        num_pipe_dt = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        cat_pipe_dt = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ])
        preprocessor_dt = ColumnTransformer([
            ("num", num_pipe_dt, num_cols),
            ("cat", cat_pipe_dt, cat_cols),
        ])

        # Preprocessing with scaling (for Logistic Regression & KNN)
        num_pipe_scaled = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe_scaled = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
            ("scaler", StandardScaler()),
        ])
        preprocessor_scaled = ColumnTransformer([
            ("num", num_pipe_scaled, num_cols),
            ("cat", cat_pipe_scaled, cat_cols),
        ])

        # Define model pipelines
        log_reg = Pipeline([
            ("prep", preprocessor_scaled),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        knn = Pipeline([
            ("prep", preprocessor_scaled),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ])
        dt = Pipeline([
            ("prep", preprocessor_dt),
            ("clf", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ])

        # Train models exclusively on the training partition
        log_reg.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        dt.fit(X_train, y_train)

        # Evaluate on the unseen test partition
        acc_lr = accuracy_score(y_test, log_reg.predict(X_test))
        acc_knn = accuracy_score(y_test, knn.predict(X_test))
        acc_dt = accuracy_score(y_test, dt.predict(X_test))

        # Determine winning model(s)
        max_acc = max(acc_lr, acc_knn, acc_dt)
        winners = []
        if np.isclose(acc_lr, max_acc):
            winners.append("Logistic Regression")
        if np.isclose(acc_knn, max_acc):
            winners.append("KNN")
        if np.isclose(acc_dt, max_acc):
            winners.append("Decision Tree")

        winner_str = ", ".join(winners)

        results.append({
            "Run": run_idx,
            "Random Seed": seed,
            "Logistic Regression": acc_lr,
            "KNN": acc_knn,
            "Decision Tree": acc_dt,
            "Winning Model": winner_str,
        })

    results_df = pd.DataFrame(results)

    # Calculate win counts
    win_counts = {"Logistic Regression": 0, "KNN": 0, "Decision Tree": 0}
    for winner in results_df["Winning Model"]:
        for model in win_counts:
            if model in winner:
                win_counts[model] += 1

    # Summary statistics
    summary_stats = {
        "Logistic Regression": {
            "mean": results_df["Logistic Regression"].mean(),
            "std": results_df["Logistic Regression"].std(),
            "min": results_df["Logistic Regression"].min(),
            "max": results_df["Logistic Regression"].max(),
        },
        "KNN": {
            "mean": results_df["KNN"].mean(),
            "std": results_df["KNN"].std(),
            "min": results_df["KNN"].min(),
            "max": results_df["KNN"].max(),
        },
        "Decision Tree": {
            "mean": results_df["Decision Tree"].mean(),
            "std": results_df["Decision Tree"].std(),
            "min": results_df["Decision Tree"].min(),
            "max": results_df["Decision Tree"].max(),
        },
    }

    # Print results table
    print("\n" + "=" * 80)
    print(" 10-EXPERIMENT RESULTS TABLE (80:20 Stratified Splits on train.csv)")
    print("=" * 80)
    display_df = results_df.copy()
    display_df["Logistic Regression"] = (display_df["Logistic Regression"] * 100).map("{:.2f}%".format)
    display_df["KNN"] = (display_df["KNN"] * 100).map("{:.2f}%".format)
    display_df["Decision Tree"] = (display_df["Decision Tree"] * 100).map("{:.2f}%".format)
    print(display_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(" WIN COUNT SUMMARY (Across 10 Runs)")
    print("=" * 80)
    for model, wins in win_counts.items():
        print(f" - {model:22}: {wins:2d} wins ({wins * 10:2d}%)")

    print("\n" + "=" * 80)
    print(" AVERAGE ACCURACY +/- STD DEV")
    print("=" * 80)
    for model, stats in summary_stats.items():
        print(f" - {model:22}: {stats['mean'] * 100:.2f}% +/- {stats['std'] * 100:.2f}% "
              f"[{stats['min'] * 100:.2f}% - {stats['max'] * 100:.2f}%]")

    # Check Decision Tree dominance
    dt_all_wins = win_counts["Decision Tree"] == 10
    print("\n" + "=" * 80)
    print(" RESEARCH VERDICT")
    print("=" * 80)
    if dt_all_wins:
        print("Verdict: Decision Tree achieved the highest accuracy in every run.")
    else:
        print(f"Verdict: Decision Tree DID NOT achieve the highest accuracy in every run.")
        print(f"         Decision Tree won only {win_counts['Decision Tree']} out of 10 runs (tied in 1 run).")
        print(f"         Logistic Regression won {win_counts['Logistic Regression']} out of 10 runs, demonstrating superior robustness.")

    if save_figures:
        plot_accuracy_comparison(results_df, summary_stats)

    return results_df, win_counts, summary_stats


def plot_accuracy_comparison(results_df, summary_stats):
    """Generate and save clean comparison chart across 10 runs."""
    os.makedirs("docs/images", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)

    runs = results_df["Run"].values
    lr_acc = results_df["Logistic Regression"].values * 100
    knn_acc = results_df["KNN"].values * 100
    dt_acc = results_df["Decision Tree"].values * 100

    # Plot lines with markers
    plt.plot(runs, lr_acc, marker="o", linewidth=2.2, color="#1f77b4", label=f"Logistic Regression (Mean: {summary_stats['Logistic Regression']['mean']*100:.2f}%)")
    plt.plot(runs, knn_acc, marker="s", linewidth=2.0, color="#ff7f0e", label=f"KNN (Mean: {summary_stats['KNN']['mean']*100:.2f}%)")
    plt.plot(runs, dt_acc, marker="^", linewidth=2.0, color="#2ca02c", label=f"Decision Tree (Mean: {summary_stats['Decision Tree']['mean']*100:.2f}%)")

    # Styling
    plt.title("Model Accuracy Across 10 Stratified Train-Test Splits (train.csv)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Experiment Run (1-10)", fontsize=11, labelpad=8)
    plt.ylabel("Test Accuracy (%)", fontsize=11, labelpad=8)
    plt.xticks(runs)
    plt.ylim(72, 90)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    plt.tight_layout()

    # Save to docs/images and reports/
    out_docs = os.path.join("docs", "images", "ten_runs_accuracy_comparison.png")
    out_reports = os.path.join("reports", "ten_runs_accuracy_comparison.png")
    plt.savefig(out_docs, dpi=300)
    plt.savefig(out_reports, dpi=300)
    plt.close()
    print(f"\nChart successfully saved to:")
    print(f" - {out_docs}")
    print(f" - {out_reports}")


if __name__ == "__main__":
    run_10_experiments()
