"""Member 1: Load, inspect, and preprocess the loan approval dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE = "loan_approval_dataset (1).csv"


def prepare_data():
    # Load and clean the dataset.
    data = pd.read_csv(DATA_FILE)
    data.columns = data.columns.str.strip()

    for col in data.select_dtypes("object"):
        data[col] = data[col].str.strip()

    print("Dataset Shape:", data.shape)
    print("\nDataset Information:")
    data.info()

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nLoan Status:")
    print(data["loan_status"].value_counts())

    # Separate features and target.
    X = data.drop(["loan_status", "loan_id"], axis=1)
    y = data["loan_status"].map({"Approved": 1, "Rejected": 0})

    # Convert categorical features into numeric columns.
    X = pd.get_dummies(
        X,
        columns=["education", "self_employed"],
        drop_first=True
    )

    print("\nNumber of features:", X.shape[1])

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features for Logistic Regression and KNN.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler
    )


if __name__ == "__main__":
    prepare_data()
