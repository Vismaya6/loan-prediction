"""Member 2: Create, train, and run predictions for the three models."""

import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from member1_data_preprocessing import prepare_data


def train_models(data_tuple=None):
    """
    Train Logistic Regression, KNN, and Decision Tree models.

    Parameters
    ----------
    data_tuple : tuple, optional
        Preprocessed tuple from prepare_data(). If None, calls prepare_data().

    Returns
    -------
    dict
        Dictionary containing trained models, predictions, times, and data splits.
    """
    if data_tuple is None:
        (
            X, X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled, scaler
        ) = prepare_data()
    else:
        (
            X, X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled, scaler
        ) = data_tuple

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Logistic Regression
    start = time.time()
    log_model.fit(X_train_scaled, y_train)
    log_train_time = time.time() - start

    # K-Nearest Neighbors
    start = time.time()
    knn_model.fit(X_train_scaled, y_train)
    knn_train_time = time.time() - start

    # Decision Tree (uses unscaled features)
    start = time.time()
    dt_model.fit(X_train, y_train)
    dt_train_time = time.time() - start

    print("\nModels trained successfully.")

    # Predict on the test data.
    log_pred = log_model.predict(X_test_scaled)
    knn_pred = knn_model.predict(X_test_scaled)
    dt_pred = dt_model.predict(X_test)

    print("Predictions completed.")

    return {
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "log_model": log_model,
        "knn_model": knn_model,
        "dt_model": dt_model,
        "log_pred": log_pred,
        "knn_pred": knn_pred,
        "dt_pred": dt_pred,
        "log_train_time": log_train_time,
        "knn_train_time": knn_train_time,
        "dt_train_time": dt_train_time,
    }


if __name__ == "__main__":
    train_models()
