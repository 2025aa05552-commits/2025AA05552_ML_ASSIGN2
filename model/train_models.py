"""
Gautham Gutta
2025AA05552
Section 1
Heart Disease Classification - Model Training Script
=====================================================
Trains 6 classification models on the Heart Disease dataset and saves them
along with evaluation metrics and the fitted scaler.

Models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor Classifier
    4. Gaussian Naive Bayes
    5. Random Forest (Ensemble)
    6. XGBoost (Ensemble)

Evaluation Metrics (per model):
    Accuracy, AUC, Precision, Recall, F1 Score, MCC
"""

import os

import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


def load_and_preprocess(csv_path: str):
    """Load heart disease CSV and return train/test splits with scaling."""
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-test split (80-20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler, X.columns.tolist()


def evaluate_model(model, X_test, y_test):
    """Return a dict of evaluation metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    return metrics, cm, report


def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    csv_path = os.path.join(project_dir, "heart-disease-dataset", "heart.csv")

    # Load data
    X_train, X_test, X_train_raw, X_test_raw, y_train, y_test, scaler, feature_names = load_and_preprocess(
        csv_path
    )

    # Save train and test splits as CSV files
    train_df = X_train_raw.copy()
    train_df["target"] = y_train.values
    train_csv_path = os.path.join(project_dir, "train.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"Train data saved to {train_csv_path} ({train_df.shape[0]} rows)")

    test_df = X_test_raw.copy()
    test_df["target"] = y_test.values
    test_csv_path = os.path.join(project_dir, "test.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Test data saved to {test_csv_path} ({test_df.shape[0]} rows)")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    all_metrics = {}
    all_confusion_matrices = {}
    all_reports = {}

    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)

        metrics, cm, report = evaluate_model(model, X_test, y_test)
        all_metrics[name] = metrics
        all_confusion_matrices[name] = cm
        all_reports[name] = report

        # Save model
        model_file = os.path.join(base_dir, f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        print(f"  {name}: {metrics}")

    # Save scaler
    with open(os.path.join(base_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save feature names
    with open(os.path.join(base_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save metrics
    with open(os.path.join(base_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save confusion matrices
    with open(os.path.join(base_dir, "confusion_matrices.json"), "w") as f:
        json.dump(all_confusion_matrices, f, indent=2)

    # Save classification reports
    with open(os.path.join(base_dir, "classification_reports.json"), "w") as f:
        json.dump(all_reports, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 90)
    print("MODEL COMPARISON TABLE")
    print("=" * 90)
    header = f"{'Model':<28} {'Accuracy':>8} {'AUC':>8} {'Precision':>9} {'Recall':>8} {'F1':>8} {'MCC':>8}"
    print(header)
    print("-" * 90)
    for name, m in all_metrics.items():
        row = f"{name:<28} {m['Accuracy']:>8.4f} {m['AUC']:>8.4f} {m['Precision']:>9.4f} {m['Recall']:>8.4f} {m['F1']:>8.4f} {m['MCC']:>8.4f}"
        print(row)
    print("=" * 90)


if __name__ == "__main__":
    main()
