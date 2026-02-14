"""
Gautham Gutta
2025AA05552
Section 1
Heart Disease Prediction - Streamlit Web Application
=====================================================
Interactive web app for heart disease classification using 6 ML models.

Features:
    - CSV dataset upload
    - Model selection dropdown
    - Evaluation metrics display
    - Confusion matrix visualisation
    - Classification report
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
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

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)

# ─── Helper paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "heart-disease-dataset")
DEFAULT_CSV = os.path.join(DATA_DIR, "heart.csv")

MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "kNN": "kNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest (Ensemble)": "Random_Forest_Ensemble.pkl",
    "XGBoost (Ensemble)": "XGBoost_Ensemble.pkl",
}

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male; 0 = female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes; 0 = no)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (0-2)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect; 3 = reversible defect)",
}


# ─── Utility functions ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str):
    """Load a saved model from the model/ directory."""
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler."""
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_saved_metrics():
    """Load pre-computed metrics from training."""
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(MODEL_DIR, "confusion_matrices.json")) as f:
        cms = json.load(f)
    with open(os.path.join(MODEL_DIR, "classification_reports.json")) as f:
        reports = json.load(f)
    return metrics, cms, reports


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return metrics, confusion matrix, and report."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return metrics, cm, report, y_pred


def plot_confusion_matrix(cm, model_name):
    """Plot a heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease (0)", "Disease (1)"],
        yticklabels=["No Disease (0)", "Disease (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig


def retrain_models(df):
    """Re-train all 6 models on the uploaded dataset and return results."""
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            eval_metric="logloss", random_state=42,
        ),
    }

    results = {}
    for name, model in model_defs.items():
        model.fit(X_train_scaled, y_train)
        metrics, cm, report, y_pred = evaluate_model(model, X_test_scaled, y_test)
        results[name] = {
            "model": model,
            "metrics": metrics,
            "cm": cm,
            "report": report,
            "y_pred": y_pred,
        }

    return results, scaler, X_test_scaled, y_test


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Title and header
    st.title("❤️ Heart Disease Prediction App")
    st.markdown(
        """
        This application demonstrates **6 Machine Learning classification models** 
        trained on the [Heart Disease UCI dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
        Upload your own CSV or use the default dataset to explore model performance.
        """
    )

    st.divider()

    # ─── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Dataset upload
        st.subheader("📂 Dataset Upload")
        uploaded_file = st.file_uploader(
            "Upload a CSV file (must contain same columns + 'target')",
            type=["csv"],
            help="CSV should have the same features as the heart disease dataset and a 'target' column.",
        )

        use_uploaded = False
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if "target" not in uploaded_df.columns:
                    st.error("Uploaded CSV must contain a 'target' column.")
                else:
                    st.success(f"Uploaded dataset: {uploaded_df.shape[0]} rows, {uploaded_df.shape[1]} columns")
                    use_uploaded = True
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

        st.divider()

        # Model selection
        st.subheader("🤖 Model Selection")
        model_names = list(MODEL_FILES.keys())
        selected_model = st.selectbox("Select a model:", model_names)

        st.divider()

        # Dataset info
        st.subheader("📊 Dataset Features")
        for feat, desc in FEATURE_DESCRIPTIONS.items():
            st.markdown(f"**{feat}**: {desc}")

    # ─── Main Content ─────────────────────────────────────────────────────────
    if use_uploaded:
        # Re-train models on uploaded data
        with st.spinner("Training models on uploaded data..."):
            results, scaler, X_test, y_test = retrain_models(uploaded_df)
        source_label = "Uploaded Dataset"
    else:
        # Use pre-trained models
        results = None
        source_label = "Default Heart Disease Dataset"

    st.info(f"📁 Data source: **{source_label}**")

    # ─── Tab layout ───────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["📈 Model Performance", "🔍 Detailed Analysis", "📊 Comparison Table"]
    )

    # ─────────── TAB 1: Selected Model Performance ───────────────────────────
    with tab1:
        st.header(f"Model: {selected_model}")

        if results:
            # Using uploaded data results
            res = results[selected_model]
            metrics = res["metrics"]
            cm = res["cm"]
            report = res["report"]
        else:
            # Using pre-computed metrics
            all_metrics, all_cms, all_reports = load_saved_metrics()
            metrics = all_metrics[selected_model]
            cm = np.array(all_cms[selected_model])
            report = all_reports[selected_model]

        # Metric cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
        col3.metric("Precision", f"{metrics['Precision']:.4f}")
        col4.metric("Recall", f"{metrics['Recall']:.4f}")
        col5.metric("F1 Score", f"{metrics['F1']:.4f}")
        col6.metric("MCC", f"{metrics['MCC']:.4f}")

        st.divider()

        # Confusion Matrix & Classification Report side by side
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(cm, selected_model)
            st.pyplot(fig)

        with right_col:
            st.subheader("Classification Report")
            # Convert report dict to a nice dataframe
            report_df = pd.DataFrame(report).transpose()
            # Keep only meaningful rows
            display_rows = [k for k in report_df.index if k not in ("accuracy",)]
            report_display = report_df.loc[display_rows]
            report_display = report_display.round(4)
            st.dataframe(report_display, use_container_width=True)

    # ─────────── TAB 2: Detailed Analysis ────────────────────────────────────
    with tab2:
        st.header("Detailed Model Analysis")

        if results:
            all_metrics_dict = {name: r["metrics"] for name, r in results.items()}
        else:
            all_metrics_dict, _, _ = load_saved_metrics()

        # Bar charts for each metric
        metrics_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        models_list = list(all_metrics_dict.keys())

        for i in range(0, len(metrics_names), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(metrics_names):
                    metric_name = metrics_names[i + j]
                    with cols[j]:
                        values = [all_metrics_dict[m][metric_name] for m in models_list]
                        fig, ax = plt.subplots(figsize=(6, 4))
                        bars = ax.barh(models_list, values, color=sns.color_palette("viridis", len(models_list)))
                        ax.set_xlim(0, 1.05)
                        ax.set_title(f"{metric_name} Comparison", fontsize=14, fontweight="bold")
                        ax.set_xlabel(metric_name)
                        for bar, val in zip(bars, values):
                            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                                    va="center", fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig)

        # Observations
        st.divider()
        st.header("📝 Model Observations")

        observations = {
            "Logistic Regression": (
                "Logistic Regression provides a solid baseline with good interpretability. "
                "It achieves reasonable accuracy and high recall, meaning it correctly identifies "
                "most patients with heart disease. However, its precision is lower compared to "
                "tree-based models, leading to more false positives. The high AUC indicates good "
                "separability between classes."
            ),
            "Decision Tree": (
                "Decision Tree achieves very high accuracy on this dataset, with perfect precision "
                "and near-perfect recall. It captures non-linear decision boundaries well. "
                "However, it may be prone to overfitting on training data, especially without "
                "pruning or depth constraints."
            ),
            "kNN": (
                "K-Nearest Neighbors shows moderate performance, achieving a balance between "
                "precision and recall. The model benefits from feature scaling and works well "
                "with the standardised features. Its performance could be improved with "
                "hyperparameter tuning (different k values and distance metrics)."
            ),
            "Naive Bayes": (
                "Gaussian Naive Bayes offers decent performance despite its strong independence "
                "assumption between features. It has high recall but lower precision, suggesting "
                "it tends to classify more instances as positive (disease present). It is fast "
                "and works well as a baseline probabilistic classifier."
            ),
            "Random Forest (Ensemble)": (
                "Random Forest, an ensemble of decision trees, achieves excellent performance "
                "across all metrics. By aggregating multiple trees, it reduces variance and "
                "overfitting risk compared to a single decision tree. It demonstrates the power "
                "of bagging-based ensemble methods for tabular data."
            ),
            "XGBoost (Ensemble)": (
                "XGBoost, a gradient-boosted ensemble, delivers top-tier performance. It excels "
                "in capturing complex feature interactions through sequential boosting. Its "
                "regularisation parameters help prevent overfitting, and it consistently shows "
                "strong results across all evaluation metrics."
            ),
        }

        for model_name, obs in observations.items():
            with st.expander(f"🔹 {model_name}", expanded=False):
                if results:
                    m = results[model_name]["metrics"]
                else:
                    m = all_metrics_dict[model_name]
                st.markdown(f"**Performance Summary**: Accuracy={m['Accuracy']:.4f}, F1={m['F1']:.4f}, MCC={m['MCC']:.4f}")
                st.markdown(obs)

    # ─────────── TAB 3: Comparison Table ─────────────────────────────────────
    with tab3:
        st.header("📊 Model Comparison Table")

        if results:
            all_metrics_dict_tab3 = {name: r["metrics"] for name, r in results.items()}
        else:
            all_metrics_dict_tab3, _, _ = load_saved_metrics()

        comparison_data = []
        for model_name, m in all_metrics_dict_tab3.items():
            comparison_data.append({
                "ML Model": model_name,
                "Accuracy": m["Accuracy"],
                "AUC": m["AUC"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "MCC": m["MCC"],
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index("ML Model")

        # Style the table - highlight best values
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color="#90EE90").format("{:.4f}"),
            use_container_width=True,
        )

        # Best model summary
        st.divider()
        st.subheader("🏆 Best Model per Metric")
        for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            best_model = comparison_df[metric].idxmax()
            best_val = comparison_df[metric].max()
            st.markdown(f"- **{metric}**: {best_model} ({best_val:.4f})")

    # ─── Footer ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Heart Disease Prediction App | BITS Pilani M.Tech (AIML) | ML Assignment 2</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
