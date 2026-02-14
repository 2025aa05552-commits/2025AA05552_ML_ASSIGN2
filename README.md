# GAUTHAM GUTTA
# 2025AA05552
# SECTION = 1
# ML Assignment 2

## a. Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early and accurate prediction of heart disease can significantly improve patient outcomes through timely intervention. This project implements and compares **6 machine learning classification models** on a heart disease dataset to predict whether a patient has heart disease (binary classification: 0 = no disease, 1 = disease). The goal is to evaluate each model's strengths and weaknesses using standard classification metrics and deploy an interactive Streamlit web application for real-time exploration.

## b. Dataset Description

| Property | Value |
|----------|-------|
| **Source** | [Heart Disease UCI — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Instances** | 1,025 |
| **Features** | 13 (+ 1 target) |
| **Target** | `target` — 0 (No Disease) / 1 (Disease) |
| **Class Distribution** | 499 (No Disease) / 526 (Disease) |
| **Missing Values** | None |

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | Sex (1 = male; 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes; 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect; 3 = reversible defect) |

## c. Models Used

All 6 models were trained with an 80/20 stratified train-test split and StandardScaler feature scaling.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| Decision Tree | 0.9854 | 0.9857 | 1.0000 | 0.9714 | 0.9855 | 0.9712 |
| kNN | 0.8634 | 0.9629 | 0.8738 | 0.8571 | 0.8654 | 0.7269 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8070 | 0.8762 | 0.8402 | 0.6602 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression provides a solid baseline with good interpretability. It achieves reasonable accuracy (0.8098) and high recall (0.9143), meaning it correctly identifies most patients with heart disease. However, its precision is comparatively lower (0.7619), leading to more false positives. The high AUC (0.9298) indicates good overall separability between classes. It is a reliable linear model suitable for understanding feature contributions. |
| Decision Tree | Decision Tree achieves excellent accuracy (0.9854) with perfect precision (1.0) and near-perfect recall (0.9714). It effectively captures non-linear decision boundaries inherent in clinical data. The strong MCC (0.9712) indicates a well-balanced classifier. However, single decision trees can be prone to overfitting, especially without pruning or depth constraints. |
| kNN | K-Nearest Neighbors shows moderate performance with a balance between precision (0.8738) and recall (0.8571). The model benefits considerably from feature scaling (StandardScaler). Its AUC (0.9629) is strong, indicating good probabilistic ranking. Performance could be further improved with hyperparameter tuning of k and distance metrics. |
| Naive Bayes | Gaussian Naive Bayes offers decent performance (accuracy 0.8293) despite its strong feature independence assumption. It has high recall (0.8762) but somewhat lower precision (0.8070), suggesting it tends to classify more instances as positive. It is computationally efficient and works well as a baseline probabilistic classifier. The AUC (0.9043) shows reasonable class separation ability. |
| Random Forest (Ensemble) | Random Forest, a bagging-based ensemble of decision trees, achieves perfect performance across all metrics. By aggregating predictions from 200 trees, it effectively reduces variance and overfitting risk compared to a single decision tree. This demonstrates the power of ensemble bagging methods on structured/tabular clinical data. The perfect scores indicate strong generalisation on this dataset's test split. |
| XGBoost (Ensemble) | XGBoost, a gradient-boosted ensemble model, also delivers perfect performance across all metrics. It excels in capturing complex feature interactions through sequential boosting, where each new tree corrects errors from previous ones. Its built-in regularisation parameters help prevent overfitting. XGBoost consistently ranks as a top performer for tabular data classification tasks. |

## Project Structure

```
project-folder/
│── app.py                          # Streamlit web application
│── requirements.txt                # Python dependencies
│── README.md                       # Project documentation
│── heart-disease-dataset/
│   └── heart.csv                   # Heart disease dataset
│── model/
│   ├── train_models.py             # Model training script
│   ├── Logistic_Regression.pkl     # Saved Logistic Regression model
│   ├── Decision_Tree.pkl           # Saved Decision Tree model
│   ├── kNN.pkl                     # Saved kNN model
│   ├── Naive_Bayes.pkl             # Saved Naive Bayes model
│   ├── Random_Forest_Ensemble.pkl  # Saved Random Forest model
│   ├── XGBoost_Ensemble.pkl        # Saved XGBoost model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── feature_names.json          # Feature column names
│   ├── metrics.json                # Pre-computed evaluation metrics
│   ├── confusion_matrices.json     # Pre-computed confusion matrices
│   └── classification_reports.json # Pre-computed classification reports
```

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Re-train models:
   ```bash
   python model/train_models.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Streamlit App Features

- **CSV Upload**: Upload your own test data (CSV with same features + target column)
- **Model Selection**: Dropdown to choose from all 6 implemented models
- **Evaluation Metrics**: Display of Accuracy, AUC, Precision, Recall, F1, and MCC
- **Confusion Matrix**: Visual heatmap of the confusion matrix for the selected model
- **Classification Report**: Detailed per-class precision, recall, and F1 scores
- **Comparison Table**: Side-by-side comparison of all 6 models across all metrics
- **Bar Charts**: Visual comparison of each metric across all models
- **Model Observations**: Detailed analysis of each model's performance characteristics

## Technologies Used

- Python 3.x
- Streamlit
- scikit-learn
- XGBoost
- pandas, NumPy
- Matplotlib, Seaborn
