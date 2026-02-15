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
| Logistic Regression | 0.7805 | 0.8738 | 0.8000 | 0.7619 | 0.7805 | 0.5619 |
| Decision Tree | 0.7805 | 0.7798 | 0.7727 | 0.8095 | 0.7907 | 0.5609 |
| kNN | 0.7561 | 0.8810 | 0.7391 | 0.8095 | 0.7727 | 0.5132 |
| Naive Bayes | 0.7805 | 0.8905 | 0.7727 | 0.8095 | 0.7907 | 0.5609 |
| Random Forest (Ensemble) | 0.8049 | 0.9024 | 0.8095 | 0.8095 | 0.8095 | 0.6095 |
| XGBoost (Ensemble) | 0.7805 | 0.8786 | 0.7727 | 0.8095 | 0.7907 | 0.5609 |


### Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression provides a solid baseline with good interpretability. It achieves an accuracy of 0.7805 and a recall of 0.7619, correctly identifying most patients with heart disease. Its precision (0.8000) is the highest among individual (non-ensemble) models, indicating fewer false positives. The AUC of 0.8738 shows good overall class separability. As a linear model, it is well-suited for understanding feature contributions to the prediction. |
| Decision Tree | Decision Tree matches the same accuracy (0.7805) as Logistic Regression but with a higher recall (0.8095), catching more true positive cases. However, its AUC (0.7798) is the lowest among all models, suggesting weaker probabilistic ranking and potential overfitting to specific decision boundaries. The MCC of 0.5609 indicates moderate correlation between predicted and actual classes. Pruning or depth constraints could help improve its generalisation. |
| kNN | K-Nearest Neighbors has the lowest accuracy (0.7561) and precision (0.7391) among all models, resulting in the most false positives. However, its AUC (0.8810) is comparatively strong, indicating good probabilistic ranking despite the lower hard-classification performance. The MCC of 0.5132 is the weakest across all models. Performance could be improved with hyperparameter tuning of k values and distance metrics, and it benefits significantly from the StandardScaler feature scaling applied. |
| Naive Bayes | Gaussian Naive Bayes achieves an accuracy of 0.7805 with a recall of 0.8095 and precision of 0.7727, despite its strong feature independence assumption. Notably, its AUC (0.8905) is the second-highest overall, demonstrating strong probabilistic class separation ability. It is computationally efficient and serves as an effective baseline probabilistic classifier. The MCC of 0.5609 shows moderate predictive correlation. |
| Random Forest (Ensemble) | Random Forest delivers the best overall performance among all 6 models, achieving the highest accuracy (0.8049), precision (0.8095), AUC (0.9024), and MCC (0.6095). By aggregating predictions from 200 decision trees via bagging, it reduces variance and overfitting compared to a single Decision Tree. The balanced precision-recall (both 0.8095) and superior F1 score (0.8095) confirm it as the most reliable model on this dataset. |
| XGBoost (Ensemble) | XGBoost achieves an accuracy of 0.7805 with recall of 0.8095 and AUC of 0.8786, performing comparably to Decision Tree and Naive Bayes on hard classification metrics. While it did not outperform Random Forest on this dataset, its gradient-boosting approach with built-in regularisation still provides competitive results. The MCC of 0.5609 indicates moderate predictive correlation. With further hyperparameter tuning (learning rate, depth, number of estimators), its performance could potentially improve. |

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
