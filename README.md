# 💳 Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-oriented Machine Learning system to detect fraudulent credit card transactions in real-time. Built for the **Codec Technologies Data Science Internship**, this project addresses extreme class imbalance using **SMOTE** (strictly applied to training data to prevent leakage), evaluates multiple supervised and unsupervised architectures, and deploys an interactive **Streamlit web application** for real-time risk assessment.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Data Preprocessing & Leakage Prevention](#data-preprocessing--leakage-prevention)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation & Real Results](#model-evaluation--real-results)
- [Why Accuracy Is Not Enough](#why-accuracy-is-not-enough)
- [Business Interpretation](#business-interpretation)
- [Streamlit Web Application](#streamlit-web-application)
- [How to Run](#how-to-run)
- [GitHub Repository Setup](#github-repository-setup)
- [Future Improvements](#future-improvements)

---

## 📖 Overview
In modern digital economies, billions of credit card transactions occur daily. While fraudulent activities account for less than 0.2% of total transaction volume, their absolute financial impact amounts to tens of billions of dollars annually. This system leverages supervised machine learning and anomaly detection algorithms to identify subtle, non-linear indicators of fraudulent card usage without burdening legitimate cardholders with high false alarms.

---

## 🎯 Problem Statement
Financial institutions face a fundamental tension between **fraud prevention** and **customer friction**:
1. **Extreme Class Imbalance:** Fraudulent transactions represent only ~0.17% of all payment events. Standard machine learning algorithms optimize for overall accuracy, frequently predicting all transactions as legitimate.
2. **High Cost of False Negatives (FN):** A fraudulent transaction missed by the system results in direct financial loss, chargeback processing costs, and customer churn.
3. **High Cost of False Positives (FP):** Declining legitimate transactions frustrates cardholders, damages brand loyalty, and overwhelms manual fraud investigation teams.

The objective is to train a model that maximizes **Recall** (catching frauds) while maintaining high **Precision** (minimizing customer disruption).

---

## 🚀 Objectives
- Build an automated, reproducible data preparation and feature scaling pipeline.
- Implement **SMOTE (Synthetic Minority Over-sampling Technique)** strictly on the training partition to prevent synthetic data leakage.
- Train and benchmark three models:
  - **Logistic Regression** (Linear baseline)
  - **Random Forest Classifier** (Non-linear ensemble)
  - **Isolation Forest** (Unsupervised anomaly detection)
- Evaluate models using real-world fraud metrics: Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall curves.
- Build and deploy an interactive **Streamlit web app** featuring instant demo transaction presets and risk scoring.

---

## 📂 Dataset
The project utilizes the benchmark [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) published by the Machine Learning Group at Université Libre de Bruxelles (ULB).

- **Total Transactions:** 284,807 transactions over a 48-hour period in September 2013 by European cardholders.
- **Unique Transactions (after deduplication):** 283,726.
- **Class Breakdown:**
  - Legitimate (`Class = 0`): 283,253 (99.833%)
  - Fraudulent (`Class = 1`): 473 (0.167%)
- **Features:**
  - `Time`: Number of seconds elapsed between the transaction and the first transaction in the dataset.
  - `Amount`: Transaction monetary value in USD.
  - `V1` to `V28`: 28 numerical features resulting from a Principal Component Analysis (PCA) transformation applied to safeguard customer confidentiality and sensitive card details.
  - `Class`: Binary target variable (`0` = Legitimate, `1` = Fraudulent).

> **Note:** The raw dataset (`creditcard.csv`, ~144 MB) exceeds GitHub's 100 MB file limit and is included in `.gitignore`. See [How to Run](#how-to-run) to automatically download it.

---

## 🛠️ Technologies Used
- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
- **Class Imbalance Handling:** `imbalanced-learn` (`imblearn.over_sampling.SMOTE`)
- **Model Serialization:** `joblib`
- **Web Deployment:** `streamlit`

---

## 📁 Project Structure
```text
fraud-detection-system/
│
├── data/
│   ├── creditcard.csv                 # Raw Kaggle/ULB dataset (downloaded locally)
│   └── sample_transactions.json       # Pre-extracted demo transactions for Streamlit
│
├── figures/                           # High-resolution generated plots & charts
│   ├── 01_class_distribution.png
│   ├── 02_amount_distribution.png
│   ├── 03_time_distribution.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_amount_boxplot.png
│   ├── 06_logistic_regression_confusion_matrix.png
│   ├── 06_random_forest_confusion_matrix.png
│   ├── 07_roc_pr_curves.png
│   └── 08_feature_importance.png
│
├── models/
│   ├── fraud_detection_model.pkl      # Production pipeline (model + scalers + feature order)
│   └── evaluation_metrics.json        # Test evaluation scores
│
├── notebooks/
│   └── fraud_detection.ipynb          # End-to-end interactive Jupyter Notebook
│
├── src/
│   └── train_model.py                 # Full reproducible Python ML training pipeline
│
├── app.py                             # Interactive Streamlit Web Application
├── requirements.txt                   # Pinned project dependencies
├── README.md                          # Comprehensive project documentation
└── .gitignore                         # Standard git ignore rules
```

---

## ⚙️ Data Preprocessing & Leakage Prevention

### 1. Missing Values & Duplicate Cleaning
- The dataset contained **0 missing values**.
- Identified and removed **1,081 duplicate transactions**, preserving 283,726 unique transactions.

### 2. Feature Scaling
- Features `V1`–`V28` are already zero-centered and scaled via PCA.
- `Amount` and `Time` exhibit significant skew and monetary outliers.
- We apply `RobustScaler` (which uses the median and Interquartile Range) rather than `StandardScaler` to prevent extreme transaction values from biasing scaling parameters.
- **Strict Leakage Prevention:** The scalers are fitted **exclusively on the training split** and then applied to transform the test split.

### 3. Stratified Train/Test Split
- An 80% train / 20% test split was executed using `random_state=42` and `stratify=y`.
- Stratification guarantees that the 0.167% minority fraud ratio is preserved identically across training (378 fraud cases) and testing (95 fraud cases).

### 4. SMOTE (Synthetic Minority Over-sampling Technique)
- **Workflow Applied:**
  ```text
  Original Dataset (283,726 records)
          ↓
  Stratified Train/Test Split (80% / 20%)
          ↓
  Training Data ONLY (226,602 Legit : 378 Fraud)
          ↓
  Apply SMOTE(random_state=42)
          ↓
  Balanced Training Set (226,602 Legit : 226,602 Fraud)
          ↓
  Model Training
          ↓
  Model Evaluation on UNTOUCHED Test Set (56,651 Legit : 95 Fraud)
  ```
- **Why SMOTE is NEVER applied to the Test Set:**
  1. Testing on synthetic records violates the core principle of evaluating models on real customer transaction behavior.
  2. Synthesizing test points with neighbors from the training set creates severe data leakage, artificially inflating precision and recall scores.

---

## 📊 Exploratory Data Analysis (EDA)

1. **Class Distribution:** Visualizes the extreme 99.83% vs 0.17% imbalance on a logarithmic scale.
2. **Transaction Amount Analysis:** Fraudulent transactions average $123.87 compared to $88.41 for legitimate transactions, but fraud amounts rarely exceed $2,125.87 to avoid automatic bank alerts.
3. **Temporal Cycle Analysis:** Legitimate transactions show steep drops between 02:00 AM and 06:00 AM, whereas fraudulent attempts occur steadily throughout early morning hours when cardholders are asleep.
4. **Correlation Analysis:** Identified top PCA components inversely correlated with fraud (`V17`, `V14`, `V12`, `V10`) and positively correlated (`V11`, `V4`, `V2`).
5. **Amount vs Fraud Boxplot:** Confirmed tighter interquartile spending distributions for fraud.

All figures are automatically generated by the training pipeline and stored in `figures/`.

---

## 🤖 Machine Learning Models

1. **Logistic Regression:** Linear probabilistic classifier trained on the SMOTE-resampled training data as a benchmark.
2. **Random Forest Classifier:** Bagging ensemble of 100 decision trees (`max_depth=15`) capable of modeling non-linear feature interactions and high-dimensional PCA structures.
3. **Isolation Forest:** Unsupervised anomaly detection algorithm that recursively partitions features. Transactions isolated in fewer splits are flagged as anomalies without needing target labels during training.

---

## 📈 Model Evaluation & Real Results

All metrics below were computed directly on the untouched test partition (56,746 total transactions: 56,651 legitimate and 95 fraudulent):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 97.40% | 0.0537 | **87.37%** | 0.1012 | 0.9601 |
| **Random Forest** | **99.92%** | **0.7653** | **78.95%** | **0.7772** | **0.9732** |
| **Isolation Forest** | 99.74% | 0.2353 | 25.26% | 0.2436 | 0.9404 |

### Confusion Matrix Breakdown (Test Set)

#### Random Forest (Production Model)
- **True Negatives (Legitimate correctly identified):** 56,628
- **False Positives (Legitimate incorrectly flagged as fraud):** 23
- **False Negatives (Fraud missed by model):** 20
- **True Positives (Fraud correctly detected):** 75

#### Logistic Regression
- **True Negatives:** 55,188
- **False Positives:** 1,463
- **False Negatives:** 12
- **True Positives:** 83

---

## 💡 Why Accuracy Is Not Enough
If a model blindly predicted all transactions as legitimate, it would achieve **99.83% accuracy**, yet allow 100% of financial fraud to occur. 

In our evaluation:
- Both Logistic Regression (97.40%) and Random Forest (99.92%) boast high accuracy.
- However, Logistic Regression produces **1,463 false positives** (Precision = 5.37%). Out of every 100 fraud alerts generated by Logistic Regression, only 5 are actually fraud!
- In an enterprise banking environment, 1,463 false alarms on just 56,000 transactions would overwhelm fraud teams and trigger widespread customer friction.
- **Random Forest achieves 76.53% Precision and 78.95% Recall (F1 = 0.7772)**, catching nearly 4 out of 5 frauds with only 23 false alarms across 56,651 normal transactions.

---

## 🏢 Business Interpretation
- **Precision (76.53%):** When Random Forest sounds an alarm, it is correct over 3 out of 4 times. This keeps manual review queues manageable.
- **Recall (78.95%):** The bank intercepts ~79% of all stolen cards and unauthorized charges before settlement.
- **False Negatives (20):** Missed transactions that cost money. In production, these are mitigated by secondary controls like 3D-Secure and cardholder confirmation SMS.
- **False Positives (23):** Only 0.04% of normal customers experience temporary friction.

---

## 💻 Streamlit Web Application

The interactive web application (`app.py`) provides:
- **Instant Demo Presets:** Buttons to load real legitimate and fraudulent transactions with a single click.
- **Full Parameter Control:** Numerical inputs for `Time`, `Amount`, and `V1`–`V28`.
- **Probabilistic Scoring:** Displays the exact fraud probability (e.g., `0.02%` vs `99.99%`).
- **Dynamic Risk Gauge:** Visual Low / Medium / Critical risk indicators.
- **Recommended Business Action:** Immediate Block, Step-Up OTP, or Instant Approval.
- **Educational Guidance:** Clear explanations of PCA features and compliance safeguards.

---

## 🚀 How to Run

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/<your-username>/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
```

### 2. Train Models & Generate Assets
Run the end-to-end training pipeline. If `creditcard.csv` is not present, download it from Kaggle and place it in `data/`:
```bash
python src/train_model.py
```
This script will:
- Clean the data and split train/test sets
- Apply SMOTE to the training set
- Train Logistic Regression, Random Forest, and Isolation Forest
- Save evaluation figures to `figures/`
- Save the final model to `models/fraud_detection_model.pkl`

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501` (or port 8502).

### 4. Run Jupyter Notebook
```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

---

## 📦 GitHub Repository Setup

To push this project to your GitHub profile:
```bash
git init
git add .
git commit -m "feat: complete credit card fraud detection system with SMOTE, ML models, and Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/fraud-detection-system.git
git push -u origin main
```
> **Notice:** The `.gitignore` file automatically excludes the 144 MB `creditcard.csv` file to comply with GitHub's 100 MB upload limit.

---

## 🔮 Future Improvements
1. **Gradient Boosting Models:** Benchmark **XGBoost**, **LightGBM**, and **CatBoost** with Bayesian hyperparameter optimization.
2. **Dynamic Decision Threshold Tuning:** Optimize the classification threshold based on a formal financial cost matrix ($ Cost_{FN} vs Cost_{FP} $).
3. **Explainable AI (XAI):** Integrate **SHAP** (SHapley Additive exPlanations) and **LIME** directly into the Streamlit app for real-time model interpretability.
4. **Streaming Architecture:** Implement Kafka and Apache Flink to ingest and score credit card transactions in under 20 milliseconds.
5. **Continuous Model Monitoring:** Track concept drift and covariate shift in spending behaviors over time using Evidently AI.
