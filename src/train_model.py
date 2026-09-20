"""
Credit Card Fraud Detection System
Production Machine Learning Training Pipeline
Internship Project for Codec Technologies
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"

def main():
    print("=" * 70)
    print("[*] CREDIT CARD FRAUD DETECTION PIPELINE - CODEC TECHNOLOGIES")
    print("=" * 70)

    # 1. Paths configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "creditcard.csv")
    models_dir = os.path.join(base_dir, "models")
    figures_dir = os.path.join(base_dir, "figures")
    data_dir = os.path.join(base_dir, "data")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 2. Load Dataset
    print("\n[STEP 1] Loading Dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please download it first.")
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names and types:")
    print(df.dtypes.value_counts())

    # 3. Data Cleaning & Sanity Checks
    print("\n[STEP 2] Data Cleaning & Sanity Verification...")
    missing_values = df.isnull().sum().sum()
    print(f"Total Missing Values: {missing_values}")

    duplicate_rows = df.duplicated().sum()
    print(f"Total Duplicate Rows: {duplicate_rows:,}")
    if duplicate_rows > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicates removed. New dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Class distribution
    legit_count = (df['Class'] == 0).sum()
    fraud_count = (df['Class'] == 1).sum()
    total_count = len(df)
    fraud_percent = (fraud_count / total_count) * 100
    print(f"\nClass Distribution:")
    print(f"  - Legitimate (Class 0): {legit_count:,} ({100 - fraud_percent:.3f}%)")
    print(f"  - Fraudulent (Class 1): {fraud_count:,} ({fraud_percent:.3f}%)")

    # Statistical Summary for Amount
    print("\nSummary Statistics for Transaction Amount:")
    print(df.groupby('Class')['Amount'].describe())

    # 4. Save Key Visualizations for Reports/README
    print("\n[STEP 3] Generating Exploratory Data Analysis Plots...")
    
    # 4.1 Class Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ["#2b5c8f", "#d9534f"]
    sns.countplot(x='Class', data=df, hue='Class', palette=palette, legend=False, ax=ax)
    ax.set_title("Transaction Class Distribution (Extreme Imbalance)", fontsize=14, weight='bold', pad=12)
    ax.set_xlabel("Transaction Class (0 = Legitimate, 1 = Fraudulent)", fontsize=11)
    ax.set_ylabel("Transaction Count", fontsize=11)
    ax.set_yscale("log")
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height):,}\n({height/total_count*100:.2f}%)",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, weight='bold',
                        xytext=(0, 4), textcoords='offset points')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "01_class_distribution.png"), dpi=300)
    plt.close()

    # 4.2 Amount Distribution by Class (Log scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, ax=ax1, color="#2b5c8f", kde=True, log_scale=(True, False))
    ax1.set_title("Legitimate Transactions - Amount (Log Scale)", fontsize=12, weight='bold')
    ax1.set_xlabel("Amount ($)")
    ax1.set_ylabel("Frequency")

    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, ax=ax2, color="#d9534f", kde=True, log_scale=(True, False))
    ax2.set_title("Fraudulent Transactions - Amount (Log Scale)", fontsize=12, weight='bold')
    ax2.set_xlabel("Amount ($)")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_amount_distribution.png"), dpi=300)
    plt.close()

    # 4.3 Time Distribution by Class
    fig, ax = plt.subplots(figsize=(10, 4.5))
    df_plot_time = df.copy()
    df_plot_time['Hour'] = (df_plot_time['Time'] / 3600) % 24
    sns.kdeplot(data=df_plot_time[df_plot_time['Class'] == 0]['Hour'], label="Legitimate (0)", color="#2b5c8f", fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(data=df_plot_time[df_plot_time['Class'] == 1]['Hour'], label="Fraudulent (1)", color="#d9534f", fill=True, alpha=0.3, ax=ax)
    ax.set_title("Transaction Time Distribution (Hour of the Day)", fontsize=14, weight='bold')
    ax.set_xlabel("Hour of Day (0 - 24)")
    ax.set_ylabel("Density")
    ax.legend(title="Class")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "03_time_distribution.png"), dpi=300)
    plt.close()

    # 4.4 Correlation Heatmap (Focus on Top Features correlated with Class)
    corr = df.corr()
    corr_target = corr['Class'].drop('Class').sort_values()
    top_corr_features = list(corr_target.head(6).index) + list(corr_target.tail(6).index) + ['Class']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[top_corr_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title("Correlation Heatmap: Top Features Correlated with Fraud Class", fontsize=13, weight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_correlation_heatmap.png"), dpi=300)
    plt.close()

    # 4.5 Fraud vs Transaction Amount Boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x='Class', y='Amount', data=df, palette=palette, hue='Class', legend=False, showfliers=False, ax=ax)
    ax.set_title("Transaction Amount by Class (Excluding Outliers)", fontsize=13, weight='bold')
    ax.set_xlabel("Class (0 = Legitimate, 1 = Fraudulent)")
    ax.set_ylabel("Amount ($)")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "05_amount_boxplot.png"), dpi=300)
    plt.close()
    print("EDA Visualizations saved to figures/")

    # 5. Save Sample Transactions for Streamlit Quick Testing
    print("\n[STEP 4] Exporting Sample Transactions for Streamlit Demo...")
    sample_legit = df[df['Class'] == 0].sample(5, random_state=42)
    sample_fraud = df[df['Class'] == 1].sample(5, random_state=42)
    sample_data = {
        "legitimate": sample_legit.drop(columns=['Class']).to_dict(orient='records'),
        "fraudulent": sample_fraud.drop(columns=['Class']).to_dict(orient='records')
    }
    with open(os.path.join(data_dir, "sample_transactions.json"), "w") as f:
        json.dump(sample_data, f, indent=2)
    print("Sample transactions saved to data/sample_transactions.json")

    # 6. Train/Test Split (BEFORE SCALING AND RESAMPLING TO PREVENT LEAKAGE)
    print("\n[STEP 5] Performing Stratified Train/Test Split...")
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training set: {X_train.shape[0]:,} samples | Legitimate: {(y_train == 0).sum():,}, Fraud: {(y_train == 1).sum():,}")
    print(f"Testing set:  {X_test.shape[0]:,} samples | Legitimate: {(y_test == 0).sum():,}, Fraud: {(y_test == 1).sum():,}")

    # 7. Feature Scaling (Fit ONLY on Training Set)
    print("\n[STEP 6] Feature Scaling (Fitting RobustScaler on Training Data Only)...")
    scaler_amount = RobustScaler()
    scaler_time = RobustScaler()

    # Create copy to avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train['scaled_amount'] = scaler_amount.fit_transform(X_train[['Amount']])
    X_train['scaled_time'] = scaler_time.fit_transform(X_train[['Time']])
    X_train = X_train.drop(columns=['Amount', 'Time'])

    X_test['scaled_amount'] = scaler_amount.transform(X_test[['Amount']])
    X_test['scaled_time'] = scaler_time.transform(X_test[['Time']])
    X_test = X_test.drop(columns=['Amount', 'Time'])

    feature_order = list(X_train.columns)

    # 8. Apply SMOTE to Training Set ONLY
    print("\n[STEP 7] Applying SMOTE (Synthetic Minority Over-sampling Technique) to Training Set...")
    print(f"Pre-SMOTE Training Class Ratio: 0: {(y_train == 0).sum():,} | 1: {(y_train == 1).sum():,}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Post-SMOTE Training Class Ratio: 0: {(y_train_resampled == 0).sum():,} | 1: {(y_train_resampled == 1).sum():,}")
    print("NOTE: Test set remains completely un-resampled to preserve authentic evaluation.")

    # 9. Model Training & Evaluation
    print("\n[STEP 8] Model Training & Comparative Evaluation...")

    # Dictionary to store performance results
    results = {}

    # --- Model 1: Logistic Regression ---
    print("\n--> Training Model 1: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_resampled, y_train_resampled)

    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

    results["Logistic Regression"] = {
        "model": lr_model,
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr),
        "roc_auc": roc_auc_score(y_test, y_prob_lr),
        "y_pred": y_pred_lr,
        "y_prob": y_prob_lr
    }

    # --- Model 2: Random Forest Classifier ---
    print("\n--> Training Model 2: Random Forest Classifier...")
    # Using n_estimators=100, max_depth=15 for efficient robust training
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_resampled, y_train_resampled)

    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    results["Random Forest"] = {
        "model": rf_model,
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf),
        "roc_auc": roc_auc_score(y_test, y_prob_rf),
        "y_pred": y_pred_rf,
        "y_prob": y_prob_rf
    }

    # --- Model 3: Isolation Forest (Unsupervised Anomaly Detection) ---
    print("\n--> Training Model 3: Isolation Forest (Anomaly Detection)...")
    # Isolation Forest is fitted on original training features (anomaly detection)
    iso_model = IsolationForest(contamination=0.0017, random_state=42, n_jobs=-1)
    iso_model.fit(X_train)  # Unsupervised, no labels

    # Isolation Forest outputs 1 for inliers (normal) and -1 for outliers (fraud)
    y_pred_iso_raw = iso_model.predict(X_test)
    y_pred_iso = np.where(y_pred_iso_raw == -1, 1, 0)
    iso_scores = -iso_model.score_samples(X_test)  # Higher anomaly score = more likely fraud

    results["Isolation Forest"] = {
        "model": iso_model,
        "accuracy": accuracy_score(y_test, y_pred_iso),
        "precision": precision_score(y_test, y_pred_iso, zero_division=0),
        "recall": recall_score(y_test, y_pred_iso),
        "f1": f1_score(y_test, y_pred_iso, zero_division=0),
        "roc_auc": roc_auc_score(y_test, iso_scores),
        "y_pred": y_pred_iso,
        "y_prob": iso_scores
    }

    # 10. Performance Comparison Table
    print("\n" + "=" * 75)
    print("[+] MODEL PERFORMANCE COMPARISON (TEST SET EVALUATION)")
    print("=" * 75)
    print(f"{'Model':<22} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<8} | {'F1 Score':<9} | {'ROC-AUC':<8}")
    print("-" * 75)
    for model_name, res in results.items():
        print(f"{model_name:<22} | {res['accuracy']:.5f}   | {res['precision']:.5f}    | {res['recall']:.5f}  | {res['f1']:.5f}   | {res['roc_auc']:.5f}")
    print("=" * 75)

    # 11. Confusion Matrices
    print("\n[STEP 9] Generating Evaluation Curves & Confusion Matrices...")
    for model_name in ["Logistic Regression", "Random Forest"]:
        cm = confusion_matrix(y_test, results[model_name]["y_pred"])
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{model_name} Confusion Matrix:")
        print(f"  True Negatives (Legitimate correctly identified):  {tn:,}")
        print(f"  False Positives (Legitimate flagged as Fraud):      {fp:,}")
        print(f"  False Negatives (Fraud missed by model):           {fn:,}")
        print(f"  True Positives (Fraud correctly detected):         {tp:,}")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred Legit (0)", "Pred Fraud (1)"],
                    yticklabels=["Actual Legit (0)", "Actual Fraud (1)"])
        ax.set_title(f"{model_name} Confusion Matrix", fontsize=13, weight='bold')
        plt.tight_layout()
        filename = f"06_{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        fig.savefig(os.path.join(figures_dir, filename), dpi=300)
        plt.close()

    # 12. ROC & Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ROC Curves
    for name, color in zip(["Logistic Regression", "Random Forest"], ["#2b5c8f", "#28a745"]):
        fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob"])
        auc_val = results[name]["roc_auc"]
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=color, linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, label="Random Guess (AUC = 0.50)")
    ax1.set_title("ROC Curves Comparison", fontsize=13, weight='bold')
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.legend(loc="lower right")

    # Precision-Recall Curves
    for name, color in zip(["Logistic Regression", "Random Forest"], ["#2b5c8f", "#28a745"]):
        precision, recall, _ = precision_recall_curve(y_test, results[name]["y_prob"])
        ap = average_precision_score(y_test, results[name]["y_prob"])
        ax2.plot(recall, precision, label=f"{name} (PR-AUC = {ap:.4f})", color=color, linewidth=2)
    ax2.set_title("Precision-Recall Curves (Vital for Imbalance)", fontsize=13, weight='bold')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "07_roc_pr_curves.png"), dpi=300)
    plt.close()

    # 13. Feature Importance (Random Forest)
    print("\n[STEP 10] Extracting Feature Importance from Random Forest...")
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_order,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    top_15_features = feat_df.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_15_features, palette="viridis", ax=ax)
    ax.set_title("Top 15 Most Important Features in Fraud Detection (Random Forest)", fontsize=13, weight='bold')
    ax.set_xlabel("Gini Feature Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "08_feature_importance.png"), dpi=300)
    plt.close()
    print("Feature importance plot saved to figures/08_feature_importance.png")

    # 14. Save the Final Selected Model and Pipeline
    print("\n[STEP 11] Saving Production Model Pipeline...")
    # We select Random Forest as the final model due to superior Precision, F1-Score, and PR-AUC
    model_artifact = {
        "model_name": "Random Forest Classifier",
        "model": rf_model,
        "scaler_amount": scaler_amount,
        "scaler_time": scaler_time,
        "feature_order": feature_order,
        "metrics": {
            "accuracy": float(results["Random Forest"]["accuracy"]),
            "precision": float(results["Random Forest"]["precision"]),
            "recall": float(results["Random Forest"]["recall"]),
            "f1": float(results["Random Forest"]["f1"]),
            "roc_auc": float(results["Random Forest"]["roc_auc"])
        }
    }
    
    model_save_path = os.path.join(models_dir, "fraud_detection_model.pkl")
    joblib.dump(model_artifact, model_save_path)
    print(f"Final model successfully saved to: {model_save_path}")

    # Also save metrics JSON for easy programmatic reference
    summary_metrics = {
        k: {
            "accuracy": float(v["accuracy"]),
            "precision": float(v["precision"]),
            "recall": float(v["recall"]),
            "f1": float(v["f1"]),
            "roc_auc": float(v["roc_auc"])
        }
        for k, v in results.items()
    }
    with open(os.path.join(models_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Metrics saved to: {os.path.join(models_dir, 'evaluation_metrics.json')}")

    print("\n" + "=" * 70)
    print("[SUCCESS] TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
