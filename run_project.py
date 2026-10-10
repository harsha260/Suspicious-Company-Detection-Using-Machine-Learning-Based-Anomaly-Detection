"""
Master Execution Script: Suspicious Company Detection Using Machine Learning
Academic Project 13: Anomaly Detection, Classification & Hybrid Model
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
)

# Set style for publication-quality charts
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_data(filepath="audit_data/audit_risk.csv"):
    print("=" * 60)
    print("1. LOADING AND PREPROCESSING DATASET")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print(f"Raw dataset shape: {df.shape}")

    # Handle missing values
    if df["Money_Value"].isnull().sum() > 0:
        median_mv = df["Money_Value"].median()
        df["Money_Value"] = df["Money_Value"].fillna(median_mv)
        print(f"Imputed missing Money_Value with median: {median_mv:.4f}")

    # Encode categorical LOCATION_ID
    le = LabelEncoder()
    df["LOCATION_ID"] = le.fit_transform(df["LOCATION_ID"].astype(str))

    # Target variable
    y = df["Risk"].values

    # Exclude direct formula columns to prevent target leakage
    # Audit_Risk = Inherent_Risk * CONTROL_RISK * Detection_Risk
    # Detection_Risk is constant 0.5
    leakage_cols = ["Risk", "Audit_Risk", "Inherent_Risk", "CONTROL_RISK", "Detection_Risk"]
    feature_cols = [c for c in df.columns if c not in leakage_cols]

    X = df[feature_cols].copy()
    print(f"Features selected ({len(feature_cols)}): {feature_cols}")
    print(f"Target distribution: Normal (0) = {(y == 0).sum()}, Suspicious (1) = {(y == 1).sum()}")

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} records | Test set: {X_test.shape[0]} records")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler, le


def normalize_scores(scores):
    """Normalize anomaly scores to [0, 1] range (higher = more anomalous)."""
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max == s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def train_unsupervised_models(X_train_scaled, X_test_scaled, contamination=0.39):
    print("\n" + "=" * 60)
    print("2. TRAINING 3 UNSUPERVISED ANOMALY DETECTION MODELS")
    print("=" * 60)

    # 1. Isolation Forest
    print("-> Training Isolation Forest...")
    iforest = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    iforest.fit(X_train_scaled)
    raw_train_if = -iforest.score_samples(X_train_scaled)
    raw_test_if = -iforest.score_samples(X_test_scaled)

    # 2. Local Outlier Factor (LOF) with novelty=True
    print("-> Training Local Outlier Factor (LOF)...")
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination)
    lof.fit(X_train_scaled)
    raw_train_lof = -lof.score_samples(X_train_scaled)
    raw_test_lof = -lof.score_samples(X_test_scaled)

    # 3. One-Class SVM
    print("-> Training One-Class SVM...")
    ocsvm = OneClassSVM(kernel='rbf', nu=0.30, gamma='scale')
    ocsvm.fit(X_train_scaled)
    raw_train_ocsvm = -ocsvm.decision_function(X_train_scaled)
    raw_test_ocsvm = -ocsvm.decision_function(X_test_scaled)

    raw_train_anomaly = np.column_stack([raw_train_if, raw_train_lof, raw_train_ocsvm])
    raw_test_anomaly = np.column_stack([raw_test_if, raw_test_lof, raw_test_ocsvm])

    # Fit MinMaxScaler on training anomaly scores so single samples can be properly normalized
    anomaly_scaler = MinMaxScaler()
    train_anomaly_features = anomaly_scaler.fit_transform(raw_train_anomaly)
    test_anomaly_features = np.clip(anomaly_scaler.transform(raw_test_anomaly), 0.0, 1.0)

    unsupervised_models = {
        "Isolation Forest": iforest,
        "Local Outlier Factor (LOF)": lof,
        "One-Class SVM": ocsvm
    }

    test_scores = {
        "Isolation Forest": test_anomaly_features[:, 0],
        "Local Outlier Factor (LOF)": test_anomaly_features[:, 1],
        "One-Class SVM": test_anomaly_features[:, 2]
    }

    return unsupervised_models, test_scores, train_anomaly_features, test_anomaly_features, anomaly_scaler


def train_supervised_models(X_train_scaled, X_test_scaled, y_train):
    print("\n" + "=" * 60)
    print("3. TRAINING 3 SUPERVISED CLASSIFICATION MODELS")
    print("=" * 60)

    # 1. Logistic Regression (Baseline)
    print("-> Training Logistic Regression (Baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    # 2. Random Forest
    print("-> Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]

    # 3. XGBoost
    print("-> Training XGBoost...")
    xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
    xgb.fit(X_train_scaled, y_train)
    xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]

    supervised_models = {
        "Logistic Regression (Baseline)": lr,
        "Random Forest": rf,
        "XGBoost": xgb
    }

    test_probs = {
        "Logistic Regression (Baseline)": lr_probs,
        "Random Forest": rf_probs,
        "XGBoost": xgb_probs
    }

    return supervised_models, test_probs


def train_hybrid_model(X_train_scaled, X_test_scaled, train_anom_feats, test_anom_feats, y_train):
    print("\n" + "=" * 60)
    print("4. TRAINING PROPOSED HYBRID MODEL (FEATURE AUGMENTATION)")
    print("=" * 60)

    # Augment features with 3 unsupervised anomaly scores
    X_train_aug = np.hstack([X_train_scaled, train_anom_feats])
    X_test_aug = np.hstack([X_test_scaled, test_anom_feats])
    print(f"Augmented feature space: {X_train_scaled.shape[1]} original + 3 anomaly scores = {X_train_aug.shape[1]} features")

    # Train Hybrid XGBoost model on augmented feature space
    hybrid_xgb = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42
    )
    hybrid_xgb.fit(X_train_aug, y_train)
    hybrid_probs = hybrid_xgb.predict_proba(X_test_aug)[:, 1]

    return hybrid_xgb, hybrid_probs, X_test_aug


def evaluate_models(all_predictions, y_test):
    print("\n" + "=" * 60)
    print("5. COMPARATIVE EVALUATION & PRECISION-RECALL BENCHMARK")
    print("=" * 60)

    results = []
    for model_name, score_or_prob in all_predictions.items():
        # Binary prediction using optimal threshold (0.5 for prob, median for anomaly scores)
        threshold = 0.5
        y_pred = (score_or_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, score_or_prob)
        pr_auc = average_precision_score(y_test, score_or_prob)

        # F2 score (recall weighted for audit risk)
        beta = 2.0
        f2 = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec + 1e-9)

        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "F2-Score (Audit Risk)": f2,
            "PR-AUC (Avg Precision)": pr_auc,
            "ROC-AUC": roc_auc
        })

    results_df = pd.DataFrame(results).sort_values(by="PR-AUC (Avg Precision)", ascending=False)
    results_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\nBenchmark Comparison Summary:")
    print(results_df.to_string(index=False))
    print(f"\n[Saved] Benchmark table written to: {results_path}")
    return results_df


def generate_visualizations(all_predictions, y_test, rf_model, feature_names):
    print("\n" + "=" * 60)
    print("6. GENERATING PUBLICATION-READY VISUALIZATIONS")
    print("=" * 60)

    # 1. Comparative Precision-Recall Curves
    plt.figure(figsize=(9, 6))
    colors = {
        "Logistic Regression (Baseline)": "#4e79a7",
        "Random Forest": "#59a14f",
        "XGBoost": "#f28e2b",
        "Isolation Forest": "#e15759",
        "Local Outlier Factor (LOF)": "#b07aa1",
        "One-Class SVM": "#9c755f",
        "Proposed Hybrid Model": "#2b5c8f"
    }

    for name, scores in all_predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        lw = 2.5 if "Hybrid" in name else 1.8
        ls = "-" if ("Supervised" in name or "Hybrid" in name or name in ["Random Forest", "XGBoost", "Logistic Regression (Baseline)"]) else "--"
        plt.plot(recall, precision, lw=lw, linestyle=ls, color=colors.get(name, None),
                 label=f"{name} (PR-AUC = {ap:.3f})")

    plt.xlabel("Recall (Abnormal Company Detection Rate)", fontsize=12, fontweight='bold')
    plt.ylabel("Precision (True Suspicious / Flagged)", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Analysis: Supervised vs Unsupervised vs Hybrid", fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    pr_path = os.path.join(OUTPUT_DIR, "pr_curves_comparison.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"[Saved] PR curves chart written to: {pr_path}")

    # 2. Comparative ROC Curves
    plt.figure(figsize=(9, 6))
    for name, scores in all_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        lw = 2.5 if "Hybrid" in name else 1.8
        ls = "-" if ("Supervised" in name or "Hybrid" in name or name in ["Random Forest", "XGBoost", "Logistic Regression (Baseline)"]) else "--"
        plt.plot(fpr, tpr, lw=lw, linestyle=ls, color=colors.get(name, None),
                 label=f"{name} (ROC-AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle=":")
    plt.xlabel("False Positive Rate (Clean Companies Flagged)", fontsize=12, fontweight='bold')
    plt.ylabel("True Positive Rate (Suspicious Companies Detected)", fontsize=12, fontweight='bold')
    plt.title("ROC Analysis: Detection Discrimination Power", fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves_comparison.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Saved] ROC curves chart written to: {roc_path}")

    # 3. Feature Importance (Random Forest)
    plt.figure(figsize=(9, 6))
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    sns.barplot(x=top_scores, y=top_features, hue=top_features, palette="mako", legend=False)
    plt.title("Top 10 Influential Financial Risk Indicators (Random Forest)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Feature Importance Score", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"[Saved] Feature importance chart written to: {fi_path}")


def generate_high_risk_ranking(X_test, y_test, hybrid_probs, test_scores):
    print("\n" + "=" * 60)
    print("7. HIGH-RISK AUDIT TRIAGE RANKING (TOP-N)")
    print("=" * 60)

    ranking_df = X_test.copy()
    ranking_df["Actual_Risk"] = y_test
    ranking_df["Suspiciousness_Score"] = np.round(hybrid_probs, 4)
    ranking_df["Isolation_Forest_Score"] = np.round(test_scores["Isolation Forest"], 4)
    ranking_df["LOF_Score"] = np.round(test_scores["Local Outlier Factor (LOF)"], 4)
    ranking_df["OneClassSVM_Score"] = np.round(test_scores["One-Class SVM"], 4)

    # Risk Tier assignment
    def assign_tier(score):
        if score >= 0.80:
            return "Critical Priority"
        elif score >= 0.60:
            return "High Priority"
        elif score >= 0.40:
            return "Medium Priority"
        else:
            return "Low / Normal"

    ranking_df["Audit_Priority"] = ranking_df["Suspiciousness_Score"].apply(assign_tier)
    ranked = ranking_df.sort_values(by="Suspiciousness_Score", ascending=False).reset_index(drop=True)

    ranking_path = os.path.join(OUTPUT_DIR, "top_high_risk_companies.csv")
    ranked.to_csv(ranking_path, index=True)

    print("\nTop 10 Ranked Suspicious Companies for Audit Priority:")
    preview_cols = ["Suspiciousness_Score", "Actual_Risk", "Audit_Priority", "Score", "TOTAL", "Money_Value", "Sector_score"]
    print(ranked[preview_cols].head(10).to_string())
    print(f"\n[Saved] Triage ranking exported to: {ranking_path}")
    return ranked


def save_trained_system(scaler, le, feature_cols, supervised_models, unsupervised_models, hybrid_model, anomaly_scaler):
    print("\n" + "=" * 60)
    print("8. SAVING TRAINED ML ARTIFACTS FOR STREAMLIT APP")
    print("=" * 60)

    bundle = {
        "scaler": scaler,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "supervised_models": supervised_models,
        "unsupervised_models": unsupervised_models,
        "hybrid_model": hybrid_model,
        "anomaly_scaler": anomaly_scaler
    }

    bundle_path = os.path.join(OUTPUT_DIR, "trained_system.joblib")
    joblib.dump(bundle, bundle_path)
    print(f"[Saved] Model bundle saved for live inference: {bundle_path}")


def main():
    # 1. Data Prep
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, feat_cols, scaler, le = load_and_preprocess_data()

    # 2. Unsupervised Anomaly Detection
    unsupervised_models, test_scores, train_anom, test_anom, anomaly_scaler = train_unsupervised_models(X_train_s, X_test_s)

    # 3. Supervised Classification
    supervised_models, test_probs = train_supervised_models(X_train_s, X_test_s, y_train)

    # 4. Proposed Hybrid Model
    hybrid_model, hybrid_probs, _ = train_hybrid_model(X_train_s, X_test_s, train_anom, test_anom, y_train)

    # Collect all predictions for comparison
    all_predictions = {
        **supervised_models,
        **test_probs,
        **test_scores,
        "Proposed Hybrid Model": hybrid_probs
    }
    # Keep only test probability / score vectors for evaluation
    eval_dict = {
        "Logistic Regression (Baseline)": test_probs["Logistic Regression (Baseline)"],
        "Random Forest": test_probs["Random Forest"],
        "XGBoost": test_probs["XGBoost"],
        "Isolation Forest": test_scores["Isolation Forest"],
        "Local Outlier Factor (LOF)": test_scores["Local Outlier Factor (LOF)"],
        "One-Class SVM": test_scores["One-Class SVM"],
        "Proposed Hybrid Model": hybrid_probs
    }

    # 5. Evaluate
    evaluate_models(eval_dict, y_test)

    # 6. Visualizations
    generate_visualizations(eval_dict, y_test, supervised_models["Random Forest"], feat_cols)

    # 7. High-Risk Ranking
    generate_high_risk_ranking(X_test, y_test, hybrid_probs, test_scores)

    # 8. Save Model Bundle
    save_trained_system(scaler, le, feat_cols, supervised_models, unsupervised_models, hybrid_model, anomaly_scaler)

    print("\n" + "=" * 60)
    print("SUCCESS: PIPELINE EXECUTION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
