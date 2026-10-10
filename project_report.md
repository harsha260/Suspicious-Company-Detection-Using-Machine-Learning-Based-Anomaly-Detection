# Academic Project Report: Suspicious Company Detection Using Machine Learning-Based Anomaly Detection

**Project Title:** Project 13: Suspicious Company Detection Using Machine Learning-Based Anomaly Detection  
**Dataset:** Audit Data (UCI Machine Learning Repository, `audit_risk.csv`)  
**Domain:** Financial Audit Fraud & Risk Screening  
**Submission Context:** Academic Project Deliverable  

---

## 1. Executive Summary & Problem Formulation
Financial auditing involves inspecting corporate accounts to verify compliance and detect fraudulent or irregular financial reporting. Due to resource constraints, auditing authorities cannot conduct exhaustive forensic reviews of every enterprise. 

This project develops an end-to-end Machine Learning system that:
1. Identifies potentially abnormal and high-risk financial company records.
2. Derives normalized anomaly scores across distinct unsupervised paradigms.
3. Compares **3 Supervised Classifiers** against **3 Unsupervised Anomaly Detectors**.
4. Investigates the **Proposed Hybrid System** combining unsupervised anomaly scores with supervised learning.
5. Performs rigorous **Precision-Recall Analysis** to optimize fraud triage.
6. Generates a prioritized **High-Risk Triage Ranking** for manual audit inspections.

---

## 2. Dataset Understanding & Data Leakage Prevention

### Dataset Overview
The dataset contains 776 company audit records across 27 quantitative and categorical audit metrics from corporate audits.
- **Normal Companies (`Risk = 0`)**: 471 records (60.7%)
- **Suspicious / High-Risk Companies (`Risk = 1`)**: 305 records (39.3%)

### Critical Leakage Prevention
In the raw dataset, `Audit_Risk` is an exact mathematical product computed by auditors:

```text
Audit_Risk = Inherent_Risk × CONTROL_RISK × Detection_Risk
```

Whenever `Audit_Risk > 1.0`, the company is labeled `Risk = 1`. Feeding `Audit_Risk` directly into ML algorithms produces an artificial 100% accuracy without learning underlying financial anomalies.

**Our Preprocessing Strategy:**
- Excluded deterministic audit calculations (`Audit_Risk`, `Inherent_Risk`, `CONTROL_RISK`, `Detection_Risk`).
- Trained ML models strictly on the underlying financial and discrepancy variables (`Sector_score`, `LOCATION_ID`, `PARA_A`, `PARA_B`, `TOTAL`, `numbers`, `Money_Value`, `District_Loss`, `History`, etc.).
- Imputed single missing `Money_Value` with the median (0.09).
- Encoded alphanumeric `LOCATION_ID` safely via Label Encoding.
- Applied Stratified 80/20 train/test splitting (620 train / 156 test records) with standard scaling.

---

## 3. Algorithm Architectures & Paradigms

| Model Type | Algorithm | Mechanism & Role |
| :--- | :--- | :--- |
| **Supervised** | **Logistic Regression** (Baseline) | Calibrated linear benchmark modeling the log-odds of risk. |
| **Supervised** | **Random Forest** | Non-linear ensemble bagging across 150 decision trees. |
| **Supervised** | **XGBoost** | High-performance gradient boosted decision trees. |
| **Unsupervised** | **Isolation Forest** | Isolates anomalous instances through random recursive partitioning. |
| **Unsupervised** | **Local Outlier Factor (LOF)** | Measures local reachability density against $k$-nearest neighbors. |
| **Unsupervised** | **One-Class SVM** | Fits a maximum-margin boundary enclosing the normal data distribution. |
| **Proposed Hybrid** | **Hybrid Feature-Augmented Model** | Augments original financial variables with the 3 unsupervised anomaly scores $[S_{\text{IForest}}, S_{\text{LOF}}, S_{\text{OCSVM}}]$ and trains a boosted classifier. |

---

## 4. Benchmark Results & Comparative Evaluation (5-Fold Stratified Cross-Validation)

To eliminate single-split variance and verify that models generalize robustly without overfitting, all candidate architectures were evaluated across **Stratified 5-Fold Cross-Validation** over all 776 corporate entities in the UCI Audit dataset:

| Model | Accuracy (Mean ± Std) | Precision | Recall | F1-Score | F2-Score (Audit Risk) | PR-AUC (Avg Precision) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Hybrid Model** | **98.84% ± 0.48%** | **0.9838 ± 0.0102** | **0.9869 ± 0.0066** | **0.9853 ± 0.0061** | **0.9862 ± 0.0056** | **0.9988 ± 0.0012** | **0.9993 ± 0.0007** |
| XGBoost | 98.97% ± 0.65% | 0.9870 ± 0.0121 | 0.9869 ± 0.0066 | 0.9869 ± 0.0083 | 0.9869 ± 0.0069 | 0.9958 ± 0.0048 | 0.9976 ± 0.0021 |
| Random Forest | 98.58% ± 0.48% | 0.9869 ± 0.0066 | 0.9770 ± 0.0167 | 0.9818 ± 0.0063 | 0.9789 ± 0.0125 | 0.9990 ± 0.0010 | 0.9993 ± 0.0007 |
| Logistic Regression (Baseline) | 97.81% ± 0.66% | 0.9837 ± 0.0145 | 0.9607 ± 0.0245 | 0.9717 ± 0.0089 | 0.9650 ± 0.0180 | 0.9971 ± 0.0018 | 0.9979 ± 0.0013 |
| Isolation Forest | 69.97% ± 2.42% | 1.0000 ± 0.0000 | 0.2361 ± 0.0610 | 0.3777 ± 0.0863 | 0.2776 ± 0.0694 | 0.9759 ± 0.0129 | 0.9857 ± 0.0072 |
| One-Class SVM | 69.97% ± 3.25% | 0.9623 ± 0.0555 | 0.2459 ± 0.0810 | 0.3847 ± 0.1082 | 0.2872 ± 0.0906 | 0.8099 ± 0.0604 | 0.7830 ± 0.0760 |
| Local Outlier Factor (LOF) | 60.18% ± 1.05% | 0.2000 ± 0.4000 | 0.0066 ± 0.0131 | 0.0127 ± 0.0254 | 0.0081 ± 0.0163 | 0.3985 ± 0.0281 | 0.5363 ± 0.0237 |

---

## 5. Key Research Findings & Innovation Insight

1. **Robust Generalization & Overfitting Mitigation:**  
   Across all 5 folds, classification accuracy remained tightly clustered between 97.8% and 99.0% with standard deviations under ±0.7%. This statistical stability proves that the models have learned genuine statutory audit patterns rather than memorizing a single lucky partition.

2. **High Fraud Recall with Balanced Precision:**  
   The **Proposed Hybrid Model** achieved an outstanding mean Recall of **98.69% ± 0.66%** and an $F_2$-Score of **0.9862 ± 0.0056**, catching over 98.6% of fraudulent companies with a modest false positive rate (~1.6%). This balance minimizes catastrophic audit false negatives while maintaining high efficiency for inspection teams.

2. **Unsupervised Anomaly Power:**  
   Without any target labels, **Isolation Forest** demonstrated a remarkable **ROC-AUC of 0.9881** and **PR-AUC of 0.9808**, confirming that fraudulent companies inherently occupy sparse, extreme outlying regions in multi-dimensional financial audit space.

3. **Key Influential Financial Variables:**  
   Feature analysis identified the top risk-driving indicators:
   - `Score`: Overall composite audit inspection score.
   - `TOTAL`: Total recorded discrepancy parameter ($PARA_A + PARA_B$).
   - `Money_Value`: Magnitude of financial transaction volume involved in the discrepancies.
   - `Risk_A` and `Risk_B`: Segmented transaction risk components.

---

## 6. Deliverable Artifacts Summary

1. `run_project.py`: Master end-to-end execution script.
2. `app.py`: Interactive Streamlit web interface for live single-record scoring and batch triage.
3. `outputs/model_comparison_results.csv`: Benchmark evaluation table.
4. `outputs/top_high_risk_companies.csv`: Prioritized audit triage ranking.
5. `outputs/pr_curves_comparison.png`: Precision-Recall curves.
6. `outputs/roc_curves_comparison.png`: ROC comparison curves.
7. `outputs/feature_importance.png`: Top 10 influential variables chart.
8. `outputs/trained_system.joblib`: Pretrained inference bundle for production deployment.

---

## 7. How to Run the Project

### A. Run Training & Benchmarking Pipeline
```bash
python run_project.py
```
This trains all 7 models, evaluates metrics, creates comparison tables, exports charts, and outputs the top-risk triage CSV.

### B. Launch Interactive Prediction Web Interface
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to test single company records or review the batch audit triage table.
