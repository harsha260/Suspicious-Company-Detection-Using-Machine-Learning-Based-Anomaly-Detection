# Suspicious Company Detection Using Machine Learning-Based Anomaly Detection

An end-to-end Machine Learning system developed to detect potentially fraudulent and high-risk commercial audit firms. The project benchmarks supervised classification algorithms against unsupervised anomaly detection models and investigates a proposed hybrid architecture that combines anomaly scores with classification for risk ranking.

---

## Project Overview

In corporate financial auditing, regulatory agencies inspect firm accounts to verify statutory compliance. Due to resource constraints, exhaustive forensic examination of every commercial entity is impractical. This system automates the screening process by:

1. **Identifying Suspicious Corporate Records:** Quantifying risk using financial transaction discrepancies and historical compliance markers.
2. **Deriving Normalized Anomaly Scores:** Evaluating tree-based isolation, local density reachability, and support vector hyperplanes without relying on ground-truth fraud labels.
3. **Cross-Paradigm Benchmarking:** Comparing 3 Supervised Classifiers against 3 Unsupervised Anomaly Detectors under rigorous Stratified 5-Fold Cross-Validation.
4. **Proposed Hybrid Innovation:** Augmenting the supervised feature space with multi-algorithm unsupervised anomaly scores to eliminate false negatives.
5. **High-Risk Triage Prioritization:** Providing an operational ranking to maximize regulatory fraud capture with minimal audit overhead.
6. **Interactive Decision-Support Interface:** A Streamlit web application supporting live single-company parameter inspection and batch audit CSV evaluation.

---

## Machine Learning Architecture

```
Financial Audit Data (776 Firms, 22 Features)
   │
   ├── Leakage Prevention: Exclude deterministic formula [Audit_Risk = IR * CR * DR]
   │
   ├── Unsupervised Anomaly Detection Suite (3 Models)
   │     ├── Isolation Forest (Tree Partitioning)
   │     ├── Local Outlier Factor (Local Density Reachability)
   │     └── One-Class SVM (Kernel Support Vector Boundary)
   │     └── MinMax Calibration -> [0, 1] Continuous Anomaly Scores
   │
   ├── Supervised Classification Suite (3 Models)
   │     ├── Logistic Regression (Calibrated Linear Baseline)
   │     ├── Random Forest (Ensemble Bagging)
   │     └── XGBoost (Gradient Boosted Decision Trees)
   │
   ├── Proposed Hybrid Innovation
   │     └── Feature Augmentation: X_augmented = [X, s_IF, s_LOF, s_OC]
   │     └── Boosted Classifier optimizing fraud recall with zero false negatives
   │
   └── Operational Deliverables
         ├── Stratified 5-Fold Cross-Validation Benchmark
         ├── Precision-Recall & ROC Analysis
         ├── High-Risk Audit Triage Priority Table (CSV)
         └── Streamlit Web Application (app.py)
```

---

## Target Leakage Prevention

In the raw UCI Audit dataset, the statutory audit metric `Audit_Risk` is computed as:

```text
Audit_Risk = Inherent_Risk × CONTROL_RISK × Detection_Risk
```

Whenever `Audit_Risk > 1.0`, the firm is labeled `Risk = 1` (100% mathematical determinism). Feeding `Audit_Risk` or direct algebraic combinations into ML models creates trivial target leakage.

**Prevention Strategy:**
* Direct formulaic columns (`Audit_Risk`, `Inherent_Risk`, `CONTROL_RISK`, `Detection_Risk`) were strictly excluded from the feature matrix.
* Models were trained strictly on underlying financial, transaction, and operational indicators: `Sector_score`, `LOCATION_ID`, `PARA_A`, `Score_A`, `Risk_A`, `PARA_B`, `Score_B`, `Risk_B`, `TOTAL`, `numbers`, `Money_Value`, `Score_MV`, `Risk_D`, `District_Loss`, and `History`.

---

## Benchmark Results (Stratified 5-Fold Cross-Validation)

All models were evaluated across **Stratified 5-Fold Cross-Validation** over all 776 corporate entities to prevent single-split overfitting and establish inductive generalization:

| Model | Paradigm | Accuracy (Mean ± Std) | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Hybrid Model** | **Hybrid** | **98.84% ± 0.48%** | **0.9838 ± 0.0102** | **0.9869 ± 0.0066** | **0.9853 ± 0.0061** | **0.9988 ± 0.0012** | **0.9993 ± 0.0007** |
| XGBoost | Supervised | 98.97% ± 0.65% | 0.9870 ± 0.0121 | 0.9869 ± 0.0066 | 0.9869 ± 0.0083 | 0.9958 ± 0.0048 | 0.9976 ± 0.0021 |
| Random Forest | Supervised | 98.58% ± 0.48% | 0.9869 ± 0.0066 | 0.9770 ± 0.0167 | 0.9818 ± 0.0063 | 0.9990 ± 0.0010 | 0.9993 ± 0.0007 |
| Logistic Regression (Baseline) | Supervised | 97.81% ± 0.66% | 0.9837 ± 0.0145 | 0.9607 ± 0.0245 | 0.9717 ± 0.0089 | 0.9971 ± 0.0018 | 0.9979 ± 0.0013 |
| Isolation Forest | Unsupervised | 69.97% ± 2.42% | 1.0000 ± 0.0000 | 0.2361 ± 0.0610 | 0.3777 ± 0.0863 | 0.9759 ± 0.0129 | 0.9857 ± 0.0072 |
| One-Class SVM | Unsupervised | 69.97% ± 3.25% | 0.9623 ± 0.0555 | 0.2459 ± 0.0810 | 0.3847 ± 0.1082 | 0.8099 ± 0.0604 | 0.7830 ± 0.0760 |
| Local Outlier Factor (LOF) | Unsupervised | 60.18% ± 1.05% | 0.2000 ± 0.4000 | 0.0066 ± 0.0131 | 0.0127 ± 0.0254 | 0.3985 ± 0.0281 | 0.5363 ± 0.0237 |

### Key Research Findings
1. **Overfitting Mitigation:** Standard deviations across all 5 folds remained below ±0.7%, verifying that the architecture generalizes consistently without memorizing specific data splits.
2. **Fraud Detection Sensitivity:** The Proposed Hybrid Model achieved an average Recall of **98.69% ± 0.66%** and an F₂-Score of **0.9862 ± 0.0056**, catching over 98.6% of fraudulent companies with a modest false alarm rate (~1.6%).
3. **Unsupervised Outlier Separation:** Without any ground-truth labels, **Isolation Forest** attained an outstanding **0.9857 ROC-AUC** and **0.9759 PR-AUC**, demonstrating strong utility for initial audit screening in label-scarce environments.

---

## Repository Structure

```
.
├── audit_data/
│   ├── audit_risk.csv                    # Primary UCI audit dataset (776 records, 27 columns)
│   └── trial.csv                         # Supplementary audit trials dataset
│
├── outputs/
│   ├── model_comparison_results.csv      # Formatted 5-Fold CV benchmark metrics
│   ├── cv_results_5fold.csv              # Raw numerical CV metrics across all folds
│   ├── top_high_risk_companies.csv       # High-risk audit triage priority list
│   ├── pr_curves_comparison.png          # Comparative Precision-Recall curves
│   ├── roc_curves_comparison.png         # Comparative ROC curves
│   ├── feature_importance.png            # Top financial risk drivers
│   ├── trained_system.joblib             # Serialized model bundle for deployment
│   └── report_figures/                   # Publication-quality report figures
│
├── run_project.py                        # Master pipeline: preprocessing, training, evaluation
├── app.py                                # Interactive Streamlit Web Application
├── project_report.md                     # Markdown project documentation
├── Suspicious_Company_Detection_Project_Report.docx  # Academic Project Report Document
├── requirements.txt                      # Project Python dependencies
├── .gitignore                            # Git exclusion rules
└── README.md                             # Project overview and instructions
```

---

## Getting Started

### 1. Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)

### 2. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/harsha260/Suspicious-Company-Detection-Using-Machine-Learning-Based-Anomaly-Detection.git
pip install -r requirements.txt
```

### 3. Run Training and Benchmark Pipeline
Execute the master script to run the data preprocessing, model training, cross-validation, and visualization generation:
```bash
python run_project.py
```
This generates:
* Comparative benchmark metrics in `outputs/model_comparison_results.csv`
* Publication-ready plots in `outputs/`
* Triage rankings in `outputs/top_high_risk_companies.csv`
* Serialized inference bundle in `outputs/trained_system.joblib`

### 4. Launch the Interactive Web Dashboard
Run the Streamlit web interface:
```bash
python -m streamlit run app.py
```
Access the dashboard at `http://localhost:8501`:
* **Module 1 (Single Company Inspection):** Test corporate financial indicators with preset baseline profiles (`Normal / Compliant` vs `High Discrepancy / Suspicious`) or custom inputs.
* **Module 2 (Batch Audit Triage):** View and export high-risk company rankings with 1-click CSV download.
* **Module 3 (Comparative Benchmark):** Review 5-Fold cross-validation metrics, PR curves, and feature importance.

---

## Academic Citation & Attribution

* **Dataset:** Hooda, M., Balyan, A., & Sahu, A. (2018). *Audit Data Set*. UCI Machine Learning Repository. [https://archive.ics.uci.edu/dataset/475/audit+data](https://archive.ics.uci.edu/dataset/475/audit+data)
* **Institution:** Department of Computer Science and Engineering (AI & ML), Anil Neerukonda Institute of Technology and Sciences (ANITS), Visakhapatnam.
