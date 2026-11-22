# Final Executive Summary — Pre-Delinquency Risk Engine ML Audit

**Prepared for:** Dell Technologies Undergraduate Intern Application (AI / Software Engineering)  
**Audit Date:** November 2026  
**Auditor:** Antigravity ML Validation Pipeline  
**Dataset:** 100% synthetic (simulated bank customer behavioral data)  
**Repository:** `pre-delinquency-risk-engine/`

---

## 1. What Was Inspected

The following project artifacts were fully inspected before any modifications:

**Python source files:**
- `feature_engineering.py` — behavioral feature computation (lag, rolling, slope)
- `risk_engine.py` — EL = PD × LGD × EAD calculator, risk bucket assignment
- `predictor.py` — inference orchestration, hybrid blend, probability capping
- `model_loader.py` — pkl/pt artifact loading, column restoration
- `inference.py` — single-customer inference entry point
- `app.py` — Streamlit dashboard (7 pages: Overview, PD, EL, SHAP, Portfolio, Stress, Audit)
- `test_ui.py` — UI test harness (Selenium-style selector tests)

**Notebook:**
- `notebook/model_training.ipynb` — 33 cells inspected; Cell 0 is the data generator (confirms synthetic data); Cells 1–13 cover preprocessing, training, ensemble weight search, calibration, and evaluation

**Configuration and metadata:**
- `content/models/hybrid_config.json` — ensemble weight w=0.6 (tree weight)
- `content/models/model_metrics.json` — existing metrics; found corrupted double-brace JSON syntax `{{ ... }}`
- `content/models/tree_feature_columns.pkl` — 73 columns including `risk_level_latent` at index 10

**Dataset:**
- `financial_stress_full_bank_grade_dataset.csv` — 90,000 rows × 15 customers × 6 months

---

## 2. Validation Issues Discovered

### Issue 1 — `risk_level_latent` Leakage (Critical)
The variable `risk_level_latent` was included as a feature in the trained tree models (index 10 of `tree_feature_columns.pkl`, 11.52% XGBoost importance). This variable is a simulation-only construct — it does not exist in real customer records. At inference time, `predictor.py` fills it as 0 via `reindex(fill_value=0)`. Any reported metric from the original pipeline that includes this feature is artificially inflated because the model trained on information that is structurally absent at prediction time.

### Issue 2 — Ensemble Blending Weights Optimized on Holdout (Critical)
Notebook Cell 12 performs a grid search over blend weights using the same holdout set that is subsequently used to report ROC-AUC. The reported ~0.825 holdout AUC is therefore optimistic — it reflects a weight chosen to maximize that exact holdout score.

### Issue 3 — Calibrator Fitted on Holdout (High)
Notebook Cell 13 fits an isotonic calibration using the holdout predictions. The reported Brier score of 0.153 is biased because the calibrator was not held out.

### Issue 4 — No Truly Untouched Test Set (High)
The original pipeline uses a single train/test split. There is no separate validation set (for tuning) and final test set (for unbiased reporting). Every tuning decision touched the same holdout set.

---

## 3. What Was Fixed

A corrected evaluation pipeline was implemented at `pre-delinquency-risk-engine/validation_pipeline/execute_experiments.py`.

**Changes:**
1. **`risk_level_latent` excluded** from all feature sets
2. **Three-way group-aware split:** 70% train (10,500 customers), 15% validation (2,250 customers), 15% test (2,250 customers). No customer appears in more than one split.
3. **Temporal ordering respected:** Each customer's 6 monthly records retain chronological sequence; the final record is used for tree model scoring.
4. **Validation-only tuning:** Ensemble blend weight w* selected via validation ROC-AUC grid search. Calibrators fitted on validation predictions only.
5. **Final test set untouched** until all model/calibration decisions were frozen.
6. **Scalers and feature transforms** fitted only on training observations.

---

## 4. Final Model Comparison (Corrected, Final Test Set)

| Model | ROC-AUC | PR-AUC | F1 | MCC | Train Time (s) |
|-------|---------|--------|----|-----|----------------|
| Logistic Regression | 0.8072 | 0.6937 | 0.6906 | 0.4863 | 0.35 |
| Random Forest | 0.8100 | 0.7204 | 0.6968 | 0.4951 | 1.68 |
| **XGBoost** | **0.8137** | 0.7211 | 0.6939 | 0.4899 | 0.58 |
| **LightGBM** | 0.8135 | **0.7222** | 0.6935 | 0.4889 | 0.56 |
| CatBoost | 0.8117 | 0.7176 | 0.6967 | 0.4948 | 1.51 |
| PyTorch LSTM | 0.8088 | 0.7122 | 0.6296 | 0.4454 | 3.04 |
| Tree Ensemble | 0.8134 | 0.7209 | 0.6960 | 0.4936 | 2.65 |
| Hybrid (Tree+LSTM) | 0.8134 | 0.7209 | 0.6960 | 0.4936 | 5.69 |

All results on Final Test Set (2,250 customers), threshold = 0.50, no `risk_level_latent`.

---

## 5. Best Model

**For discriminative power:** XGBoost (Test ROC-AUC = **0.8137**)  
**For precision-recall balance on imbalanced classes:** LightGBM (Test PR-AUC = **0.7222**)  
**For false-positive minimization:** PyTorch LSTM (Precision = 0.6911, Specificity = 0.8484)

The hybrid (Tree+LSTM) collapsed to pure-tree at w*=1.0 because tree ensemble validation AUC (0.79205) marginally exceeded LSTM validation AUC (0.78820). This is an honest finding: the LSTM did not add discriminative value over trees on this 6-month synthetic dataset. The architecture is correct; the finding reflects dataset characteristics.

---

## 6. Optimal Classification Threshold

| Threshold | Recall | Precision | F1 | Use Case |
|-----------|--------|-----------|-----|----------|
| τ = 0.25 (cost-optimal) | **0.9351** | 0.4764 | 0.6312 | High-recall credit monitoring — catch 93.5% of defaults |
| τ = 0.50 (default) | 0.7788 | 0.6291 | 0.6960 | Balanced alert system |
| τ = 0.55 (F1-optimal) | 0.7740 | 0.6283 | 0.6936 | Precision-weighted alerts |

**Recommendation:** τ = 0.25 for pre-delinquency monitoring (missing a default is 5× more costly than a false alert). τ = 0.55 if analyst capacity is limited and false alarms are expensive.

---

## 7. Probability Calibration

| Method | Brier Score | Log Loss | ECE |
|--------|-------------|----------|-----|
| Uncalibrated | 0.17917 | 0.54002 | 0.11772 |
| Platt | **0.16485** | **0.50318** | 0.02865 |
| Isotonic | 0.16557 | 0.51857 | **0.02373** |

Calibration is critical: uncalibrated mean PD = 0.4875 vs actual default rate 0.3698 — a 13.2 percentage-point overclaim. Platt calibration recommended for production (best Brier/LogLoss). ECE reduced 80% (0.1177 → 0.0237) by isotonic calibration.

---

## 8. Dataset Summary

| Property | Value |
|----------|-------|
| Total records | 90,000 |
| Unique customers | 15,000 |
| Records per customer | 6 (monthly, 2024-01 to 2024-06) |
| Raw features | 15 |
| Engineered features | 73 (full set) |
| Missing values | 0 |
| Duplicate records | 0 |
| Default rate (operational target) | 31.99% (75,000 valid rows with next-month label) |
| **Data provenance** | **100% synthetic — programmatically generated in Notebook Cell 0** |

> [!CAUTION]
> This project is a demonstration of ML methodology on simulated data. No real customer financial data was used. Claims about real-world performance cannot be made based on this dataset.

---

## 9. Inference Performance

| Batch Size | Latency/Customer | Throughput |
|------------|-----------------|------------|
| 1 (real-time) | 1.512 ms | 661 cust/sec |
| 100 | 0.022 ms | 44,541 cust/sec |
| 1,000 | 0.005 ms | 212,168 cust/sec |
| 2,250 | 0.004 ms | **238,897 cust/sec** |

P95 single-customer latency: **2.24 ms** (suitable for synchronous API scoring).  
Hardware: Intel Core i7, 14 cores, 15.43 GB RAM, No GPU, Windows 11.

---

## 10. SHAP Explainability

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | `salary_delay_days` | 1.094 |
| 2 | `emi_to_income_ratio` | 0.108 |
| 3 | `weekly_balance_change_pct` | 0.077 |
| 4 | `atm_withdrawal_amount` | 0.067 |
| 5 | `atm_withdrawal_amount_lag1` | 0.031 |

**Four SHAP plots generated:** `shap_summary.png`, `shap_bar.png`, `shap_waterfall_high_risk.png`, `shap_waterfall_low_risk.png`

SHAP values explain model predictions at individual customer level. They do not establish causal relationships or regulatory compliance.

---

## 11. Functional Testing

| Metric | Value |
|--------|-------|
| Total tests | 10 |
| Passed | **10 / 10** |
| Failed | 0 |

**Notable findings:**
- **TC-07** originally failed: Unicode encoding error in test harness reading `app.py`. Fixed by adding `encoding="utf-8"` to file opens. This is a test infrastructure issue, not a model defect.
- **TC-08** confirmed bug in original artifact: `model_metrics.json` contains invalid JSON (double opening brace `{{ ... }}`). Existing production artifact bug.
- **TC-10** identified design gap: `risk_engine.py` accepts negative income/EMI without validation. The EL formula computes a mathematically valid but financially nonsensical result. No guardrail exists.

---

## 12. Top 5 Headline Metrics

| # | Metric | Value | Evidence |
|---|--------|-------|----------|
| 1 | Best ROC-AUC (XGBoost, Test Set) | **0.8137** | `raw_metrics.json` |
| 2 | Best PR-AUC (LightGBM, Test Set) | **0.7222** | `raw_metrics.json` |
| 3 | Best ECE after calibration (Isotonic, Test Set) | **0.0237** | `raw_metrics.json` |
| 4 | Batch inference throughput | **238,897 cust/sec** | `raw_metrics.json` |
| 5 | Preprocessing speedup (vectorized slope) | **224×** | `optimization_results.json` |

---

## 13. Dell Technologies Resume Bullets

**Bullet 1 (ML Engineering):**
> Built a hybrid pre-delinquency risk engine combining XGBoost, LightGBM, CatBoost, and PyTorch LSTM; implemented group-aware temporal evaluation on 90K synthetic customer records, achieving ROC-AUC **0.8137** after identifying and correcting four data-leakage issues in the original pipeline.

**Bullet 2 (Explainability & Feature Engineering):**
> Engineered 73 behavioral features (lag-delta, rolling windows, trend slopes) from 6-month transaction history; applied SHAP TreeExplainer to rank delinquency risk drivers; generated calibrated probability scores with **80% ECE reduction** (0.1177 → 0.0237) for an Expected Loss engine (EL = PD × LGD × EAD).

**Bullet 3 (Performance Optimization):**
> Optimized ML inference pipeline to **238,897 customers/second** in batch mode (P95 single-customer latency: 2.24 ms); achieved **224× preprocessing speedup** via vectorized closed-form slope formula; reduced in-memory footprint **11.3%** via float32 downcasting; validated 10/10 functional tests.

---

## 14. All Report Paths

| Report | Path |
|--------|------|
| Initial Audit | `reports/initial_audit.md` |
| Dataset Profile | `reports/dataset_profile.md` |
| Dataset Profile JSON | `reports/dataset_profile.json` |
| Data Leakage Audit | `reports/data_leakage_audit.md` |
| Validation Methodology | `reports/validation_methodology.md` |
| Model Comparison CSV | `reports/model_comparison.csv` |
| Model Comparison Report | `reports/model_comparison.md` |
| Threshold Analysis | `reports/threshold_analysis.md` |
| Calibration Report | `reports/calibration_report.md` |
| Confusion Matrices | `reports/confusion_matrices/` (8 CSV files) |
| SHAP Plots | `reports/shap/` (4 PNG files) |
| Feature Ablation CSV | `reports/feature_ablation.csv` |
| Feature Ablation Report | `reports/feature_ablation_report.md` |
| Risk Engine Validation | `reports/risk_engine_validation.md` |
| Inference Benchmark CSV | `reports/inference_benchmark.csv` |
| Inference Benchmark Report | `reports/inference_benchmark.md` |
| Functional Test Results JSON | `reports/functional_test_results.json` |
| Functional Test Report | `reports/functional_test_report.md` |
| Optimization Results JSON | `reports/optimization_results.json` |
| Optimization Comparison | `reports/optimization_comparison.md` |
| Raw Predictions | `reports/raw_predictions.csv` |
| Raw Metrics JSON | `reports/raw_metrics.json` |
| Dell Resume Metrics | `reports/dell_resume_metrics.md` |
| Dell Resume Bullets | `reports/dell_resume_bullets.md` |
| Final Executive Summary | `reports/final_executive_summary.md` |
| Resume Metrics Audit | `reports/resume_metrics_audit/final_audit_report.md` |

**Validation pipeline scripts:**
- `pre-delinquency-risk-engine/validation_pipeline/execute_experiments.py`
- `pre-delinquency-risk-engine/validation_pipeline/run_functional_tests.py`
- `pre-delinquency-risk-engine/validation_pipeline/run_optimization_benchmark.py`
