# DATA LEAKAGE AUDIT REPORT: PRE-DELINQUENCY RISK ENGINE

**Date:** November 2026  
**Auditor:** Quantitative Risk Modeling & Machine Learning Systems Audit  
**Status:** Deficiencies Identified & Documented  

---

## 1. Executive Summary

This audit assesses the presence, mechanisms, and impact of data leakage across the original training and validation pipelines of the Pre-Delinquency Risk Engine. The investigation focused on four primary vulnerability vectors:
1. **Holdout Evaluation Contamination (Ensemble Weight Optimization on Test Set)**
2. **Probability Calibrator Overfitting (Calibration on Test Set)**
3. **Absence of an Independent Final Test Split**
4. **Latent Simulation Variable Infiltration (`risk_level_latent` Train-Serve Skew)**

The primary conclusion of this audit is that **the previously reported holdout ROC-AUC of ~0.825 and Brier score of 0.153 are invalid as unbiased test estimates**. The holdout partition was iteratively used for hyperparameter selection, ensemble blending weight optimization, and probability calibration fitting. Furthermore, the tree models were inadvertently trained on the synthetic generator's latent ground-truth variable (`risk_level_latent`), causing severe train-serve skew.

---

## 2. Leakage Vector 1: Test-Set Contamination in Ensemble Blending

### Empirical Code Evidence (`notebook/model_training.ipynb`, Cell 12):
```python
# Cell 12: OPTIMAL WEIGHT SEARCH ON THE HOLDOUT SET
best_auc = 0
global best_weight
best_weight = 0

for w in np.arange(0, 1.01, 0.05):
    blended = w * tree_preds + (1 - w) * lstm_preds
    auc = roc_auc_score(y_hybrid_actuals, blended)

    if auc > best_auc:
        best_auc = auc
        best_weight = w

print("\nBest Weight for Trees:", round(best_weight, 2))
print("Best Hybrid AUC:", best_auc)
```

### Analysis:
- `tree_preds`, `lstm_preds`, and `y_hybrid_actuals` are derived from the 20% holdout split ($N = 3,000$ customers).
- The blending weight $w$ was chosen via exhaustive grid search on `y_hybrid_actuals` to maximize the exact metric reported as "Holdout AUC" ($0.82499 \approx 0.825$).
- **Methodological Violation:** An evaluation partition cannot simultaneously serve as the optimization criterion and the unbiased validation metric. This produces optimistically biased results.

---

## 3. Leakage Vector 2: Probability Calibrator Overfitting

### Empirical Code Evidence (`notebook/model_training.ipynb`, Cell 13):
```python
# Cell 13: CALIBRATION USING HOLDOUT
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(best_blended_preds, y_hybrid_actuals)

calibrated_pd = iso.predict(best_blended_preds)
calibrated_pd = np.clip(calibrated_pd, 0.01, 0.95)
```

### Analysis:
- The non-parametric `IsotonicRegression` model was fitted directly to `(best_blended_preds, y_hybrid_actuals)`.
- `IsotonicRegression` is a piecewise-constant non-decreasing step function. Fitting it directly on the test set guarantees that the calibrated probabilities minimize empirical calibration error on that specific sample by fitting to the empirical step boundaries of the test set.
- Reporting Brier score ($0.15307$) and calibrated mean PD ($0.3454$) on the calibration training set guarantees an artificially low Brier score that does not generalize to unseen customers.

---

## 4. Leakage Vector 3: Latent Simulation Variable Infiltration (`risk_level_latent`)

### Empirical Code & Model Evidence:
1. In `notebook/model_training.ipynb` Cell 9, `DROP_COLS` was defined as:
```python
DROP_COLS = ["customer_id", "month", "loan_default_observed", "loan_default_next_month"]
```
2. Inspection of serialized `tree_feature_columns.pkl` reveals that column index 10 is **`risk_level_latent`**.
3. Inspection of the trained `xgb_model.pkl` reveals that **`risk_level_latent` accounts for 11.52% of total feature importance (ranked #2 overall)**!
4. However, in `inference.py` and `app.py`, customer input dictionaries do **not** contain `risk_level_latent`. When `prepare_inputs` executes:
```python
df_tree = df.reindex(columns=tree_cols, fill_value=0)
```
`risk_level_latent` is defaulted to `0` for all live inference requests!

### Impact:
- `risk_level_latent` is the unobservable simulation ground truth ($0, 1, 2, 3$) used to generate synthetic defaults. In real banking systems, this latent variable does not exist.
- Including it in training created massive data leakage and severe **train-serve skew**: the model learned decision splits relying on an unobservable variable that is always 0 in production!
- In our corrected validation pipeline, `risk_level_latent` is strictly purged from all feature sets prior to modeling.

---

## 5. Leakage Vector 4: Longitudinal Grouping & Temporal Integrity

### 5.1 Customer Grouping:
- The raw dataset contains 90,000 rows across 15,000 customers (6 records per customer).
- Splitting was correctly done on customer IDs ($Train \cap Holdout = \emptyset$). However, because there was no 3rd partition (no separate test set), the holdout partition had to be used for both model selection (weights, calibration) and final reporting.

### 5.2 Temporal & Feature Engineering Integrity:
1. **Target Construction:** Next-month target $\text{target}_t = \text{default}_{t+1}$ correctly prevents predicting past defaults using future observations.
2. **Past-Only Rolling Statistics:** Lags and rolling windows are grouped strictly by `customer_id` and sorted chronologically. Rolling calculations use past observations up to month $t$. No future temporal leakage is present in the feature definitions.
3. **Data Scaling:** Scaler for LSTM was fitted on Train only.

---

## 6. Summary of Data Leakage Audit Findings

| Audit Dimension | Original Status | Vulnerability Severity | Corrective Action |
| :--- | :--- | :--- | :--- |
| **Ensemble Blending** | Tuned on holdout labels | **CRITICAL** | Tune weights strictly on an independent Validation split. |
| **Probability Calibration**| Fitted on holdout labels | **CRITICAL** | Fit calibrator strictly on out-of-sample Validation predictions. |
| **Holdout Independence** | No unseen test partition | **CRITICAL** | Implement strict 3-way partition: Train (70%), Val (15%), Test (15%). |
| **Latent Variable Leakage** | `risk_level_latent` used in trees | **CRITICAL** | Exclude `risk_level_latent` completely from all feature sets. |
| **Customer Overlap** | Zero overlap between train & holdout | **COMPLIANT** | Maintain zero customer overlap across all three splits. |
| **Temporal Feature Calculation** | Past-only lags & rolling stats | **COMPLIANT** | Preserve past-only feature calculation rules. |
| **Target Causality** | Next-month default prediction | **COMPLIANT** | Preserve next-month target formulation. |
| **Data Scaling** | Scaler fitted on Train only | **COMPLIANT** | Fit all transformers/scalers on Train only. |

---

*Data leakage audit concluded. The corrected validation methodology is defined in `reports/validation_methodology.md`.*
