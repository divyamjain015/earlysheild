# INITIAL AUDIT REPORT: PRE-DELINQUENCY RISK ENGINE

**Date:** November 2026  
**Auditor:** ML Validation & AI Systems Engineering  
**Project:** Pre-Delinquency Risk Engine  
**Target Repository:** `pre-delinquency-risk-engine`

---

## 1. Executive Summary

This initial audit provides an exhaustive codebase and architecture inspection of the Pre-Delinquency Risk Engine repository prior to executing quantitative benchmark interventions. The project is an end-to-end financial risk modeling application combining gradient-boosted decision trees (XGBoost, LightGBM, CatBoost) and sequential deep learning (PyTorch LSTM) to predict probability of customer delinquency (PD), assign risk buckets, and compute portfolio Expected Loss ($EL = PD \times LGD \times EAD$).

While the engineering design features modern ML practices (modular feature engineering, SHAP explanations, Streamlit UI), our audit reveals **critical data leakage and methodological flaws in the model evaluation pipeline**—specifically test-set contamination during hybrid ensemble blending weight optimization and probability calibration. Furthermore, the dataset was confirmed to be **100% synthetically simulated** through an automated generation script embedded in the training notebook.

---

## 2. Python Source File Inspection

| File | Primary Responsibilities | Critical Findings / Vulnerabilities |
| :--- | :--- | :--- |
| `feature_engineering.py` | Generates 10-feature lag, delta, rolling (mean, max, std, slope), and persistence features. | Uses `np.polyfit` inside Pandas `rolling.apply` (high computational complexity: $O(N)$ polynomial solver overhead). Fills missing values with `0` globally (`df.fillna(0)`), creating subtle distributional discrepancies compared to the notebook training pipeline which dropped NaNs (`df.dropna()`). |
| `risk_engine.py` | Computes Expected Loss ($EL = PD \times LGD \times EAD$) and assigns discrete risk buckets (`VERY LOW`, `LOW`, `MEDIUM`, `HIGH`, `VERY HIGH`). | LGD and EAD are strictly rule-based heuristic formulas parameterized via `hybrid_config.json`, **not** empirically learned credit risk models. Lacks input range validation (e.g., negative income, negative EMI). |
| `predictor.py` | Orchestrates tree and LSTM preprocessing, runs inference, performs ensemble blending, applies calibration, clips PD to $[0.01, 0.95]$. | Mixes single-customer inference logic with hardcoded assumptions; assumes fixed batch dimensions for LSTM tensors. |
| `model_loader.py` | Loads serialized models (`xgb_model.pkl`, `lgb_model.pkl`, `cat_model.cbm`, `lstm_model_state.pt`, `calibrator.pkl`, scalers, columns, config). | Instantiates custom `LSTMModel` (1-layer, hidden size 64) and deserializes weights. Uses relative paths; lacks fallback if model artifacts are corrupted. |
| `inference.py` | End-to-end script demonstrating inference on dummy customer data, computing EL, and generating SHAP waterfall & bar plots. | Uses `shap.KernelExplainer` over the full blended black-box pipeline, keeping LSTM input fixed while perturbing tree features. Slow execution (~minutes) due to model evaluations across background samples. |
| `app.py` | Streamlit interactive web dashboard for real-time risk assessment, financial ratio display, and risk tiering. | Hardcoded customer selector reads directly from raw dataset. Appends current month to history, runs feature engineering on dynamic dataframe, renders cached SHAP plots from disk. |
| `test_ui.py` | Selenium-based automated browser test for Streamlit interface. | Out of sync with `app.py`: queries for `text_input("Customer ID")` whereas `app.py` uses `selectbox("Select Customer")`. Fails upon execution. |

---

## 3. Model Training Notebook Inspection (`notebook/model_training.ipynb`)

The training notebook contains 33 executable cells covering:
1. **Synthetic Data Synthesis (Cell 0):** Generates 15,000 synthetic customers across 6 monthly time slices (90,000 records).
2. **Feature Engineering (Cell 1 & Cell 9):** Computes lag 1, lag 2, delta 1, delta 2, 3-month rolling mean/max/std/slope, and persistence flags for high EMI, salary delay, and high credit card utilization.
3. **Data Splitting (Cell 9):** Performs an 80/20 customer group split (12,000 train customers / 3,000 holdout customers).
4. **Tree Optimization (Cell 9):** Employs Optuna with 3-fold `GroupKFold` on the training partition to tune hyperparameters for XGBoost, LightGBM, and CatBoost.
5. **LSTM Sequence Modeling (Cell 10):** Formulates 3-month sequences of raw numerical features per customer, standardizes with `StandardScaler` (fit on train), and trains a 1-layer LSTM for 15 epochs using BCE loss.
6. **Ensemble Blending (Cell 12):** Evaluates simple average (0.5/0.5), fixed blend (0.6/0.4), and grid searches optimal weight $w \in [0, 1]$ in steps of 0.05.
7. **Probability Calibration (Cell 13):** Fits an `IsotonicRegression` calibrator on holdout predictions against holdout targets.

---

## 4. Dataset Schema & Metadata Analysis

- **File:** `financial_stress_full_bank_grade_dataset.csv`
- **Total Records:** 90,000
- **Unique Customers:** 15,000 (customer IDs `CUST0000000` through `CUST0014999`)
- **Records per Customer:** Exactly 6 records per customer (balanced temporal panel)
- **Temporal Horizon:** Discrete monthly intervals from `2024-01` to `2024-06`
- **Missing Values:** Exactly 0 missing values across all 17 raw columns
- **Duplicate Rows:** 0

### Schema Breakdown:
- **Identifiers & Time:** `customer_id` (string), `month` (string YYYY-MM)
- **Demographics / Segment:** `customer_segment` (`salaried` [70%], `self_employed` [30%]), `region_tier` (`tier_1` [40%], `tier_2` [35%], `tier_3` [25%])
- **Product Profile:** `product_type` (`personal_loan`, `credit_card`, `bnpl`), `active_products_count` (1, 2, or 3)
- **Financial & Behavioral Metrics:**
  - `credit_card_utilization` (float, 0.20 to 0.95)
  - `total_monthly_obligation` (float)
  - `emi_amount` (int)
  - `days_to_emi` (int, 10 to 25)
  - `emi_to_income_ratio` (float)
  - `salary_delay_days` (int, 0 to 30)
  - `weekly_balance_change_pct` (float, -40% to +10%)
  - `atm_withdrawal_amount` (float)
  - `monthly_income` (int, 20,000 to 150,000)
- **Latent & Target Variables:**
  - `risk_level_latent` (int: 0, 1, 2, 3) - unobservable simulation ground truth
  - `loan_default_observed` (binary 0/1): Observed delinquency event in the current month

---

## 5. Target Variable Definition

- In the raw dataset, `loan_default_observed` indicates whether a delinquency occurred in month $t$ (overall positive rate: **30.72%**; 27,647 defaults out of 90,000 observations).
- In the training pipeline (Cell 9 and Cell 10), the target is transformed to **predictive pre-delinquency** by shifting by -1 within each customer group:
  $$\text{loan\_default\_next\_month}_t = \text{loan\_default\_observed}_{t+1}$$
- This aligns with the real-world operational goal: predicting whether a currently performing customer will transition to default in the subsequent month ($t+1$).
- Because month 6 has no observed $t+1$ outcome, month 6 rows have NaN targets and cannot be used as target labels for next-month prediction.

---

## 6. Dataset Provenance: Real vs. Synthetic Investigation

**Determination: The dataset is 100% SYNTHETIC / SIMULATED.**

### Undeniable Evidence:
1. **Explicit Data Generator in Codebase:** Cell 0 of `notebook/model_training.ipynb` contains the exact simulation script that generated `financial_stress_full_bank_grade_dataset.csv`.
2. **Fixed Parametric Formulas:**
   - Latent risk variable `risk` initialized via `np.random.choice([0, 1, 2, 3], p=[0.5, 0.25, 0.15, 0.10])` with monthly Markov drift $\Delta \in \{-1, 0, 1\}$.
   - Base delinquency probability explicitly defined as:
     $$\text{base\_prob} \in [0.03, 0.08, 0.22, 0.45][\text{risk}]$$
   - Multiplicative pressure formula:
     $$\text{pressure} = 1 + 1.5 \left(\frac{\text{total\_obligation}}{\text{income}}\right) + 0.02 \cdot \text{salary\_delay\_days} - 0.01 \cdot \max(\text{weekly\_balance\_change}, 0)$$
   - Deterministic demographic biases added: $+5\%$ for `self_employed`, $+5\%$ for `tier_3`.
   - Binary labels generated via Bernoulli sampling: `np.random.rand() < default_prob`.
3. **Disclosure:** In all technical documentation and resumes, this dataset must be clearly stated as a **synthetically generated bank-grade behavioral dataset** designed to replicate realistic credit stress dynamics, rather than empirical proprietary bank data.

---

## 7. Audit of Existing Trained Models and Artifacts

| Artifact | Type | Stored Size | Configured Parameters / Notes |
| :--- | :--- | :--- | :--- |
| `xgb_model.pkl` / `.json` | XGBoost Classifier | ~380 KB | Optuna-tuned tree model |
| `lgb_model.pkl` / `.txt` | LightGBM Classifier | ~700 KB | Optuna-tuned tree model |
| `cat_model.cbm` | CatBoost Classifier | ~1.1 MB | Optuna-tuned tree model |
| `lstm_model_state.pt` | PyTorch State Dict | ~72 KB | 1-layer LSTM, input_dim=15, hidden_dim=64, dropout=0.2 |
| `lstm_scaler.pkl` | Scikit-learn StandardScaler | ~2 KB | Fitted on 12,000 training customers |
| `calibrator.pkl` | IsotonicRegression / Calibrator | ~25 KB | Calibrated on holdout set |
| `hybrid_config.json` | JSON Configuration | 1.1 KB | Tree weight: 0.9, LSTM weight: 0.1, EL parameters |
| `model_metrics.json` | JSON Metrics File | 550 B | **Corrupted Syntax:** Double opening brace on line 2 (`{ { ... } }`) |

---

## 8. Critical Validation & Leakage Audit

### Flaw 1: Holdout Contamination During Ensemble Weight Search
In Cell 12 of `model_training.ipynb`:
```python
for w in np.arange(0, 1.01, 0.05):
    blended = w * tree_preds + (1 - w) * lstm_preds
    auc = roc_auc_score(y_hybrid_actuals, blended)
    if auc > best_auc:
        best_auc = auc
        best_weight = w
```
Here, `tree_preds`, `lstm_preds`, and `y_hybrid_actuals` are derived from the 20% **holdout partition** (3,000 customers). Searching for the blending weight $w$ on the holdout partition and selecting $w=0.90$ directly overfits the holdout set.

### Flaw 2: Holdout Contamination During Probability Calibration
In Cell 13 of `model_training.ipynb`:
```python
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(best_blended_preds, y_hybrid_actuals)
calibrated_pd = iso.predict(best_blended_preds)
```
The calibrator was fitted on the exact same holdout predictions and evaluated against those same holdout ground-truth labels. Reporting Brier score ($0.153$) and ROC-AUC ($0.825$) on data used to fit the calibrator violates standard machine learning validation protocol.

### Flaw 3: Absence of an Untouched Final Test Set
The pipeline partitioned data into only **Train (80%)** and **Holdout (20%)**. Because the holdout set was used for model tuning, blending, and calibration, **zero independent test data remained** for un-biased risk estimation.

---

## 9. Required Corrective Action Plan

1. **Re-architect the Data Split:**
   - Establish a strict 3-way partition: **Train (70%)**, **Validation (15%)**, and **Test (15%)** stratified or grouped strictly by `customer_id`.
   - Zero customer overlap across splits ($Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$).
2. **Proper Role Assignment:**
   - **Train Set:** Fit tree models (XGBoost, LightGBM, CatBoost), fit baseline models (Logistic Regression, Random Forest), fit LSTM, fit preprocessing scalers/encoders.
   - **Validation Set:** Tune hyperparameters, find optimal ensemble blending weights ($w_{tree}$ vs $w_{lstm}$), fit probability calibrators (Platt / Isotonic) on validation out-of-sample predictions.
   - **Untouched Final Test Set:** Evaluate all models once using the chosen configuration to produce unbiased, statistically defensible metrics.
3. **Fix Preprocessing and Serialized Configs:**
   - Fix corrupted `model_metrics.json`.
   - Optimize polynomial slope calculation to remove $O(N)$ execution bottlenecks.
   - Ensure reproducible benchmarking across models and hardware.

---

*Initial audit concluded. Proceeding with dataset profiling and corrected validation pipeline execution.*
