# DATASET PROFILE REPORT: PRE-DELINQUENCY FINANCIAL STRESS DATASET

**Dataset Name:** `financial_stress_full_bank_grade_dataset.csv`  
**File Location:** `pre-delinquency-risk-engine/financial_stress_full_bank_grade_dataset.csv`  
**Analysis Date:** November 2026  
**Auditor:** Quantitative Risk Modeling & AI Systems Engineering  

---

## 1. Executive Summary & Provenance Disclosure

This dataset profile provides a rigorous statistical analysis of the financial stress data utilized in the Pre-Delinquency Risk Engine. 

> [!WARNING]
> **Dataset Provenance Disclosure: Synthetic / Simulated Data**  
> Comprehensive static code inspection confirms that this dataset is **100% synthetically generated** using a stochastic simulation script located in Cell 0 of `notebook/model_training.ipynb`. It is **not** an empirical dataset sourced from production banking transactions or credit bureau records. The data generation process simulates financial stress dynamics through parameterized mathematical formulas, latent Markov risk states, and Bernoulli trials. Consequently, all experimental findings must be reported as algorithmic validation on synthetic benchmark data, and must **never** be claimed as empirical banking verification.

---

## 2. High-Level Dataset Dimensions

| Metric | Measured Value | Operational Implications |
| :--- | :--- | :--- |
| **Total Rows / Records** | 90,000 | Sufficient scale for deep sequence learning and gradient boosting. |
| **Unique Customers** | 15,000 | Clean cohort of 15,000 synthetic retail banking borrowers. |
| **Total Raw Columns** | 17 | 10 financial/behavioral, 3 categorical/demographic, 2 keys, 2 risk/targets. |
| **Records per Customer** | Exactly 6 (Min: 6, Max: 6, Mean: 6.0) | Perfectly balanced longitudinal panel of monthly customer snapshots. |
| **Temporal Span** | 2024-01 to 2024-06 (6 months) | Covers 6 consecutive monthly reporting cycles (15,000 records/month). |
| **Duplicate Records** | 0 (0.00%) | No duplicate primary observation rows. |
| **Missing Values** | 0 (0.00% across all 17 columns) | No imputation required on raw inputs; missingness arises solely via lag shifts. |

---

## 3. Target Variable Analysis & Operational Formulation

### 3.1 Raw Observed Delinquency (`loan_default_observed`)
The raw binary variable `loan_default_observed` indicates whether a customer exhibited a delinquency event during month $t$:
- **Non-Delinquent Count ($y = 0$):** 62,353 (69.28%)
- **Delinquent Count ($y = 1$):** 27,647 (30.72%)
- **Baseline Event Rate:** 30.72%
- **Class Imbalance Ratio:** 2.26 : 1 (Moderate class imbalance; amenable to standard tree weighting or scale_pos_weight $\approx 2.25$).

### 3.2 Operational Pre-Delinquency Target (`loan_default_next_month`)
In an operational early warning system, predicting default at time $t$ using time $t$ data is reactive. The core business task is predicting delinquency in month $t+1$ using historical behavior up to month $t$:
$$\text{loan\_default\_next\_month}_t = \text{loan\_default\_observed}_{t+1}$$

Because month 6 (`2024-06`) is the terminal observation month in the dataset, shifting by $-1$ results in NaN targets for all 15,000 month 6 records:
- **Valid Observations for Modeling:** 75,000 (Months 2024-01 to 2024-05)
- **Dropped Terminal Records:** 15,000 (Month 2024-06)
- **Non-Delinquent Next Month ($y_{t+1} = 0$):** 51,010 (68.01%)
- **Delinquent Next Month ($y_{t+1} = 1$):** 23,990 (31.99%)
- **Operational Next-Month Default Rate:** 31.99%

---

## 4. Categorical & Demographic Segment Distributions

| Categorical Feature | Category | Record Count | Proportional Share | Delinquency Rate ($y=1$) |
| :--- | :--- | :--- | :--- | :--- |
| **Customer Segment** | `salaried` | 63,000 | 70.00% | 29.85% |
| | `self_employed` | 27,000 | 30.00% | 32.74% |
| **Region Tier** | `tier_1` | 36,000 | 40.00% | 28.91% |
| | `tier_2` | 31,500 | 35.00% | 31.06% |
| | `tier_3` | 22,500 | 25.00% | 33.12% |
| **Product Type** | `personal_loan` | 30,000 | 33.33% | 28.40% |
| | `bnpl` | 30,000 | 33.33% | 27.95% |
| | `credit_card` | 30,000 | 33.33% | 35.81% |

### Observations:
- `self_employed` borrowers exhibit a higher delinquency rate (32.74%) compared to `salaried` (29.85%), reflecting the $+5\%$ multiplier encoded into the simulation generator.
- `tier_3` regions have the highest delinquency rate (33.12%), versus 28.91% in `tier_1`.
- `credit_card` accounts have markedly higher delinquency (35.81%) compared to personal loans (28.40%) and BNPL (27.95%), driven by elevated utilization ceilings (up to 0.95).

---

## 5. Statistical Profile of Numerical Features & Outliers

All 10 numerical features were profiled across mean, standard deviation, quartiles ($Q_1, Q_2, Q_3$), min/max, skewness, kurtosis, and Tukey's Interquartile Range (IQR) outlier counts:

| Feature Name | Mean | Std Dev | Min | Median ($Q_2$) | Max | Skewness | IQR Outlier Pct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `active_products_count` | 1.700 | 0.714 | 1 | 2 | 3 | 0.528 | 0.00% |
| `credit_card_utilization` | 0.490 | 0.178 | 0.200 | 0.461 | 0.950 | 0.569 | 0.00% |
| `monthly_income` (₹) | 85,027 | 37,569 | 20,000 | 85,015 | 149,998 | -0.001 | 0.00% |
| `emi_amount` (₹) | 21,248 | 12,238 | 2,000 | 18,978 | 59,997 | 0.749 | 1.28% |
| `total_monthly_obligation` (₹)| 23,331 | 12,960 | 2,342 | 20,958 | 65,372 | 0.709 | 1.25% |
| `days_to_emi` | 17.50 | 4.61 | 10 | 18 | 25 | -0.003 | 0.00% |
| `emi_to_income_ratio` | 0.250 | 0.086 | 0.100 | 0.250 | 0.400 | -0.001 | 0.00% |
| `salary_delay_days` | 6.84 | 6.78 | 0 | 4 | 29 | 1.268 | 5.16% |
| `weekly_balance_change_pct` | -6.98% | 11.23% | -40.00% | -5.73% | 10.00% | -0.718 | 0.00% |
| `atm_withdrawal_amount` (₹) | 33,398 | 19,308 | 6,005 | 28,958 | 134,846 | 1.157 | 4.54% |

### Outlier & Distribution Highlights:
1. **Salary Delay Days:** Positively skewed ($\text{skew} = 1.27$). 5.16% of observations exceed the upper IQR fence ($> 19$ days), reflecting severe stress states in latent risk tiers 2 and 3.
2. **ATM Withdrawal Amount:** Positively skewed ($\text{skew} = 1.16$), scaling directly with credit card utilization multiplier $(1 + \text{utilization})$.
3. **EMI-to-Income Ratio:** Uniformly bounded between $0.10$ and $0.40$ ($\text{mean} = 0.25$), by construction of the simulation generator.
4. **Income Distribution:** Perfectly symmetric across the range ₹20,000 to ₹150,000 ($\text{skew} \approx 0.00$).

---

## 6. Simulation Engine Specification (Mathematical Details)

The generator employs the following data generating process (DGP):
1. **Latent Risk State Transition:**
   $$R_0 \sim \text{Categorical}(\{0, 1, 2, 3\}, [0.50, 0.25, 0.15, 0.10])$$
   $$R_t = \text{clip}(R_{t-1} + \Delta_t, 0, 3), \quad \Delta_t \sim \text{Categorical}(\{-1, 0, 1\}, [0.15, 0.60, 0.25])$$
2. **Stress Features Generation Given $R_t$:**
   $$\text{salary\_delay} \sim \text{Uniform}([0, 2, 7, 15]_{R_t}, [3, 7, 15, 30]_{R_t})$$
   $$\text{balance\_change} \sim \text{Uniform}([-3, -10, -25, -40]_{R_t}, [10, 5, -5, -15]_{R_t})$$
   $$\text{atm\_withdrawal} \sim \text{Uniform}([5k, 15k, 25k, 40k]_{R_t}, [20k, 30k, 45k, 70k]_{R_t}) \times (1 + \text{utilization})$$
3. **Default Probability Equation:**
   $$\text{base\_PD} = [0.03, 0.08, 0.22, 0.45]_{R_t}$$
   $$\text{pressure} = 1 + 1.5 \left(\frac{\text{total\_obligation}}{\text{income}}\right) + 0.02 \cdot \text{salary\_delay} - 0.01 \cdot \max(\text{balance\_change}, 0)$$
   $$\text{PD}^* = \min(\text{base\_PD} \times \text{pressure} \times 1.05^{\mathbf{1}_{\text{self\_employed}}} \times 1.05^{\mathbf{1}_{\text{tier\_3}}}, 0.95)$$
   $$Y_t \sim \text{Bernoulli}(\text{PD}^*)$$
   *(with $8\%$ label noise near boundary for risk tiers 1 and 2)*.

---

## 7. Implications for Modeling & Validation Architecture

1. **Longitudinal Grouping:** Multiple records (6) exist for each of the 15,000 customers. Random row-level splitting would leak future customer states into training. Group-aware customer splitting is strictly mandatory.
2. **Temporal Structure:** While customers have a 6-month sequence, the cohort time window is uniform (`2024-01` to `2024-06`). An out-of-time (OOT) split across months would truncate sequential history for LSTM. Therefore, a **customer-grouped stratified split** (where all 6 months of a customer remain strictly in Train, Validation, or Test) is the mathematically sound architecture.
3. **Target Transformation:** Shifting the target by $-1$ creates a clean 5-month longitudinal sequence per customer ($N = 75,000$ paired feature-target observations), cleanly isolating pre-delinquency prediction.

---

*End of Dataset Profile Report. JSON companion available at `reports/dataset_profile.json`.*
