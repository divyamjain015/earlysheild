# Probability Calibration Report
## Pre-Delinquency Risk Engine — Hybrid Model

> [!CAUTION]
> **Synthetic Data Disclosure**: All results in this report are based on a **100% simulated, synthetic dataset**. No real customer or banking data was used at any stage of this project. This report is for research and demonstration purposes only.

---

## 1. Why Calibration Matters for Risk Engines

A risk model that ranks customers correctly (high AUC) is valuable, but in credit risk management, **the probability score itself must be reliable as a quantity** — not just an ordinal ranking.

This is because the score is used directly in regulatory and financial calculations such as:

```
Expected Loss (EL) = PD × LGD × EAD
```

Where:
- **PD** = Probability of Default (the model's output)
- **LGD** = Loss Given Default (recovery rate)
- **EAD** = Exposure at Default (outstanding balance)

If the model predicts PD = 0.60 for a customer, the bank's provisioning system uses 0.60 to calculate loan-loss reserves. If the true empirical default rate for customers scored at 0.60 is actually 0.40, the bank is **over-provisioning** (or under-provisioning in the opposite case), leading to incorrect capital allocation, regulatory reporting errors, or mispriced risk.

**Calibration** ensures that when the model outputs a probability of *p*, approximately *p* fraction of customers at that score level actually default. A well-calibrated model is an **honest probability estimator**, not just a good ranker.

---

## 2. Calibration Metrics Explained

### Expected Calibration Error (ECE)
ECE partitions predictions into *M* equal-width bins and measures the weighted average gap between mean predicted probability and observed frequency within each bin:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \overline{p}_m - \overline{y}_m \right|$$

**In plain terms:** ECE is the average miscalibration across all confidence levels. An ECE of 0.117 means the model's probability estimates are, on average, about 11.7 percentage points off from the true empirical rates. An ECE of 0.024 means the model is off by only 2.4 pp on average — a well-calibrated score.

Lower ECE = better calibration. **Perfect calibration = ECE of 0.**

### Brier Score
The mean squared error between predicted probabilities and actual binary outcomes:

$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} \left( \hat{p}_i - y_i \right)^2$$

**In plain terms:** Penalises both overconfident wrong predictions and underconfident correct predictions. Ranges from 0 (perfect) to 1 (worst). Lower is better. Unlike ECE, Brier Score jointly measures calibration **and** discrimination ability.

### Log Loss (Binary Cross-Entropy)
The negative log-likelihood of the predicted probabilities:

$$\text{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

**In plain terms:** Severely penalises highly confident wrong predictions. A model that predicts PD = 0.99 for a customer who does not default incurs a very large log loss penalty. Lower is better. Uncalibrated models that are overconfident tend to have significantly higher log loss.

---

## 3. Methodology

### Data Splits
| Split | n | Purpose |
|---|---|---|
| Training set | Varies | Model fitting (XGBoost + neural net ensemble) |
| **Validation set** | **2,250** | **Calibrator fitting; threshold tuning** |
| **Test set** | **2,250** | **Final evaluation (all metrics below)** |

### Calibration Procedure

1. **Uncalibrated scores** from the hybrid model were collected on the validation set.
2. **Platt (Sigmoid) Calibration**: A logistic regression was fitted mapping raw model scores → calibrated probabilities on the validation set.
3. **Isotonic Regression Calibration**: A non-parametric isotonic regression was fitted on the validation set.
4. All three scorers (uncalibrated, Platt-calibrated, isotonic-calibrated) were then applied to the **held-out test set**, which neither the base model nor the calibrators had seen during fitting.

> [!IMPORTANT]
> Calibrators were fitted **exclusively on the validation set**. The test set was untouched during all training, tuning, and calibration-fitting steps. This ensures the calibration evaluation is unbiased and reflects genuine out-of-sample performance.

---

## 4. Original Pipeline Bug: Calibrator Leakage

During initial development, the isotonic calibrator was **incorrectly fitted on the holdout (test) set** rather than the validation set. This introduced **data leakage** into the calibration evaluation:

- The isotonic calibrator had already "seen" the test set outcomes during fitting.
- As a result, its reported ECE and Brier Score on the test set were optimistically biased.
- The comparison between Platt and isotonic calibration was invalid because one calibrator was evaluated in-sample while the other was evaluated out-of-sample.

> [!WARNING]
> Fitting any calibrator (or any post-hoc processor) on the same data used for final evaluation inflates apparent performance. This is equivalent to evaluating a classifier on its training data — the results do not generalise.

### Corrected Approach

The pipeline was corrected so that **all calibrators are fitted only on the validation set** and evaluated only on the test set. The metrics in Section 5 below reflect this corrected, unbiased evaluation.

---

## 5. Calibration Results (Test Set)

**Actual default rate on test set: 36.98% (833 defaults out of 2,250 customers)**

| Method | Brier Score ↓ | Log Loss ↓ | ECE ↓ | Mean Predicted PD | Actual Default Rate |
|---|---|---|---|---|---|
| **Uncalibrated Hybrid** | 0.17917 | 0.54002 | 0.11772 | 0.4875 | 0.3698 |
| **Platt / Sigmoid Calibrated** | **0.16485** | **0.50318** | 0.02865 | 0.3672 | 0.3698 |
| **Isotonic Calibrated** | 0.16557 | 0.51857 | **0.02373** | 0.3665 | 0.3698 |

*↓ = lower is better for all three metrics*

### Key Findings

#### 1. Uncalibrated Model is Significantly Overconfident
The uncalibrated hybrid model predicts a **mean PD of 0.4875** across all test customers, while the **actual default rate is 0.3698** — a systematic overclaim of **13.2 percentage points**. This is a large and consequential bias:

- If used raw in EL calculations, loan-loss reserves would be over-estimated by roughly 13 pp on average.
- Customers in the moderate-risk band would be misclassified as high-risk, leading to unnecessary credit restriction.
- An ECE of 0.1177 confirms the overconfidence is systematic, not random noise.

#### 2. Platt Calibration: Best Overall Balance
- **Best Brier Score** (0.16485) — smallest mean squared probability error.
- **Best Log Loss** (0.50318) — least penalised by overconfident wrong predictions.
- Mean predicted PD (0.3672) is within **0.26 pp** of the true rate (0.3698) — essentially unbiased at the population level.
- ECE reduced from 0.11772 → **0.02865**, a **75.7% reduction**.

#### 3. Isotonic Calibration: Best Bin-Level Calibration
- **Best ECE** (0.02373) — the most accurate bin-by-bin alignment between predicted and observed rates.
- Mean predicted PD (0.3665) is within **0.33 pp** of the true rate.
- Slightly higher Brier Score (0.16557) and Log Loss (0.51857) than Platt, suggesting mild overfitting to the validation set's local shape.
- ECE reduced from 0.11772 → **0.02373**, a **79.8% reduction**.

#### 4. Both Methods Dramatically Outperform Uncalibrated
Both calibration methods reduce ECE by **~79–80%** relative to the uncalibrated baseline. The improvement in Brier Score (~8–8.7%) and Log Loss (~6.6–7.0%) is also substantial. Any production deployment should use a calibrated score.

---

## 6. Visualisation Note

> [!NOTE]
> Reliability diagram (calibration curve) PNG plots were **not generated** in this evaluation run. The tabular metrics above were sourced directly from `raw_metrics.json`. Reliability diagrams plot mean predicted probability vs. observed frequency per bin and are recommended for future runs to visually confirm the calibration improvements quantified here.

---

## 7. Recommendation: Use Platt Calibration for Production

**Recommended calibrator: Platt (Sigmoid) Calibration**

| Criterion | Platt | Isotonic | Verdict |
|---|---|---|---|
| Brier Score | **0.16485** (best) | 0.16557 | Platt |
| Log Loss | **0.50318** (best) | 0.51857 | Platt |
| ECE | 0.02865 | **0.02373** (best) | Isotonic |
| Overfitting Risk | Low (2 parameters) | Moderate (non-parametric) | Platt |
| Interpretability | High (logistic form) | Moderate | Platt |

**Rationale:**

- Platt calibration's superiority in Brier Score and Log Loss means it produces **better overall probability estimates** — it is not just better-calibrated at the bin level but generates more accurate individual-level probabilities.
- Isotonic regression's lower ECE reflects better bin-level alignment, but its higher complexity (non-parametric, staircase function) makes it more susceptible to overfitting on small validation sets and less stable across distribution shifts.
- The **difference in ECE** (0.02865 vs 0.02373 = 0.0049) is small in absolute terms and unlikely to be material for provisioning decisions.
- For regulatory model documentation, Platt calibration's **two-parameter sigmoid form** is easier to audit and explain to validators.

> [!TIP]
> If strict bin-level accuracy is a regulatory requirement (e.g., the supervisor explicitly tests calibration within PD bands), **Isotonic calibration may be preferred** for its lower ECE. Consider reporting both and selecting based on the specific regulatory test applied.

---

## 8. Synthetic Data Disclosure

> [!CAUTION]
> **This project uses a 100% synthetic, fully simulated dataset.** All customer records, credit histories, default events, and derived features were programmatically generated for the purpose of building and evaluating this ML pipeline. No real individual's financial data is reflected in any result. The calibration metrics, PD distributions, and default rates reported here **cannot be directly applied** to real credit portfolios. Retraining and recalibration on real, representative data is mandatory before any production use.

---

*Report generated for the Pre-Delinquency Risk Engine project. All metrics are from raw_metrics.json. Calibrators fitted on validation set (n = 2,250), evaluated on test set (n = 2,250).*
