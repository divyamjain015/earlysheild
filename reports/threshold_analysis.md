# Threshold Analysis Report
## Pre-Delinquency Risk Engine — Hybrid Model

> [!CAUTION]
> **Synthetic Data Disclosure**: All results in this report are based on a **100% simulated, synthetic dataset**. No real customer or banking data was used at any stage of this project. This report is for research and demonstration purposes only.

---

## 1. Overview

In binary classification, the default decision threshold of **τ = 0.50** is mathematically arbitrary — it simply means "predict the positive class whenever the model's estimated probability exceeds 50%." For many real-world applications, this default is suboptimal, and **credit risk monitoring is a prime example**.

### Why τ = 0.50 is the Wrong Starting Point for Pre-Delinquency Detection

A pre-delinquency model predicts which customers are at risk of defaulting **before** delinquency occurs. The two error types carry fundamentally different business costs:

- **False Positive (FP):** A customer is flagged as high-risk but is actually fine. Cost = unnecessary intervention (outbound call, restructuring offer, operational overhead). This is recoverable and relatively cheap.
- **False Negative (FN):** A customer who will default is not flagged. Cost = missed opportunity to intervene, potentially leading to write-off, credit loss, or regulatory exposure. This is expensive and often unrecoverable.

Because of this **asymmetric cost structure**, the optimal threshold should be tuned to reflect the business cost ratio — not defaulted to 0.50. Moving the threshold lower (e.g., τ = 0.25) increases **recall** (fewer missed defaults) at the expense of **precision** (more unnecessary interventions). Moving it higher (e.g., τ = 0.55) balances precision and recall more evenly.

---

## 2. Threshold Selection Methodology

### Evaluation Protocol
Thresholds were selected using the **validation set only** (n = 2,250 customers, held out during training). Final metrics reported here are from the **test set** (n = 2,250 customers, unseen during all threshold tuning). This prevents threshold-search overfitting to the test set.

### Two Optimization Criteria

**F1-Optimal Threshold**
The threshold that maximizes the harmonic mean of precision and recall on the validation set.
- Validation F1-optimal threshold: **τ* = 0.55** (val F1 = 0.6745)
- Treats FP and FN errors equally in terms of model performance.

**Cost-Weighted Threshold**
A business-driven criterion that minimizes a weighted loss function:

```
Cost = 5 × FN + 1 × FP
```

This reflects a **5:1 cost ratio** — a missed default (FN) is assumed to be five times more costly than an unnecessary intervention (FP). This ratio is a commonly cited heuristic in credit risk literature and aligns with regulatory emphasis on loss prevention.

- Validation cost-optimal threshold: **τ* = 0.25**
- Aggressively prioritises recall (catching near-all defaults) even at the expense of many false alarms.

### Model Discriminative Power (Threshold-Independent)
- **ROC-AUC = 0.8134** — the model correctly ranks a random defaulter above a random non-defaulter ~81% of the time.
- **PR-AUC = 0.7209** — solid performance on the precision-recall curve, relevant for the imbalanced positive class.

> [!NOTE]
> ROC-AUC and PR-AUC are properties of the full score distribution and do **not** change when the decision threshold is changed. Only accuracy, precision, recall, F1, specificity, and confusion matrix counts depend on τ.

---

## 3. Business Impact Analysis

### Cost Interpretation in a Banking Context

| Error Type | Model Label | Reality | Business Consequence |
|---|---|---|---|
| False Positive (FP) | High-risk | Low-risk | Unnecessary outreach / restructuring offer — resource cost only |
| False Negative (FN) | Low-risk | High-risk (defaults) | Missed intervention → potential credit loss, write-off, regulatory flag |

With a **5:1 FN:FP cost ratio**, every missed defaulter costs the equivalent of five unnecessary interventions. This strongly favours **high-recall strategies** in the absence of budget constraints.

### Impact Comparison on Test Set (n = 2,250)

| Threshold | Missed Defaults (FN) | Unnecessary Interventions (FP) | Weighted Cost (5×FN + 1×FP) |
|---|---|---|---|
| τ = 0.25 (cost-optimal) | **54** | 855 | **1,125** |
| τ = 0.50 (default) | 184 | 382 | 1,302 |
| τ = 0.55 (F1-optimal) | 188 | 381 | 1,321 |

The cost-optimal threshold at τ = 0.25 produces the **lowest total weighted cost** (1,125 vs 1,302 for default τ = 0.50), confirming it is the right operational choice when FN costs dominate.

---

## 4. Full Threshold Comparison Table

**All metrics measured on the final test set (n = 2,250 customers)**

| Metric | τ = 0.25 (Cost-Optimal) | τ = 0.50 (Default) | τ = 0.55 (F1-Optimal) |
|---|---|---|---|
| **Accuracy** | 0.5960 | 0.7484 | 0.7471 |
| **Precision** | 0.4764 | 0.6291 | 0.6283 |
| **Recall** | **0.9351** | 0.7788 | 0.7740 |
| **F1-Score** | 0.6312 | **0.6960** | 0.6936 |
| **Specificity** | 0.3970 | 0.7306 | 0.7313 |
| **Balanced Accuracy** | 0.6661 | **0.7547** | 0.7527 |
| **MCC** | 0.3594 | **0.4936** | 0.4898 |
| **FPR** | 0.6030 | 0.2694 | 0.2687 |
| **FNR** | **0.0649** | 0.2212 | 0.2260 |
| **True Negatives (TN)** | 563 | 1,036 | 1,037 |
| **False Positives (FP)** | 855 | 382 | 381 |
| **False Negatives (FN)** | **54** | 184 | 188 |
| **True Positives (TP)** | **778** | 648 | 644 |
| **ROC-AUC** | 0.8134 | 0.8134 | 0.8134 |
| **PR-AUC** | 0.7209 | 0.7209 | 0.7209 |

> [!NOTE]
> ROC-AUC and PR-AUC are identical across all thresholds — they reflect the model's inherent discriminative capacity, not any specific operating point.

### Key Observations

- **τ = 0.25** captures **93.5% of all true defaults** (recall = 0.9351) but generates substantial false alarms (FPR = 0.603). It has the lowest accuracy (0.596) because it over-flags non-defaulters, but it delivers the **lowest weighted cost** under a 5:1 FN:FP regime.
- **τ = 0.50** is a balanced operating point with the highest balanced accuracy (0.7547) and MCC (0.4936). It is the best general-purpose threshold but underperforms in asymmetric-cost environments.
- **τ = 0.55** is marginally superior to τ = 0.50 in specificity (0.7313 vs 0.7306) and FP count (381 vs 382), but practically equivalent. Its marginal benefit does not justify a change from the default in precision-sensitive settings unless F1 on validation is the governing criterion.

---

## 5. Operational Recommendation

### Scenario A: High-Recall Credit Risk Monitoring → **Use τ = 0.25**

Recommended when:
- The business objective is to **minimise missed defaults** at all reasonable costs.
- The intervention cost (outbound call, payment restructuring) is low relative to credit loss.
- Regulatory requirements mandate conservative identification of at-risk customers.
- Loss given default (LGD) is high (secured/unsecured personal loans, credit cards).

**Expected test-set performance:** Recall = 93.51%, FNR = 6.49%, FP = 855.

> [!IMPORTANT]
> At τ = 0.25, the false positive rate is 60.3%. Operations teams must be staffed to handle the elevated intervention volume. Without capacity planning, false-alarm fatigue may reduce the effectiveness of outreach.

### Scenario B: Precision-Sensitive Alert Systems → **Use τ = 0.55**

Recommended when:
- Intervention resources are constrained and **alert precision matters** (e.g., manual relationship-manager outreach for high-value customers).
- The cost of a false alarm approaches the cost of a missed default.
- The model feeds a downstream human review workflow with limited throughput.

**Expected test-set performance:** Precision = 62.83%, F1 = 0.6936, FP = 381.

### Threshold Selection Should Be Revisited Periodically

Thresholds should be re-evaluated whenever:
- Portfolio composition changes (e.g., economic cycle, product mix shift).
- The empirical default rate drifts from the training distribution (3698% test-set rate used here).
- Intervention costs or recovery rates are updated by the finance/risk team.

---

## 6. Synthetic Data Disclosure

> [!CAUTION]
> **This project uses a 100% synthetic, fully simulated dataset.** All customer records, credit histories, default events, and derived features were programmatically generated for the purpose of building and evaluating this ML pipeline. No real individual's financial data is reflected in any result. The model, threshold values, and performance metrics reported here **cannot be directly applied** to real credit portfolios without retraining on verified, representative data.

---

*Report generated for the Pre-Delinquency Risk Engine project. All metrics are from raw_metrics.json evaluated on the held-out test set (n = 2,250).*
