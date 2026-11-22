# Risk Engine Validation Report
## Pre-Delinquency Risk Engine — Expected Loss & Functional Test Validation

> [!NOTE]
> All test results reported here were obtained on a **100% synthetic dataset**. No real customer data, real loan records, or live financial data were used in any test execution.

---

## 1. Expected Loss Formula & Component Definitions

The risk engine computes **Expected Loss (EL)** using the standard credit risk decomposition:

$$\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$$

| Component | Definition | Source / Method | Range / Values |
|---|---|---|---|
| **PD** — Probability of Default | Likelihood that a customer will become delinquent within the forecast horizon | Output of the **hybrid ML model** (LightGBM tree ensemble + LSTM sequence model) | \[0.01, 0.95\] — hard floor/cap enforced |
| **LGD** — Loss Given Default | Fraction of exposure expected to be unrecoverable upon default | **Rule-based** assignment, NOT empirically estimated from observed recovery data | 0.40 (secured loans) · 0.70 (unsecured loans) |
| **EAD** — Exposure at Default | Total outstanding credit exposure at the time of default | Stated `loan_outstanding_amount` from the customer record | Positive currency units (same denomination as EL output) |
| **EL** | Expected monetary loss | Product of above three components | Currency units (same as EAD) |

### 1.1 PD Clipping
Raw model output probabilities are clipped to the interval **\[0.01, 0.95\]** before computing EL. This prevents degenerate edge cases (zero EL for any active exposure; near-certain default without empirical evidence) and is enforced programmatically (see TC-06).

### 1.2 Risk Bucket Classification

Customers are classified into three risk tiers based on PD:

| Bucket | PD Range | Interpretation |
|---|---|---|
| 🟢 **Low Risk** | PD < 0.33 | Below delinquency alert threshold; standard monitoring |
| 🟡 **Medium Risk** | 0.33 ≤ PD < 0.67 | Elevated concern; flag for proactive outreach |
| 🔴 **High Risk** | PD ≥ 0.67 | Imminent delinquency risk; intervention recommended |

---

## 2. Validation Methodology

Validation was conducted in two phases:

### 2.1 Mathematical Spot Checks
Hand-computed EL values were compared against engine output for known PD/LGD/EAD combinations. This verifies that the formula implementation is arithmetically correct and that rounding or floating-point handling does not introduce material errors.

**Example spot check (TC-01):**
- Inputs: PD = 0.6, LGD = 0.70 (unsecured), EAD = \$10,000
- Expected EL = 0.6 × 0.70 × 10,000 = **\$4,200**
- Engine output verified to match within floating-point tolerance

### 2.2 Boundary & Edge Case Testing
Tests probed the edges of valid input ranges and system state:
- **Floor/cap enforcement** on PD (TC-06)
- **Zero-value inputs** (zero income, zero EMI) to ensure no division-by-zero or silent failure (TC-03)
- **Insufficient history** — customers with fewer than 3 months of records are zero-padded before inference (TC-04)
- **Missing feature columns** — reindexing with fill values ensures alignment between training and inference schemas (TC-05)
- **Negative input values** — testing whether the engine rejects or silently accepts negative income/EMI (TC-10)

### 2.3 Infrastructure & Artifact Tests
Tests verified correct operation of downstream artifacts:
- SHAP plot generation and file persistence (TC-09)
- UI selector compatibility checking (TC-07)
- `model_metrics.json` syntax validation — specifically double-brace `{{}}` Jinja-style errors that cause silent mis-parse (TC-08)

---

## 3. Functional Test Results

All 10 functional tests **PASSED**.

| Test ID | Test Case | Result | Execution Time |
|---|---|---|---|
| **TC-01** | Expected Loss Mathematical Verification | ✅ PASSED | 0.204 ms |
| **TC-02** | Risk Bucket Boundary Classification | ✅ PASSED | 0.003 ms |
| **TC-03** | Zero Income / EMI Boundary in Expected Loss | ✅ PASSED | 0.003 ms |
| **TC-04** | Insufficient History Zero-Padding (< 3 months) | ✅ PASSED | 1072.3 ms |
| **TC-05** | Missing Feature Column Alignment via Reindexing | ✅ PASSED | 0.293 ms |
| **TC-06** | Probability Floor (0.01) and Cap (0.95) Enforcement | ✅ PASSED | 0.035 ms |
| **TC-07** | UI Selector Incompatibility Detection | ✅ PASSED | 6.536 ms |
| **TC-08** | `model_metrics.json` Double-Brace Syntax Error Detection | ✅ PASSED | 0.553 ms |
| **TC-09** | SHAP Plot Generation & Persistence | ✅ PASSED | 0.310 ms |
| **TC-10** | Negative Income / EMI Sanity Guardrail *(Identified Vulnerability)* | ✅ PASSED | 0.007 ms |

> [!WARNING]
> **TC-04** (zero-padding for short-history customers) has an unusually high execution time of **1072.3 ms** relative to all other tests. This latency should be profiled in a production setting — it may reflect LSTM inference on padded sequences or I/O overhead in the test harness.

> [!IMPORTANT]
> **TC-10 — Design Gap Identified:** The test *passed* in the sense that it successfully *detected* the vulnerability: the engine **does not reject negative income or EMI inputs**. The model accepts and processes them without raising an error or guardrail. This is a **design gap, not a test failure** — the test was designed to surface this behaviour, and it succeeded in doing so. See Section 5 for the remediation recommendation.

---

## 4. LGD & EAD Limitation Disclosure

### 4.1 LGD is Rule-Based — Not Empirically Calibrated

The current LGD values (0.40 for secured, 0.70 for unsecured) are **fixed constants** assigned by rule based on loan type. They are **not** derived from historical recovery data or statistical estimation.

In production banking under regulatory frameworks (Basel III/IV, IFRS 9), LGD estimation requires:
- Historical records of observed defaults and subsequent recoveries
- Downturn LGD adjustments (stressed recovery scenarios)
- Collateral valuation models for secured exposure
- Time-to-recovery discounting

**The current LGD values should not be treated as calibrated or regulatory-compliant estimates.** They are placeholder constants suitable for a proof-of-concept system only.

### 4.2 EAD Uses Stated Outstanding Balance — No CCF Adjustment

EAD is sourced directly from the `loan_outstanding_amount` field in the customer record. This is the **stated drawn balance**, with no adjustment for:
- **Credit Conversion Factors (CCFs):** Under the Basel framework, undrawn credit lines must be converted to EAD-equivalent using CCFs. This system does not model off-balance-sheet exposure.
- **Accrued interest or fees** that would increase EAD at the time of default
- **Partial drawdown dynamics** for revolving facilities

For term loans with no undrawn component, the stated balance is a reasonable approximation. For revolving credit or overdraft facilities, it will systematically understate true exposure.

### 4.3 No Maturity or Discounting Applied

The EL calculation is a **point-in-time, undiscounted** estimate. No:
- **Maturity adjustments** (longer-tenor loans carry higher risk of multi-period default)
- **Discount factors** to present-value future expected losses

Applied. In lifetime expected credit loss models (e.g., IFRS 9 Stage 2/3), discounting is mandatory.

---

## 5. Recommendations for Production Hardening

| Priority | Area | Recommendation |
|---|---|---|
| 🔴 **High** | Input Validation (TC-10 gap) | Add a **pre-inference guardrail** that rejects or flags records with negative `income` or `emi_amount`. Raise a validation error or return a sentinel risk bucket rather than silently processing invalid inputs. |
| 🔴 **High** | LGD Calibration | Replace rule-based LGD constants with **empirically estimated values** derived from observed default and recovery data. Engage credit risk modelling / actuarial team for Basel-compliant LGD model development. |
| 🟡 **Medium** | EAD Modelling | Incorporate **Credit Conversion Factors** for revolving/overdraft exposures. Source EAD from a system-of-record that tracks both drawn and undrawn balance. |
| 🟡 **Medium** | TC-04 Latency | Profile and optimize the zero-padding / LSTM inference path for short-history customers. Consider caching padded feature tensors or pre-computing them at ingestion time. |
| 🟡 **Medium** | Lifetime EL | For regulatory reporting purposes, extend to **lifetime EL** with appropriate discount rates and maturity-adjusted PD term structures. |
| 🟢 **Low** | EL Discounting | Apply present-value discounting to EL for forward-looking loss provisioning under IFRS 9 Stage 2/3 requirements. |
| 🟢 **Low** | Monitoring | Implement **population stability index (PSI)** and **feature drift monitoring** to detect dataset shift between training and live inference. |

---

## 6. Synthetic Dataset Disclosure

> [!CAUTION]
> **All validation tests in this report were executed on a 100% synthetic dataset.** The synthetic data was artificially generated to simulate realistic behavioral and financial patterns for pre-delinquency modelling. No real customer records, real loan portfolios, or live financial data were used at any stage of model development, testing, or validation.
>
> Functional test results (PASS/FAIL) verify the **correctness of the engine's logic** given its inputs and configuration. They do **not** constitute evidence that the model's risk predictions are accurate, well-calibrated, or appropriate for real-world deployment.
>
> Before any production use, the system must be:
> - Re-trained and re-validated on observed customer data
> - Reviewed and approved by the institution's model risk management function
> - Assessed for compliance with applicable regulatory requirements (e.g., Basel, IFRS 9, SR 11-7)

---

*Report generated by the Pre-Delinquency Risk Engine validation pipeline.*
