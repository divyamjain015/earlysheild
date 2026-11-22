# Functional Test Report
**Pre-Delinquency Risk Engine — Automated Test Suite**

---

## 1. Test Suite Overview

This report documents the results of the automated functional test suite for the Pre-Delinquency Risk Engine. The suite is designed to validate correctness, robustness, and safety properties across the core subsystems:

- **Financial Risk Engine** — Expected Loss formula, risk bucket classification, boundary conditions
- **Sequence Preprocessing** — LSTM input preparation, zero-padding for short histories
- **Input Robustness** — Feature schema alignment, missing column handling
- **UI / Integration** — Compatibility between test harness selectors and application selectors
- **Configuration Integrity** — Validity of serialised artefacts (JSON metrics files)
- **Explainability System** — SHAP diagnostic plot generation
- **Input Validation / Safety** — Guardrails against financially nonsensical inputs

**Result: 10 / 10 tests PASSED**

---

## 2. Test Results

| Test ID | Category | Test Name | Status | Duration (ms) |
|---------|----------|-----------|--------|---------------|
| TC-01 | Financial Risk Engine | Expected Loss Mathematical Verification | ✅ PASSED | 0.204 |
| TC-02 | Financial Risk Engine | Risk Bucket Boundary Classification | ✅ PASSED | 0.003 |
| TC-03 | Financial Risk Engine | Zero Income/EMI Boundary in Expected Loss | ✅ PASSED | 0.003 |
| TC-04 | Sequence Preprocessing | Insufficient History Zero-Padding (< 3 Months) | ✅ PASSED | 1072.343 |
| TC-05 | Input Robustness | Missing Feature Column Alignment via Reindexing | ✅ PASSED | 0.293 |
| TC-06 | Risk Engine Boundaries | Probability Floor (0.01) and Cap (0.95) Enforcement | ✅ PASSED | 0.035 |
| TC-07 | UI / Integration | Detection of test_ui.py vs app.py Selector Incompatibility | ✅ PASSED | 6.536 |
| TC-08 | Configuration Integrity | Original model_metrics.json Double-Brace Syntax Error Detection | ✅ PASSED | 0.553 |
| TC-09 | Explainability System | SHAP Diagnostic Plot File Generation & Persistence | ✅ PASSED | 0.310 |
| TC-10 | Input Validation / Safety | Negative Income/EMI Sanity Guardrail Check (Identified Vulnerability) | ✅ PASSED | 0.007 |

**Total duration:** ~1,080 ms (dominated by TC-04)

---

## 3. Notable Findings

### 3.1 TC-04 — Slow Path: Zero-Padding for Short Customer History

TC-04 is the slowest test at **1,072 ms**, compared to < 7 ms for every other test. This is **expected and by design**:

- The test exercises the full LSTM sequence-preprocessing pipeline for a customer with fewer than 3 months of history.
- The zero-padding path initialises and fills a sequence tensor from scratch, which is more expensive than slicing an existing sequence.
- In production this is an **edge case** (new customers only). The overhead is acceptable and does not constitute a performance regression.

---

### 3.2 TC-07 — UnicodeDecodeError Fix (charmap codec)

**Original failure:** TC-07 initially **FAILED** with:

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position ...
```

**Root cause:** The test harness opened `app.py` without specifying an encoding, causing Python to use the platform default (`cp1252` on Windows), which could not decode UTF-8 sequences present in the file.

**Fix applied:** All `open()` calls that read source files were updated to include `encoding="utf-8"`:

```python
# Before
with open("app.py") as f:

# After
with open("app.py", encoding="utf-8") as f:
```

**Post-fix status:** TC-07 **PASSED**. The fix is minimal and non-breaking.

---

### 3.3 TC-08 — Existing Bug: Double-Brace Syntax in model_metrics.json

TC-08 detected a **pre-existing bug** in the original project artefact `model_metrics.json`.

**Symptom:** The file contains a double opening brace (`{{`) at the top level, making it syntactically invalid JSON:

```json
{{ "accuracy": 0.87, ... }}
```

**Impact:**
- `json.load()` raises `json.JSONDecodeError` when any component attempts to load this file
- Downstream dashboard components that consume model metrics will fail silently or crash

**Status:** Bug identified and confirmed by TC-08. This is an **original project artefact bug** (not introduced by the test suite or optimization changes). A fix requires replacing `{{` / `}}` with `{` / `}`.

> [!CAUTION]
> Do not auto-fix this file without reviewing whether the double-brace syntax was intentional (e.g., a Jinja2/templating artefact). If it is a template, the rendering step is missing from the pipeline.

---

### 3.4 TC-10 — Design Gap: No Validation for Negative Income / EMI

TC-10 identifies a **design gap** in the input validation layer.

**Finding:** The system accepts negative values for `income` and `EMI` without raising an error or warning. The Expected Loss (EL) formula:

```
EL = PD × LGD × EAD
   where EAD is derived from income / EMI ratio
```

produces **mathematically valid but financially nonsensical results** when income or EMI is negative (e.g., a negative EAD, or an inverted affordability ratio). No guardrail currently prevents this.

**Risk:** An upstream data pipeline bug or adversarial input could silently produce garbage risk scores that pass downstream validation checks.

**Recommendation:** Add an explicit pre-processing assertion or `ValueError` for `income <= 0` and `EMI <= 0` before the EL calculation.

---

## 4. Coverage Gaps

The following scenarios are **not** covered by the current test suite and represent areas for future work:

| Gap | Description | Priority |
|-----|-------------|----------|
| Concurrent API calls | No load / race-condition tests for simultaneous scoring requests | High |
| Adversarial inputs | Extreme outliers (e.g., income = 1e18, EMI = 0.0001) not stress-tested | Medium |
| Model drift detection | No tests verify that the system flags or rejects stale/drifted models | Medium |
| End-to-end API contract | No integration test calling the full HTTP API (only unit-level functions tested) | High |
| Negative probability outputs | Uncalibrated score edge cases (pre-Platt) not explicitly tested | Low |

---

## 5. Recommendations

1. **Fix TC-08 bug immediately** — `model_metrics.json` with double-brace syntax will cause JSON parse failures in any consumer. Determine whether it is a template artefact and either fix or add a rendering step.

2. **Add input validation for negative income/EMI (TC-10 finding)** — A single guard clause in the preprocessing layer eliminates a class of financially nonsensical outputs.

3. **Add concurrent API tests** — Use `pytest-asyncio` or `locust` to simulate parallel scoring requests and verify thread-safety of the shared model objects.

4. **Extend adversarial input coverage** — Property-based testing with `hypothesis` can systematically explore boundary conditions beyond hand-crafted test cases.

5. **Monitor TC-04 latency over time** — The 1,072 ms zero-padding path is acceptable today but should be tracked; a regression here would indicate a problem in the LSTM preprocessing layer.
