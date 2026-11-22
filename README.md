# EarlyShield — Pre-Delinquency Risk Engine

**A hybrid machine learning system for predicting customer delinquency risk from historical financial behavior.**

EarlyShield combines gradient-boosted decision trees, sequential deep learning, behavioral feature engineering, and explainability to support early identification of potential financial stress.

> **Project scope:** A prototype built using synthetic customer financial data for experimentation and demonstration. Not validated for real-world lending decisions.

## Project Overview

Traditional credit risk models often rely on static customer information. EarlyShield explores how historical financial behavior and month-to-month changes can be incorporated into risk prediction.

The system estimates:

- **Probability of Delinquency (PD):** Estimated likelihood of delinquency
- **Risk Category:** Low, medium, or high
- **Expected Loss:** An illustrative estimate based on PD, LGD, and EAD
- **Risk Drivers:** Feature-level explanations using SHAP

## System Architecture

```text
Historical Customer Financial Data
                |
                v
       Feature Engineering
                |
        +-------+-------+
        |               |
        v               v
   Tree Models      LSTM Model
   - XGBoost        - Temporal Inputs
   - LightGBM       - Sequential Patterns
   - CatBoost
        |               |
        +-------+-------+
                |
                v
         Hybrid Ensemble
                |
                v
      Probability Calibration
                |
                v
       Delinquency Risk Score
                |
        +-------+-------+
        |               |
        v               v
   Risk Category    Expected Loss
                |
                v
       Streamlit Dashboard
```

## Machine Learning Models

| Model | Purpose |
|---|---|
| XGBoost | Gradient-boosted tree classification |
| LightGBM | Efficient gradient-boosted tree classification |
| CatBoost | Gradient-boosted tree classification |
| LSTM (PyTorch) | Learning patterns from historical customer sequences |

The project includes saved model artifacts and a hybrid prediction pipeline combining tree-based and sequential predictions.

## Behavioral Feature Engineering

The feature engineering module contains transformations including:

- Lag features
- Rolling statistics
- Month-to-month changes
- Utilization trends
- Financial burden ratios
- Persistence indicators

Examples include credit utilization changes, repayment burden, and historical financial stress patterns.

The engineered feature set is broader than the subset of features used by the saved trained models.

## Prediction Workflow

1. Load historical customer financial records.
2. Incorporate current financial inputs.
3. Generate behavioral features.
4. Prepare tabular and sequential model inputs.
5. Generate predictions from tree models and LSTM.
6. Combine model predictions using a hybrid ensemble.
7. Apply probability calibration.
8. Calculate risk category and expected loss.

## Explainability

The project includes SHAP-based analysis to help interpret model behavior, including feature importance and waterfall visualizations.

The current dashboard uses pre-generated explanations. Generating explanations dynamically for each selected customer is a future improvement.

## Streamlit Dashboard

The dashboard code supports:

- Customer selection
- Historical financial behavior visualization
- Financial input controls
- Delinquency risk estimation
- Risk categorization
- Expected loss calculation
- Display of SHAP explanations

## Repository Structure

```text
earlysheild/
├── app.py
├── feature_engineering.py
├── predictor.py
├── model_loader.py
├── risk_engine.py
├── inference.py
├── test_ui.py
├── notebook/
│   └── model_training.ipynb
├── content/
│   └── models/
├── sample_data/
├── financial_stress_full_bank_grade_dataset.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/divyamjain015/earlysheild.git
cd earlysheild
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

**Note:** The project requires compatible Python and library versions, along with its saved model artifacts. Installation and end-to-end execution are being verified.

## Risk Metrics

### Probability of Delinquency

The model generates a probability score between 0 and 1.

### Expected Loss

\[
EL = PD \times LGD \times EAD
\]

Where:

- **PD:** Probability of Delinquency
- **LGD:** Loss Given Default
- **EAD:** Exposure at Default

Expected loss values in this prototype are illustrative and depend on the selected assumptions.

## Evaluation Status

The repository contains a training notebook and model artifacts. A reproducible evaluation using a clean, independent test set is planned.

Final ROC-AUC and performance optimization claims will be documented once they have been independently verified.

## Tech Stack

**Language:** Python

**Machine Learning:** XGBoost, LightGBM, CatBoost, Scikit-learn

**Deep Learning:** PyTorch

**Data Processing:** Pandas, NumPy

**Explainability:** SHAP

**Dashboard:** Streamlit

**Visualization:** Matplotlib

## Future Improvements

- Independent model evaluation and benchmark reporting
- Customer-specific SHAP explanations
- Automated tests and continuous integration
- Model monitoring and drift detection
- API-based inference
- Deployment and containerization

## Acknowledgments

Built upon the original pre-delinquency risk engine codebase. Original commit attribution is preserved in the Git history. Subsequent improvements and validation are tracked through new commits.

## Disclaimer

EarlyShield is an educational machine learning prototype using synthetic financial data. Its predictions are not intended for actual credit approval, lending, or financial decision-making.
