💳 Pre-Delinquency Risk Engine
A hybrid AI-powered banking risk analytics system that predicts the probability of customer delinquency using:
•	Tree-based Machine Learning Models
•	Sequential Deep Learning (LSTM)
•	Behavioral Feature Engineering
•	SHAP Explainability
•	Streamlit Dashboard
The system analyzes historical customer financial behavior and estimates:
•	Probability of Delinquency (PD)
•	Risk Bucket
•	Expected Loss
•	Key Risk Drivers
________________________________________
🚀 Project Overview
Traditional credit scoring systems often rely only on static customer snapshots.
This project improves upon that by using:
•	historical behavioral sequences,
•	financial stress trends,
•	repayment patterns,
•	utilization deterioration,
•	and temporal learning using LSTM.
The final system behaves closer to real-world banking risk engines.
________________________________________


🏗 Architecture
Historical Customer Dataset
            ↓
Feature Engineering
            ↓
Tree Models + LSTM
            ↓
Hybrid Ensemble
            ↓
Probability Calibration
            ↓
PD Prediction
            ↓
Expected Loss Estimation
            ↓
SHAP Explainability
            ↓
Streamlit Dashboard
________________________________________
🧠 Models Used
Tree Models
•	XGBoost
•	LightGBM
•	CatBoost
These models learn:
•	nonlinear patterns,
•	threshold-based behavior,
•	tabular financial relationships.

________________________________________
LSTM (Long Short-Term Memory)
The LSTM model learns:
•	sequential customer behavior,
•	month-wise financial deterioration,
•	temporal repayment patterns,
•	utilization trends,
•	behavioral evolution.
Example:
Jan → Feb → Mar → Apr
0.20 → 0.35 → 0.60 → 0.90 utilization
The model detects worsening financial stress over time.
________________________________________
📊 Key Features Used
Raw Features
•	Credit Card Utilization
•	EMI Amount
•	Monthly Income
•	Salary Delay Days
•	Weekly Balance Change
•	ATM Withdrawal Amount
•	Total Monthly Obligation
•	Active Products Count
________________________________________
Engineered Features
•	Lag Features
•	Rolling Means
•	Utilization Trends
•	Delta Features
•	Persistence Indicators
•	Ratio Features

Examples:
credit_card_utilization_lag1
credit_card_utilization_delta_1
emi_to_income_ratio_mean_3m
utilization_high_persist_3m
________________________________________
📈 Financial Risk Metrics
EMI-to-Income Ratio
EMI Amount / Monthly Income
Measures customer repayment burden.
________________________________________
Obligation-to-Income Ratio
Total Monthly Obligation / Monthly Income
Captures overall debt stress.
________________________________________
⚠ Risk Prediction Workflow
Step 1
Load historical customer financial records.
Step 2
Append current month financial behavior.
Step 3
Generate engineered behavioral features.
Step 4
Prepare:
•	Tree Model Inputs
•	LSTM Sequential Inputs
Step 5
Generate predictions from:
•	XGBoost
•	LightGBM
•	CatBoost
•	LSTM
Step 6
Blend predictions using hybrid ensemble.
Step 7
Calibrate final probabilities.
Step 8
Generate:
•	Probability of Delinquency
•	Risk Bucket
•	Expected Loss
________________________________________
🔍 Explainability
The project uses SHAP (SHapley Additive exPlanations) for model explainability.
Generated visualizations:
•	SHAP Waterfall Plot
•	Top Feature Importance Plot
These plots explain:
•	why the model predicted risk,
•	which features increased risk,
•	which features reduced risk.
________________________________________
🖥 Streamlit Dashboard Features
The frontend dashboard provides:
•	Customer selection
•	Historical customer sequence visualization
•	Current month financial input
•	Dynamic risk prediction
•	Expected loss estimation
•	Risk interpretation
•	SHAP explainability display
________________________________________
📂 Project Structure
PD_Model/
│
├── content/
│   └── models/
│       ├── xgb_model.pkl
│       ├── lgb_model.pkl
│       ├── cat_model.cbm
│       ├── lstm_model.pt
│       └── calibrator.pkl
│
├── shap_plots/
│   ├── shap_waterfall.png
│   └── shap_top_features.png
│
├── financial_stress_full_bank_grade_dataset.csv
│
├── app.py
├── inference.py
├── predictor.py
├── feature_engineering.py
├── model_loader.py
├── risk_engine.py
├── requirements.txt
└── README.md
________________________________________
⚙ Installation
1. Clone Repository
git clone <repository_url>
2. Move Into Project
cd PD_Model
3. Create Virtual Environment
Windows
python -m venv venv
Activate:
.\venv\Scripts\activate
________________________________________
📦 Install Dependencies
pip install -r requirements.txt
________________________________________
▶ Running the Project
Generate SHAP Plots
python inference.py
This generates:
shap_plots/shap_waterfall.png
shap_plots/shap_top_features.png
________________________________________

Launch Streamlit Dashboard
streamlit run app.py
or:
python -m streamlit run app.py
Open browser:
http://localhost:8501
________________________________________
📊 Risk Buckets
PD Range	Risk Level
0.00 – 0.20	LOW
0.20 – 0.50	MEDIUM
> 0.50	HIGH
________________________________________
💰 Expected Loss Formula
Expected Loss = PD × LGD × EAD
Where:
•	PD = Probability of Delinquency
•	LGD = Loss Given Default
•	EAD = Exposure At Default
________________________________________
🧪 Model Validation
The project validates predictions using:
•	behavioral sanity testing,
•	monotonicity testing,
•	trend-based testing,
•	SHAP validation,
•	financial logic consistency.
Example:
•	increasing utilization should generally increase risk,
•	worsening salary delays should increase PD,
•	stable customers should receive lower risk scores.
________________________________________
🛠 Tech Stack
Backend / ML
•	Python
•	Pandas
•	NumPy
•	Scikit-learn
•	XGBoost
•	LightGBM
•	CatBoost
•	PyTorch
•	SHAP
________________________________________
Frontend
•	Streamlit
________________________________________
Visualization
•	Matplotlib
•	SHAP
________________________________________
🔮 Future Improvements
Potential production-level extensions:
•	Dynamic SHAP generation
•	Real-time streaming predictions
•	Database integration
•	Model monitoring and drift detection
•	FastAPI deployment
•	Docker containerization
•	Cloud deployment (AWS/GCP/Azure)
•	Automated retraining pipelines
•	Real-time transaction analysis
________________________________________
📌 Key Highlights
✅ Hybrid Ensemble Architecture
✅ Sequential Behavioral Modeling
✅ Banking Risk Analytics
✅ SHAP Explainability
✅ Streamlit Interactive Dashboard
✅ Expected Loss Estimation
✅ Financial Feature Engineering
✅ Temporal Customer Analysis
________________________________________
👨‍💻 Author
Developed as an AI-powered banking risk analytics and pre-delinquency prediction system.
