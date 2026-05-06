import streamlit as st
import pandas as pd

from model_loader import load_models
from feature_engineering import engineer_features
from predictor import prepare_inputs, predict_pd
from risk_engine import expected_loss, risk_bucket


# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def get_models():
    return load_models("content/models")


models = get_models()


# ==========================================
# LOAD HISTORICAL DATA
# ==========================================
@st.cache_data
def load_history():
    df = pd.read_csv(
        "financial_stress_full_bank_grade_dataset.csv"
    )

    df["month"] = pd.to_datetime(df["month"])

    return df


history_df = load_history()


# ==========================================
# STREAMLIT UI
# ==========================================
st.title("💳 Pre-Delinquency Risk Engine")


# ==========================================
# CUSTOMER SELECTION
# ==========================================
st.sidebar.header("Customer Selection")

customer_id = st.sidebar.selectbox(
    "Select Customer",
    sorted(history_df["customer_id"].unique())
)


# ==========================================
# CUSTOMER HISTORY
# ==========================================
customer_history = history_df[
    history_df["customer_id"] == customer_id
].sort_values("month")


# ==========================================
# SHOW CUSTOMER HISTORY
# ==========================================
st.subheader("📜 Customer Historical Data")

st.dataframe(
    customer_history.tail(6)
)


# ==========================================
# CURRENT MONTH INPUTS
# ==========================================
st.sidebar.header(
    "Current Month Financial Details"
)

month = st.sidebar.text_input(
    "Current Month",
    "2024-07"
)

customer_segment = st.sidebar.selectbox(
    "Customer Segment",
    ["salaried", "self_employed"]
)

region_tier = st.sidebar.selectbox(
    "Region Tier",
    ["tier_1", "tier_2", "tier_3"]
)

product_type = st.sidebar.selectbox(
    "Product Type",
    ["personal_loan", "credit_card"]
)

active_products = st.sidebar.number_input(
    "Active Products Count",
    min_value=1,
    max_value=20,
    value=2
)

utilization = st.sidebar.slider(
    "Credit Card Utilization",
    min_value=0.0,
    max_value=1.0,
    value=0.40
)

emi = st.sidebar.number_input(
    "Primary EMI Amount",
    min_value=0,
    max_value=500000,
    value=18000
)

other_obligations = st.sidebar.number_input(
    "Other Monthly Obligations",
    min_value=0,
    max_value=500000,
    value=10000
)

income = st.sidebar.number_input(
    "Monthly Income",
    min_value=1,
    max_value=1000000,
    value=60000
)

salary_delay = st.sidebar.number_input(
    "Salary Delay Days",
    min_value=0,
    max_value=31,
    value=2
)

weekly_balance_change = st.sidebar.slider(
    "Weekly Balance Change %",
    min_value=-100,
    max_value=100,
    value=-2
)

atm_withdrawal = st.sidebar.number_input(
    "ATM Withdrawal Amount",
    min_value=0,
    max_value=500000,
    value=10000
)


# ==========================================
# COMPUTED FEATURES
# ==========================================
total_monthly_obligation = (
    emi + other_obligations
)

emi_to_income_ratio = (
    emi / max(income, 1)
)

obligation_to_income_ratio = (
    total_monthly_obligation / max(income, 1)
)


# ==========================================
# SHOW FINANCIAL SUMMARY
# ==========================================
st.sidebar.subheader("Computed Financial Ratios")

st.sidebar.write(
    f"EMI-to-Income Ratio: "
    f"{emi_to_income_ratio:.2f}"
)

st.sidebar.write(
    f"Obligation-to-Income Ratio: "
    f"{obligation_to_income_ratio:.2f}"
)

st.sidebar.write(
    f"Total Monthly Obligation: "
    f"₹{total_monthly_obligation}"
)


# ==========================================
# PREDICT BUTTON
# ==========================================
if st.button("Predict Risk"):

    # ==========================================
    # CREATE NEW CURRENT MONTH ROW
    # ==========================================
    new_row = pd.DataFrame([{
        "customer_id": customer_id,
        "month": pd.to_datetime(month),

        "customer_segment": customer_segment,
        "region_tier": region_tier,
        "product_type": product_type,

        "active_products_count": active_products,

        "credit_card_utilization": utilization,

        "total_monthly_obligation":
            total_monthly_obligation,

        "emi_amount": emi,

        "days_to_emi": 15,

        "emi_to_income_ratio":
            emi_to_income_ratio,

        "salary_delay_days":
            salary_delay,

        "weekly_balance_change_pct":
            weekly_balance_change,

        "atm_withdrawal_amount":
            atm_withdrawal,

        "monthly_income":
            income
    }])


    # ==========================================
    # APPEND NEW ROW TO HISTORY
    # ==========================================
    customer_full_history = pd.concat(
        [customer_history, new_row],
        ignore_index=True
    )

    customer_full_history = (
        customer_full_history
        .sort_values("month")
    )


    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================
    df_eng = engineer_features(
        customer_full_history
    )


    # ==========================================
    # ONE HOT ENCODING
    # ==========================================
    cat_cols = df_eng.select_dtypes(
        include=["object"]
    ).columns.tolist()

    df_eng = pd.get_dummies(
        df_eng,
        columns=cat_cols,
        drop_first=True
    )


    # ==========================================
    # PREPARE MODEL INPUTS
    # ==========================================
    tree_input, lstm_input = prepare_inputs(
        models,
        df_eng
    )


    # ==========================================
    # PREDICT PD
    # ==========================================
    pd_score = predict_pd(
        models,
        tree_input,
        lstm_input
    )


    # ==========================================
    # EXPECTED LOSS
    # ==========================================
    last_row = customer_full_history.iloc[-1]

    salary_flag = int(
        last_row["salary_delay_days"] > 5
    )

    util_flag = int(
        last_row["credit_card_utilization"] > 0.75
    )

    el, lgd, ead = expected_loss(
        models,
        pd_score,

        last_row["emi_amount"],

        last_row[
            "credit_card_utilization"
        ],

        last_row["monthly_income"],

        salary_flag,
        util_flag
    )


    # ==========================================
    # RISK BUCKET
    # ==========================================
    bucket = risk_bucket(pd_score)


    # ==========================================
    # RESULTS
    # ==========================================
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted PD",
            f"{pd_score:.4f}"
        )

    with col2:
        st.metric(
            "Risk Bucket",
            bucket
        )

    with col3:
        st.metric(
            "Expected Loss (₹)",
            round(el, 2)
        )

    st.progress(float(pd_score))


    # ==========================================
    # INTERPRETATION
    # ==========================================
    st.subheader("🧠 Risk Interpretation")

    if pd_score < 0.20:

        st.success(
            "Low delinquency probability detected."
        )

    elif pd_score < 0.50:

        st.warning(
            "Moderate delinquency risk detected."
        )

    else:

        st.error(
            "High delinquency risk detected."
        )


    # ==========================================
    # SHOW FULL SEQUENCE USED
    # ==========================================
    st.subheader(
        "📈 Final Customer Sequence Used"
    )

    st.dataframe(
        customer_full_history.tail(12)
    )


    # ==========================================
    # IMPORTANT RISK FACTORS
    # ==========================================
    st.subheader("⚠ Key Financial Indicators")

    risk_df = pd.DataFrame({
        "Metric": [
            "Credit Utilization",
            "EMI Amount",
            "Other Obligations",
            "Total Obligation",
            "Monthly Income",
            "EMI-to-Income Ratio",
            "Obligation-to-Income Ratio",
            "Salary Delay Days"
        ],

        "Value": [
            utilization,
            emi,
            other_obligations,
            total_monthly_obligation,
            income,
            round(
                emi_to_income_ratio,
                3
            ),
            round(
                obligation_to_income_ratio,
                3
            ),
            salary_delay
        ]
    })

    st.dataframe(risk_df)


    # ==========================================
    # SHAP PLOTS
    # ==========================================
    st.subheader(
        "🔍 Model Explainability"
    )

    try:

        st.image(
            "shap_plots/shap_waterfall.png",
            caption="SHAP Waterfall Plot"
        )

        st.image(
            "shap_plots/shap_top_features.png",
            caption="Top SHAP Features"
        )

    except:

        st.warning(
            "SHAP plots not found. "
            "Run inference.py first."
        )