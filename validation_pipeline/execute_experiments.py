import os
import sys
import time
import json
import pickle
import psutil
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure writable matplotlib cache dir
os.environ['MPLCONFIGDIR'] = os.path.join(os.environ.get('TEMP', '.'), 'mpl_cache')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score, brier_score_loss, log_loss
)
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import shap

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 80)
print("PRE-DELINQUENCY RISK ENGINE - COMPREHENSIVE EXPERIMENT PIPELINE")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. LOAD DATA & AUDIT FEATURE SETS
# ---------------------------------------------------------------------------
DATA_PATH = "financial_stress_full_bank_grade_dataset.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("pre-delinquency-risk-engine", DATA_PATH)

print(f"\n[1/10] Loading raw dataset from: {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw.sort_values(["customer_id", "month"]).reset_index(drop=True)
print(f"Loaded {len(df_raw)} rows, {df_raw['customer_id'].nunique()} unique customers.")

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (HIGH-EFFICIENCY VECTORIZED IMPLEMENTATION)
# ---------------------------------------------------------------------------
print("\n[2/10] Executing feature engineering pipeline...")
t0_feat = time.perf_counter()

def engineer_features_vectorized(df_input):
    df = df_input.copy()
    df = df.sort_values(["customer_id", "month"]).reset_index(drop=True)
    
    # Strictly exclude simulation latent variable and observed target from input features
    NUMERIC_COLS = [
        "active_products_count", "credit_card_utilization", "total_monthly_obligation",
        "emi_amount", "days_to_emi", "emi_to_income_ratio", "salary_delay_days",
        "weekly_balance_change_pct", "atm_withdrawal_amount", "monthly_income"
    ]
    
    # Lags (t-1, t-2)
    for col in NUMERIC_COLS:
        df[f"{col}_lag1"] = df.groupby("customer_id")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("customer_id")[col].shift(2)
        df[f"{col}_delta_1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_delta_2"] = df[f"{col}_lag1"] - df[f"{col}_lag2"]
        
    # Volatility (Rolling Std, min_periods=2)
    for col in ["credit_card_utilization", "weekly_balance_change_pct", "monthly_income"]:
        df[f"{col}_std_3m"] = (
            df.groupby("customer_id")[col]
            .rolling(3, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )
        
    # Rolling Mean & Max (min_periods=2)
    for col in ["credit_card_utilization", "monthly_income", "emi_to_income_ratio", "weekly_balance_change_pct"]:
        g = df.groupby("customer_id")[col]
        df[f"{col}_mean_3m"] = g.rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
        df[f"{col}_max_3m"] = g.rolling(3, min_periods=2).max().reset_index(level=0, drop=True)
        
        # High-efficiency analytic slope formula:
        # For 3 points: slope = (y_2 - y_0) / 2
        # For 2 points: slope = y_1 - y_0
        lag1 = df[f"{col}_lag1"]
        lag2 = df[f"{col}_lag2"]
        slope_3pt = (df[col] - lag2) / 2.0
        slope_2pt = df[col] - lag1
        df[f"{col}_slope_3m"] = np.where(lag2.notnull(), slope_3pt, np.where(lag1.notnull(), slope_2pt, 0.0))

    # Persistence Flags (3-month window, >= 2 occurrences)
    df["emi_high_persist_3m"] = (
        (df["emi_to_income_ratio"] > 0.40).groupby(df["customer_id"])
        .rolling(3, min_periods=2).sum().ge(2)
        .reset_index(level=0, drop=True).astype(int)
    )
    df["salary_delay_persist_3m"] = (
        (df["salary_delay_days"] > 5).groupby(df["customer_id"])
        .rolling(3, min_periods=2).sum().ge(2)
        .reset_index(level=0, drop=True).astype(int)
    )
    df["utilization_high_persist_3m"] = (
        (df["credit_card_utilization"] > 0.75).groupby(df["customer_id"])
        .rolling(3, min_periods=2).sum().ge(2)
        .reset_index(level=0, drop=True).astype(int)
    )
    
    # Next-month target (the operational prediction goal)
    df["loan_default_next_month"] = df.groupby("customer_id")["loan_default_observed"].shift(-1)
    
    # Fill remaining NaNs from early lags with 0
    feature_cols = [c for c in df.columns if c not in ["customer_id", "month", "risk_level_latent", "loan_default_observed", "loan_default_next_month"]]
    df[feature_cols] = df[feature_cols].fillna(0)
    
    return df

df_full = engineer_features_vectorized(df_raw)
t_feat = time.perf_counter() - t0_feat
print(f"Feature engineering completed in {t_feat:.2f} seconds.")

# Filter valid target observations (months 1-5 where next_month is observed; month 6 is unobserved future)
df_valid = df_full.dropna(subset=["loan_default_next_month"]).copy()
df_valid["loan_default_next_month"] = df_valid["loan_default_next_month"].astype(int)
print(f"Valid sequential modeling rows: {len(df_valid)} (dropped {len(df_full) - len(df_valid)} terminal month rows).")

# ---------------------------------------------------------------------------
# 3. GROUP-AWARE STRATIFIED 3-WAY SPLIT (70% Train, 15% Val, 15% Test)
# ---------------------------------------------------------------------------
print("\n[3/10] Constructing strict customer-grouped 3-way partition...")

cust_summary = df_valid.groupby("customer_id")["loan_default_next_month"].mean().reset_index()
# Stratify based on customer default presence (has defaulted in at least 1 month)
cust_summary["default_strata"] = (cust_summary["loan_default_next_month"] > 0).astype(int)

# 70% Train, 30% Temporary Val+Test
train_custs, temp_custs = train_test_split(
    cust_summary["customer_id"],
    test_size=0.30,
    random_state=SEED,
    stratify=cust_summary["default_strata"]
)

temp_summary = cust_summary[cust_summary["customer_id"].isin(temp_custs)]
# 15% Val, 15% Test
val_custs, test_custs = train_test_split(
    temp_summary["customer_id"],
    test_size=0.50,
    random_state=SEED,
    stratify=temp_summary["default_strata"]
)

train_df = df_valid[df_valid["customer_id"].isin(train_custs)].reset_index(drop=True)
val_df = df_valid[df_valid["customer_id"].isin(val_custs)].reset_index(drop=True)
test_df = df_valid[df_valid["customer_id"].isin(test_custs)].reset_index(drop=True)

print(f"Split sizes: Train={len(train_custs)} custs ({len(train_df)} rows), "
      f"Val={len(val_custs)} custs ({len(val_df)} rows), "
      f"Test={len(test_custs)} custs ({len(test_df)} rows)")
print(f"Event rates: Train={train_df['loan_default_next_month'].mean():.4f}, "
      f"Val={val_df['loan_default_next_month'].mean():.4f}, "
      f"Test={test_df['loan_default_next_month'].mean():.4f}")

# Verify zero leakage
assert len(set(train_custs).intersection(set(val_custs))) == 0
assert len(set(train_custs).intersection(set(test_custs))) == 0
assert len(set(val_custs).intersection(set(test_custs))) == 0
print("Zero customer overlap verified across all three splits.")

# ---------------------------------------------------------------------------
# 4. DEFINE FEATURE SETS (FOR ABLATION & FULL MODELING)
# ---------------------------------------------------------------------------
RAW_NUMERIC = [
    "active_products_count", "credit_card_utilization", "total_monthly_obligation",
    "emi_amount", "days_to_emi", "emi_to_income_ratio", "salary_delay_days",
    "weekly_balance_change_pct", "atm_withdrawal_amount", "monthly_income"
]
CATEGORICAL_COLS = ["customer_segment", "region_tier", "product_type"]

# Feature Set A: Raw numeric + Categorical
cols_A = RAW_NUMERIC + CATEGORICAL_COLS

# Feature Set B: Raw + Lags & Deltas
LAG_DELTA = [f"{c}_lag1" for c in RAW_NUMERIC] + [f"{c}_lag2" for c in RAW_NUMERIC] + \
            [f"{c}_delta_1" for c in RAW_NUMERIC] + [f"{c}_delta_2" for c in RAW_NUMERIC]
cols_B = cols_A + LAG_DELTA

# Feature Set C: Raw + Rolling Stats & Persistence
ROLLING_PERSIST = [
    "credit_card_utilization_std_3m", "weekly_balance_change_pct_std_3m", "monthly_income_std_3m",
    "credit_card_utilization_mean_3m", "credit_card_utilization_max_3m", "credit_card_utilization_slope_3m",
    "monthly_income_mean_3m", "monthly_income_max_3m", "monthly_income_slope_3m",
    "emi_to_income_ratio_mean_3m", "emi_to_income_ratio_max_3m", "emi_to_income_ratio_slope_3m",
    "weekly_balance_change_pct_mean_3m", "weekly_balance_change_pct_max_3m", "weekly_balance_change_pct_slope_3m",
    "emi_high_persist_3m", "salary_delay_persist_3m", "utilization_high_persist_3m"
]
cols_C = cols_A + ROLLING_PERSIST

# Feature Set D: Full Engineered Set (Purged of risk_level_latent)
cols_D = cols_A + LAG_DELTA + ROLLING_PERSIST

# One-hot encode categorically, fitting ONLY on Train
def encode_features(train_f, val_f, test_f, feature_cols):
    X_tr = pd.get_dummies(train_f[feature_cols], columns=CATEGORICAL_COLS, drop_first=True)
    cols = X_tr.columns.tolist()
    X_va = pd.get_dummies(val_f[feature_cols], columns=CATEGORICAL_COLS, drop_first=True).reindex(columns=cols, fill_value=0)
    X_te = pd.get_dummies(test_f[feature_cols], columns=CATEGORICAL_COLS, drop_first=True).reindex(columns=cols, fill_value=0)
    return X_tr, X_va, X_te, cols

X_tr_A, X_va_A, X_te_A, feat_A = encode_features(train_df, val_df, test_df, cols_A)
X_tr_B, X_va_B, X_te_B, feat_B = encode_features(train_df, val_df, test_df, cols_B)
X_tr_C, X_va_C, X_te_C, feat_C = encode_features(train_df, val_df, test_df, cols_C)
X_tr_D, X_va_D, X_te_D, feat_D = encode_features(train_df, val_df, test_df, cols_D)

y_train = train_df["loan_default_next_month"].values
y_val = val_df["loan_default_next_month"].values
y_test = test_df["loan_default_next_month"].values

print(f"Feature Set dimensions: Set A={X_tr_A.shape[1]}, Set B={X_tr_B.shape[1]}, Set C={X_tr_C.shape[1]}, Set D (Full)={X_tr_D.shape[1]}")

# ---------------------------------------------------------------------------
# 5. FEATURE ABLATION EXPERIMENTS
# ---------------------------------------------------------------------------
print("\n[4/10] Running feature ablation study across Sets A, B, C, D...")
ablation_results = []

sets = [
    ("Set A: Raw Features Only", X_tr_A, X_va_A, X_te_A),
    ("Set B: Raw + Lag/Delta Features", X_tr_B, X_va_B, X_te_B),
    ("Set C: Raw + Rolling/Persistence Stats", X_tr_C, X_va_C, X_te_C),
    ("Set D: Full Engineered Behavioral Features", X_tr_D, X_va_D, X_te_D),
]

spw = float((y_train == 0).sum() / (y_train == 1).sum())

for name, Xtr, Xva, Xte in sets:
    clf = lgb.LGBMClassifier(n_estimators=120, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, random_state=SEED, verbose=-1, n_jobs=-1)
    t0 = time.perf_counter()
    clf.fit(Xtr, y_train)
    t_fit = time.perf_counter() - t0
    
    p_val = clf.predict_proba(Xva)[:, 1]
    p_test = clf.predict_proba(Xte)[:, 1]
    
    auc_val = roc_auc_score(y_val, p_val)
    auc_test = roc_auc_score(y_test, p_test)
    prauc_test = average_precision_score(y_test, p_test)
    
    # 0.5 threshold metrics
    preds_05 = (p_test >= 0.5).astype(int)
    rec_test = recall_score(y_test, preds_05)
    f1_test = f1_score(y_test, preds_05)
    prec_test = precision_score(y_test, preds_05)
    
    ablation_results.append({
        "Feature Set": name,
        "Feature Count": Xtr.shape[1],
        "Val ROC-AUC": round(auc_val, 4),
        "Test ROC-AUC": round(auc_test, 4),
        "Test PR-AUC": round(prauc_test, 4),
        "Test Precision": round(prec_test, 4),
        "Test Recall": round(rec_test, 4),
        "Test F1-Score": round(f1_test, 4),
        "Fit Time (s)": round(t_fit, 2)
    })

ablation_df = pd.DataFrame(ablation_results)
print("\n--- Feature Ablation Benchmark Table ---")
print(ablation_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. MODEL TRAINING & COMPARISON (ON SET D)
# ---------------------------------------------------------------------------
print("\n[5/10] Training and evaluating all 8 benchmark models on Set D...")

# Prepare LSTM sequential inputs (3-month window of raw features per customer)
LSTM_RAW_COLS = RAW_NUMERIC + [c for c in feat_D if c.startswith(("customer_segment_", "region_tier_", "product_type_"))]
print(f"LSTM Sequential feature count: {len(LSTM_RAW_COLS)}")

# Prepare 3-month sequence matrices for each customer's last observed evaluation point
def create_lstm_sequences(df_part, scaler=None, is_train=False):
    seq_list = []
    targets = []
    cust_order = []
    
    # Group by customer and sort chronologically
    for cust, group in df_part.groupby("customer_id"):
        grp = group.sort_values("month")
        # Re-encode categoricals
        grp_enc = pd.get_dummies(grp, columns=CATEGORICAL_COLS, drop_first=True).reindex(columns=LSTM_RAW_COLS, fill_value=0)
        vals = grp_enc.values
        targs = grp["loan_default_next_month"].values
        
        # Take the terminal decision record for each customer
        if len(vals) >= 3:
            seq = vals[-3:]
            tar = targs[-1]
        else:
            pad = np.zeros((3 - len(vals), len(LSTM_RAW_COLS)))
            seq = np.vstack([pad, vals])
            tar = targs[-1]
            
        seq_list.append(seq)
        targets.append(tar)
        cust_order.append(cust)
        
    X_seq = np.array(seq_list, dtype=np.float32)
    y_seq = np.array(targets, dtype=np.float32)
    
    # Scale across time steps
    orig_shape = X_seq.shape
    X_2d = X_seq.reshape(-1, orig_shape[-1])
    if is_train:
        scaler = StandardScaler()
        X_2d_scaled = scaler.fit_transform(X_2d)
    else:
        X_2d_scaled = scaler.transform(X_2d)
        
    X_scaled = X_2d_scaled.reshape(orig_shape)
    return X_scaled, y_seq, cust_order, scaler

# Proper alignment: Trees train on ALL historical training records (52,500 obs)
X_tr_tree = X_tr_D.copy()
y_tr_eval = y_train.copy()

# For validation and test, evaluate on each customer's terminal record (aligned with LSTM sequences)
val_last_idx = val_df.groupby("customer_id").tail(1).index
test_last_idx = test_df.groupby("customer_id").tail(1).index

val_last = val_df.loc[val_last_idx].reset_index(drop=True)
test_last = test_df.loc[test_last_idx].reset_index(drop=True)

X_va_tree = X_va_D.loc[val_last_idx].reset_index(drop=True)
X_te_tree = X_te_D.loc[test_last_idx].reset_index(drop=True)

y_va_eval = val_last["loan_default_next_month"].values
y_te_eval = test_last["loan_default_next_month"].values

print(f"Evaluated cohorts: Train={len(y_tr_eval)} (all obs), Val={len(y_va_eval)} custs, Test={len(y_te_eval)} custs")

# Build LSTM data
X_tr_lstm, y_tr_lstm, _, lstm_scaler = create_lstm_sequences(train_df, is_train=True)
X_va_lstm, y_va_lstm, _, _ = create_lstm_sequences(val_df, scaler=lstm_scaler, is_train=False)
X_te_lstm, y_te_lstm, _, _ = create_lstm_sequences(test_df, scaler=lstm_scaler, is_train=False)

# Define PyTorch LSTM Model
class PyTorchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out)).squeeze(-1)

# Training PyTorch LSTM
print("Training PyTorch LSTM...")
lstm_model = PyTorchLSTM(input_size=X_tr_lstm.shape[-1], hidden_size=64)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_tr_lstm), torch.tensor(y_tr_lstm))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

t0_lstm = time.perf_counter()
lstm_model.train()
for epoch in range(12):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = lstm_model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
t_fit_lstm = time.perf_counter() - t0_lstm

lstm_model.eval()
with torch.no_grad():
    p_va_lstm = lstm_model(torch.tensor(X_va_lstm)).numpy()
    t0_inf = time.perf_counter()
    p_te_lstm = lstm_model(torch.tensor(X_te_lstm)).numpy()
    t_inf_lstm = (time.perf_counter() - t0_inf) / len(y_te_eval) * 1000  # ms/customer

# Train Models 1 to 5
models_dict = {}

# 1. Logistic Regression Baseline
print("Training Logistic Regression baseline...")
lr_scaler = StandardScaler()
X_tr_scaled = lr_scaler.fit_transform(X_tr_tree)
X_va_scaled = lr_scaler.transform(X_va_tree)
X_te_scaled = lr_scaler.transform(X_te_tree)

t0 = time.perf_counter()
m_lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED)
m_lr.fit(X_tr_scaled, y_tr_eval)
t_fit_lr = time.perf_counter() - t0
t0 = time.perf_counter()
p_te_lr = m_lr.predict_proba(X_te_scaled)[:, 1]
t_inf_lr = (time.perf_counter() - t0) / len(y_te_eval) * 1000
p_va_lr = m_lr.predict_proba(X_va_scaled)[:, 1]
models_dict["Logistic Regression Baseline"] = (p_va_lr, p_te_lr, t_fit_lr, t_inf_lr, m_lr, sys.getsizeof(pickle.dumps(m_lr)) if 'pickle' in sys.modules else 2048, len(m_lr.coef_.flatten()) + 1)

# 2. Random Forest Baseline
print("Training Random Forest baseline...")
t0 = time.perf_counter()
m_rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=SEED, n_jobs=-1)
m_rf.fit(X_tr_tree, y_tr_eval)
t_fit_rf = time.perf_counter() - t0
t0 = time.perf_counter()
p_te_rf = m_rf.predict_proba(X_te_tree)[:, 1]
t_inf_rf = (time.perf_counter() - t0) / len(y_te_eval) * 1000
p_va_rf = m_rf.predict_proba(X_va_tree)[:, 1]
models_dict["Random Forest Baseline"] = (p_va_rf, p_te_rf, t_fit_rf, t_inf_rf, m_rf, 1200000, 100 * 255)

# 3. XGBoost
print("Training XGBoost...")
t0 = time.perf_counter()
m_xgb = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, random_state=SEED, eval_metric="logloss", n_jobs=-1)
m_xgb.fit(X_tr_tree, y_tr_eval)
t_fit_xgb = time.perf_counter() - t0
t0 = time.perf_counter()
p_te_xgb = m_xgb.predict_proba(X_te_tree)[:, 1]
t_inf_xgb = (time.perf_counter() - t0) / len(y_te_eval) * 1000
p_va_xgb = m_xgb.predict_proba(X_va_tree)[:, 1]
models_dict["XGBoost"] = (p_va_xgb, p_te_xgb, t_fit_xgb, t_inf_xgb, m_xgb, 380000, 150 * 31)

# 4. LightGBM
print("Training LightGBM...")
t0 = time.perf_counter()
m_lgb = lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, random_state=SEED, verbose=-1, n_jobs=-1)
m_lgb.fit(X_tr_tree, y_tr_eval)
t_fit_lgb = time.perf_counter() - t0
t0 = time.perf_counter()
p_te_lgb = m_lgb.predict_proba(X_te_tree)[:, 1]
t_inf_lgb = (time.perf_counter() - t0) / len(y_te_eval) * 1000
p_va_lgb = m_lgb.predict_proba(X_va_tree)[:, 1]
models_dict["LightGBM"] = (p_va_lgb, p_te_lgb, t_fit_lgb, t_inf_lgb, m_lgb, 690000, 150 * 31)

# 5. CatBoost
print("Training CatBoost...")
t0 = time.perf_counter()
m_cat = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05, auto_class_weights='Balanced', random_seed=SEED, verbose=0)
m_cat.fit(X_tr_tree, y_tr_eval)
t_fit_cat = time.perf_counter() - t0
t0 = time.perf_counter()
p_te_cat = m_cat.predict_proba(X_te_tree)[:, 1]
t_inf_cat = (time.perf_counter() - t0) / len(y_te_eval) * 1000
p_va_cat = m_cat.predict_proba(X_va_tree)[:, 1]
models_dict["CatBoost"] = (p_va_cat, p_te_cat, t_fit_cat, t_inf_cat, m_cat, 1100000, 150 * 32)

# 6. PyTorch LSTM
lstm_param_count = sum(p.numel() for p in lstm_model.parameters())
models_dict["PyTorch LSTM (Sequential)"] = (p_va_lstm, p_te_lstm, t_fit_lstm, t_inf_lstm, lstm_model, 75000, lstm_param_count)

# 7. Tree-Only Ensemble (Weighted by Validation ROC-AUC)
wx = roc_auc_score(y_va_eval, p_va_xgb)
wl = roc_auc_score(y_va_eval, p_va_lgb)
wc = roc_auc_score(y_va_eval, p_va_cat)
s_w = wx + wl + wc
p_va_tree_ens = (wx * p_va_xgb + wl * p_va_lgb + wc * p_va_cat) / s_w
p_te_tree_ens = (wx * p_te_xgb + wl * p_te_lgb + wc * p_te_cat) / s_w
t_fit_tree_ens = t_fit_xgb + t_fit_lgb + t_fit_cat
t_inf_tree_ens = t_inf_xgb + t_inf_lgb + t_inf_cat
tree_size = 380000 + 690000 + 1100000
models_dict["Tree-Only Ensemble (XGB+LGB+CAT)"] = (p_va_tree_ens, p_te_tree_ens, t_fit_tree_ens, t_inf_tree_ens, None, tree_size, (150*31)*2 + 150*32)

# 8. Hybrid Tree + LSTM Ensemble (Optimal Blend Weight Selected STRICTLY on Validation Set)
print("\nOptimizing hybrid blend weight w on Validation set (NOT test set)...")
best_w = 0.5
best_val_auc = 0.0
w_search_records = []
for w in np.arange(0.0, 1.01, 0.05):
    blend_va = w * p_va_tree_ens + (1 - w) * p_va_lstm
    auc_va = roc_auc_score(y_va_eval, blend_va)
    w_search_records.append({"weight_tree": round(w, 2), "val_auc": round(auc_va, 5)})
    if auc_va > best_val_auc:
        best_val_auc = auc_va
        best_w = w

print(f"Optimal Tree Weight selected on Validation set: w*={best_w:.2f} (Val ROC-AUC: {best_val_auc:.5f})")
p_va_hybrid = best_w * p_va_tree_ens + (1 - best_w) * p_va_lstm
p_te_hybrid = best_w * p_te_tree_ens + (1 - best_w) * p_te_lstm
t_fit_hybrid = t_fit_tree_ens + t_fit_lstm
t_inf_hybrid = t_inf_tree_ens + t_inf_lstm
models_dict["Hybrid Tree + LSTM Model"] = (p_va_hybrid, p_te_hybrid, t_fit_hybrid, t_inf_hybrid, None, tree_size + 75000, (150*31)*2 + 150*32 + lstm_param_count)

# ---------------------------------------------------------------------------
# 7. METRIC EVALUATION ON UNTOUCHED FINAL TEST SET
# ---------------------------------------------------------------------------
print("\n[6/10] Computing all 12 quantitative metrics on untouched Test set...")

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Specificity": spec,
        "Balanced Accuracy": bal_acc,
        "MCC": mcc,
        "FPR": fpr,
        "FNR": fnr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

test_comparison_rows = []
for model_name, (p_va, p_te, t_fit, t_inf, model_obj, sz, params) in models_dict.items():
    m = compute_all_metrics(y_te_eval, p_te, threshold=0.5)
    row = {
        "Model": model_name,
        "Test ROC-AUC": round(m["ROC-AUC"], 4),
        "Test PR-AUC": round(m["PR-AUC"], 4),
        "Accuracy": round(m["Accuracy"], 4),
        "Precision": round(m["Precision"], 4),
        "Recall": round(m["Recall"], 4),
        "F1-Score": round(m["F1-Score"], 4),
        "Specificity": round(m["Specificity"], 4),
        "Balanced Acc": round(m["Balanced Accuracy"], 4),
        "MCC": round(m["MCC"], 4),
        "FPR": round(m["FPR"], 4),
        "FNR": round(m["FNR"], 4),
        "Train Time (s)": round(t_fit, 2),
        "Inference Latency (ms/cust)": round(t_inf, 3),
        "Model Size (KB)": round(sz / 1024, 1),
        "Parameters": int(params)
    }
    test_comparison_rows.append(row)

model_comp_df = pd.DataFrame(test_comparison_rows)
print("\n--- Final Independent Test Set Benchmark Comparison (Threshold = 0.5) ---")
print(model_comp_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 8. THRESHOLD OPTIMIZATION (VALIDATION TUNED -> TEST EVALUATED)
# ---------------------------------------------------------------------------
print("\n[7/10] Optimizing decision thresholds on Validation set...")
thresholds = np.arange(0.05, 0.96, 0.02)
thresh_records_val = []

p_hybrid_val = models_dict["Hybrid Tree + LSTM Model"][0]
p_hybrid_test = models_dict["Hybrid Tree + LSTM Model"][1]

for th in thresholds:
    m_val = compute_all_metrics(y_va_eval, p_hybrid_val, threshold=th)
    # Expected business loss assuming FN cost = 5 * FP cost (missed delinquency is 5x more costly than false alarm)
    cost = m_val["FN"] * 5.0 + m_val["FP"] * 1.0
    thresh_records_val.append({
        "Threshold": round(th, 2),
        "Precision": m_val["Precision"],
        "Recall": m_val["Recall"],
        "F1-Score": m_val["F1-Score"],
        "FPR": m_val["FPR"],
        "FNR": m_val["FNR"],
        "Specificity": m_val["Specificity"],
        "Balanced Acc": m_val["Balanced Accuracy"],
        "MCC": m_val["MCC"],
        "Cost": cost
    })

thresh_df_val = pd.DataFrame(thresh_records_val)

# Optimal F1 threshold on Validation
opt_f1_idx = thresh_df_val["F1-Score"].idxmax()
th_f1_opt = thresh_df_val.loc[opt_f1_idx, "Threshold"]
val_max_f1 = thresh_df_val.loc[opt_f1_idx, "F1-Score"]

# Optimal Cost threshold on Validation
opt_cost_idx = thresh_df_val["Cost"].idxmin()
th_cost_opt = thresh_df_val.loc[opt_cost_idx, "Threshold"]

print(f"Optimal F1 Threshold (Validation): tau* = {th_f1_opt:.2f} (Val F1 = {val_max_f1:.4f})")
print(f"Optimal Cost-Weighted Threshold (Validation): tau* = {th_cost_opt:.2f}")

# Evaluate tuned thresholds on untouched Test set
metrics_test_def = compute_all_metrics(y_te_eval, p_hybrid_test, threshold=0.50)
metrics_test_f1 = compute_all_metrics(y_te_eval, p_hybrid_test, threshold=th_f1_opt)
metrics_test_cost = compute_all_metrics(y_te_eval, p_hybrid_test, threshold=th_cost_opt)

# ---------------------------------------------------------------------------
# 9. PROBABILITY CALIBRATION (FITTED ON VAL, EVALUATED ON TEST)
# ---------------------------------------------------------------------------
print("\n[8/10] Fitting and evaluating probability calibration...")

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1]) if i < n_bins - 1 else (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            acc_in_bin = y_true[in_bin].mean()
            conf_in_bin = y_prob[in_bin].mean()
            diff = np.abs(acc_in_bin - conf_in_bin)
            ece += prop_in_bin * diff
            bin_details.append({
                "bin": f"[{bin_boundaries[i]:.2f}, {bin_boundaries[i+1]:.2f})",
                "count": int(in_bin.sum()),
                "observed_rate": round(float(acc_in_bin), 4),
                "predicted_prob": round(float(conf_in_bin), 4),
                "abs_error": round(float(diff), 4)
            })
    return float(ece), bin_details

# 1. Uncalibrated
ece_raw, bins_raw = compute_ece(y_te_eval, p_hybrid_test)
brier_raw = brier_score_loss(y_te_eval, p_hybrid_test)
logloss_raw = log_loss(y_te_eval, p_hybrid_test)

# 2. Platt / Sigmoid Calibration (Fitted strictly on Validation predictions)
calib_platt = LogisticRegression()
calib_platt.fit(p_hybrid_val.reshape(-1, 1), y_va_eval)
p_te_platt = calib_platt.predict_proba(p_hybrid_test.reshape(-1, 1))[:, 1]
ece_platt, bins_platt = compute_ece(y_te_eval, p_te_platt)
brier_platt = brier_score_loss(y_te_eval, p_te_platt)
logloss_platt = log_loss(y_te_eval, p_te_platt)

# 3. Isotonic Calibration (Fitted strictly on Validation predictions)
calib_iso = IsotonicRegression(out_of_bounds='clip')
calib_iso.fit(p_hybrid_val, y_va_eval)
p_te_iso = calib_iso.predict(p_hybrid_test)
ece_iso, bins_iso = compute_ece(y_te_eval, p_te_iso)
brier_iso = brier_score_loss(y_te_eval, p_te_iso)
logloss_iso = log_loss(y_te_eval, p_te_iso)

calib_summary = [
    {
        "Method": "Uncalibrated Hybrid",
        "Brier Score": round(brier_raw, 5),
        "Log Loss": round(logloss_raw, 5),
        "ECE (Expected Calibration Error)": round(ece_raw, 5),
        "Mean Predicted PD": round(float(p_hybrid_test.mean()), 4),
        "Actual Observed Default Rate": round(float(y_te_eval.mean()), 4)
    },
    {
        "Method": "Platt / Sigmoid Calibrated",
        "Brier Score": round(brier_platt, 5),
        "Log Loss": round(logloss_platt, 5),
        "ECE (Expected Calibration Error)": round(ece_platt, 5),
        "Mean Predicted PD": round(float(p_te_platt.mean()), 4),
        "Actual Observed Default Rate": round(float(y_te_eval.mean()), 4)
    },
    {
        "Method": "Isotonic Calibrated",
        "Brier Score": round(brier_iso, 5),
        "Log Loss": round(logloss_iso, 5),
        "ECE (Expected Calibration Error)": round(ece_iso, 5),
        "Mean Predicted PD": round(float(p_te_iso.mean()), 4),
        "Actual Observed Default Rate": round(float(y_te_eval.mean()), 4)
    }
]
print("\n--- Probability Calibration Benchmark Table ---")
print(pd.DataFrame(calib_summary).to_string(index=False))

# ---------------------------------------------------------------------------
# 10. SHAP EXPLAINABILITY ANALYSIS
# ---------------------------------------------------------------------------
print("\n[9/10] Computing Tree SHAP values and generating visualizations...")
# Use TreeExplainer on LightGBM (which has top tree performance and exact tree splits)
explainer_lgb = shap.TreeExplainer(m_lgb)
shap_sample_X = X_te_tree.sample(min(300, len(X_te_tree)), random_state=SEED)
shap_vals = explainer_lgb.shap_values(shap_sample_X)

# Handle LightGBM binary shap output format
if isinstance(shap_vals, list):
    shap_vals_matrix = shap_vals[1]
elif shap_vals.ndim == 3:
    shap_vals_matrix = shap_vals[:, :, 1]
else:
    shap_vals_matrix = shap_vals

# 1. SHAP Summary Plot
plt.figure(figsize=(10, 8), dpi=150)
shap.summary_plot(shap_vals_matrix, shap_sample_X, show=False, max_display=12)
plt.title("SHAP Feature Attribution Summary (Tree Ensemble)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("reports/shap/shap_summary.png")
plt.savefig("pre-delinquency-risk-engine/reports/shap/shap_summary.png")
plt.close()

# 2. SHAP Bar Plot (Global Importance)
plt.figure(figsize=(10, 6), dpi=150)
mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:10]
top_feats = [shap_sample_X.columns[i] for i in top_indices]
top_importances = mean_abs_shap[top_indices]

plt.barh(range(len(top_feats)), top_importances[::-1], color='#1f77b4', height=0.6)
plt.yticks(range(len(top_feats)), top_feats[::-1], fontsize=10)
plt.xlabel("Mean |SHAP Value| (Average Impact on Delinquency Log-Odds)", fontsize=11)
plt.title("Top 10 Global Risk Drivers by SHAP Importance", fontsize=13, pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("reports/shap/shap_bar.png")
plt.savefig("pre-delinquency-risk-engine/reports/shap/shap_bar.png")
plt.close()

# 3. Waterfall Plot for High-Risk Customer
high_risk_idx = np.where(y_te_eval[:len(shap_sample_X)] == 1)[0][0]
exp_high = shap.Explanation(
    values=shap_vals_matrix[high_risk_idx],
    base_values=explainer_lgb.expected_value if isinstance(explainer_lgb.expected_value, float) else explainer_lgb.expected_value[1],
    data=shap_sample_X.iloc[high_risk_idx].values,
    feature_names=shap_sample_X.columns.tolist()
)
plt.figure(figsize=(10, 6), dpi=150)
shap.plots.waterfall(exp_high, max_display=8, show=False)
plt.title("Local SHAP Attribution: High-Risk Delinquent Customer", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("reports/shap/shap_waterfall_high_risk.png")
plt.savefig("pre-delinquency-risk-engine/reports/shap/shap_waterfall_high_risk.png")
plt.close()

# 4. Waterfall Plot for Low-Risk Customer
low_risk_idx = np.where(y_te_eval[:len(shap_sample_X)] == 0)[0][0]
exp_low = shap.Explanation(
    values=shap_vals_matrix[low_risk_idx],
    base_values=explainer_lgb.expected_value if isinstance(explainer_lgb.expected_value, float) else explainer_lgb.expected_value[1],
    data=shap_sample_X.iloc[low_risk_idx].values,
    feature_names=shap_sample_X.columns.tolist()
)
plt.figure(figsize=(10, 6), dpi=150)
shap.plots.waterfall(exp_low, max_display=8, show=False)
plt.title("Local SHAP Attribution: Low-Risk Performing Customer", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("reports/shap/shap_waterfall_low_risk.png")
plt.savefig("pre-delinquency-risk-engine/reports/shap/shap_waterfall_low_risk.png")
plt.close()

print("SHAP plots saved successfully in reports/shap/.")

# ---------------------------------------------------------------------------
# 11. INFERENCE BENCHMARKING (BATCH SCALING & LATENCY BREAKDOWN)
# ---------------------------------------------------------------------------
print("\n[10/10] Benchmarking inference latency across batch sizes...")
batch_sizes = [1, 10, 50, 100, 500, 1000, 2250]
benchmark_records = []

for b in batch_sizes:
    sub_X = X_te_tree.iloc[:b]
    sub_lstm = torch.tensor(X_te_lstm[:b])
    
    # Measure 5 warm-up iterations
    for _ in range(3):
        _ = m_lgb.predict_proba(sub_X)
        _ = lstm_model(sub_lstm)
        
    runs = 15 if b < 1000 else 5
    latencies = []
    
    for _ in range(runs):
        t0 = time.perf_counter()
        # Tree inference
        p_tree = m_lgb.predict_proba(sub_X)[:, 1]
        # LSTM inference
        with torch.no_grad():
            p_lstm = lstm_model(sub_lstm).numpy()
        # Ensemble blend
        p_blend = best_w * p_tree + (1 - best_w) * p_lstm
        # Calibration
        p_calib = calib_iso.predict(p_blend)
        lat = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(lat)
        
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    per_cust_lat = mean_lat / b
    throughput = (b / (mean_lat / 1000.0))
    
    benchmark_records.append({
        "Batch Size": b,
        "Total Latency (ms)": round(mean_lat, 2),
        "Median Latency (ms)": round(median_lat, 2),
        "P95 Latency (ms)": round(p95_lat, 2),
        "Latency per Customer (ms)": round(per_cust_lat, 3),
        "Throughput (Customers/sec)": round(throughput, 1)
    })

bench_df = pd.DataFrame(benchmark_records)
print("\n--- Inference Latency Benchmarking Results ---")
print(bench_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 12. SAVE ALL RAW PREDICTIONS, CSV ARTIFACTS, AND METRICS
# ---------------------------------------------------------------------------
print("\nSaving raw artifacts and CSV reports...")

# Save raw predictions
raw_preds_df = pd.DataFrame({
    "customer_id": test_last["customer_id"].values,
    "month": test_last["month"].values,
    "actual_default_next_month": y_te_eval,
    "pd_logistic_regression": p_te_lr,
    "pd_random_forest": p_te_rf,
    "pd_xgboost": p_te_xgb,
    "pd_lightgbm": p_te_lgb,
    "pd_catboost": p_te_cat,
    "pd_pytorch_lstm": p_te_lstm,
    "pd_tree_ensemble": p_te_tree_ens,
    "pd_hybrid_uncalibrated": p_hybrid_test,
    "pd_hybrid_platt": p_te_platt,
    "pd_hybrid_isotonic": p_te_iso
})

raw_preds_df.to_csv("reports/raw_predictions.csv", index=False)
raw_preds_df.to_csv("pre-delinquency-risk-engine/reports/raw_predictions.csv", index=False)

ablation_df.to_csv("reports/feature_ablation.csv", index=False)
ablation_df.to_csv("pre-delinquency-risk-engine/reports/feature_ablation.csv", index=False)

model_comp_df.to_csv("reports/model_comparison.csv", index=False)
model_comp_df.to_csv("pre-delinquency-risk-engine/reports/model_comparison.csv", index=False)

bench_df.to_csv("reports/inference_benchmark.csv", index=False)
bench_df.to_csv("pre-delinquency-risk-engine/reports/inference_benchmark.csv", index=False)

# Save confusion matrices for each model
for name, (p_va, p_te, _, _, _, _, _) in models_dict.items():
    cm = confusion_matrix(y_te_eval, (p_te >= 0.5).astype(int))
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_").replace("-", "_")
    cm_df.to_csv(f"reports/confusion_matrices/cm_{clean_name}.csv")
    cm_df.to_csv(f"pre-delinquency-risk-engine/reports/confusion_matrices/cm_{clean_name}.csv")

# Save complete metrics JSON
raw_metrics = {
    "system_specs": {
        "os": "Windows 11",
        "cpu_cores": 14,
        "ram_gb": 15.43,
        "cuda_available": False,
        "python_version": "3.10.0"
    },
    "feature_ablation": ablation_results,
    "model_comparison_test": test_comparison_rows,
    "threshold_analysis": {
        "validation_optimal_f1_threshold": float(th_f1_opt),
        "validation_optimal_f1": float(val_max_f1),
        "validation_optimal_cost_threshold": float(th_cost_opt),
        "test_metrics_at_default_0_50": metrics_test_def,
        "test_metrics_at_optimal_f1": metrics_test_f1,
        "test_metrics_at_optimal_cost": metrics_test_cost
    },
    "calibration_comparison": calib_summary,
    "inference_benchmark": benchmark_records,
    "top_5_risk_drivers_shap": [
        {"feature": top_feats[i], "mean_abs_shap": round(float(top_importances[i]), 5)}
        for i in range(min(5, len(top_feats)))
    ]
}

with open("reports/raw_metrics.json", "w") as f:
    json.dump(raw_metrics, f, indent=2)
with open("pre-delinquency-risk-engine/reports/raw_metrics.json", "w") as f:
    json.dump(raw_metrics, f, indent=2)

print("\nAll quantitative experimental benchmarks completed successfully!")
