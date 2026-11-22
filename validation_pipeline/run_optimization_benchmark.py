import os
import sys
import time
import json
import numpy as np
import pandas as pd

print("=" * 80)
print("RUNNING OPTIMIZATION BENCHMARK: BEFORE VS AFTER")
print("=" * 80)

# Load data
df = pd.read_csv("pre-delinquency-risk-engine/financial_stress_full_bank_grade_dataset.csv")
df = df.sort_values(["customer_id", "month"]).reset_index(drop=True)

# 1. Rolling Slope: np.polyfit vs Vectorized Analytic Formula
print("\n[1/3] Benchmarking Rolling Slope Calculation...")
col = "credit_card_utilization"

# Test on 3,000 rows to avoid waiting minutes, then extrapolate reliably
sample_df = df.iloc[:3000].copy()

# A. Original polyfit approach
t0 = time.perf_counter()
g = sample_df.groupby("customer_id")[col]
_ = g.rolling(3, min_periods=2).apply(
    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
).reset_index(level=0, drop=True)
t_polyfit_sample = time.perf_counter() - t0
# Extrapolate to 90,000 rows
t_polyfit_est = t_polyfit_sample * (90000 / 3000)

# B. Vectorized analytic slope approach on full 90,000 rows
t0 = time.perf_counter()
lag1 = df.groupby("customer_id")[col].shift(1)
lag2 = df.groupby("customer_id")[col].shift(2)
slope_3pt = (df[col] - lag2) / 2.0
slope_2pt = df[col] - lag1
slope_vec = np.where(lag2.notnull(), slope_3pt, np.where(lag1.notnull(), slope_2pt, 0.0))
t_vectorized = time.perf_counter() - t0

speedup_slope = t_polyfit_est / max(t_vectorized, 0.001)
print(f"Original np.polyfit (estimated 90k rows): {t_polyfit_est:.2f} s")
print(f"Vectorized analytic formula (actual 90k rows): {t_vectorized:.4f} s")
print(f"Preprocessing Slope Speedup: {speedup_slope:.1f}x faster!")

# 2. Memory Optimization: float64 vs float32
print("\n[2/3] Benchmarking Memory Downcasting Optimization...")
mem_before = df.memory_usage(deep=True).sum() / (1024 ** 2)

df_opt = df.copy()
for c in df_opt.select_dtypes(include=["float64"]).columns:
    df_opt[c] = df_opt[c].astype(np.float32)
for c in df_opt.select_dtypes(include=["int64"]).columns:
    if c != "customer_id":
        df_opt[c] = df_opt[c].astype(np.int32)
        
mem_after = df_opt.memory_usage(deep=True).sum() / (1024 ** 2)
mem_reduction_pct = (1.0 - (mem_after / mem_before)) * 100.0
print(f"Memory before downcasting: {mem_before:.2f} MB")
print(f"Memory after downcasting: {mem_after:.2f} MB")
print(f"RAM consumption reduction: {mem_reduction_pct:.1f}%")

# 3. Inference Latency: Single vs Batch Vectorization
print("\n[3/3] Benchmarking Inference Vectorization Speedup...")
# Load benchmark numbers from raw_metrics.json
with open("reports/raw_metrics.json") as f:
    raw_m = json.load(f)
bench_list = raw_m["inference_benchmark"]

single_lat = bench_list[0]["Latency per Customer (ms)"]  # Batch 1
batch_lat = bench_list[-1]["Latency per Customer (ms)"]  # Batch 2250
single_tp = bench_list[0]["Throughput (Customers/sec)"]
batch_tp = bench_list[-1]["Throughput (Customers/sec)"]
speedup_inf = single_lat / batch_lat

print(f"Single-customer latency: {single_lat:.3f} ms ({single_tp:.1f} cust/sec)")
print(f"Batch-2250 latency per customer: {batch_lat:.4f} ms ({batch_tp:.1f} cust/sec)")
print(f"Inference throughput scaling factor: {speedup_inf:.1f}x speedup per customer")

opt_results = {
    "preprocessing_optimization": {
        "original_method": "np.polyfit inside rolling.apply",
        "optimized_method": "Vectorized closed-form analytic slope formula (y_t - y_{t-2})/2",
        "original_time_est_seconds": round(t_polyfit_est, 2),
        "optimized_time_actual_seconds": round(t_vectorized, 4),
        "speedup_factor": round(speedup_slope, 1)
    },
    "memory_optimization": {
        "baseline_ram_mb": round(mem_before, 2),
        "optimized_ram_mb": round(mem_after, 2),
        "reduction_percentage": round(mem_reduction_pct, 1)
    },
    "inference_vectorization": {
        "single_customer_latency_ms": round(single_lat, 3),
        "batch_latency_per_customer_ms": round(batch_lat, 4),
        "single_throughput_per_sec": round(single_tp, 1),
        "batch_throughput_per_sec": round(batch_tp, 1),
        "throughput_gain_factor": round(speedup_inf, 1)
    }
}

with open("reports/optimization_results.json", "w") as f:
    json.dump(opt_results, f, indent=2)
with open("pre-delinquency-risk-engine/reports/optimization_results.json", "w") as f:
    json.dump(opt_results, f, indent=2)

print("\nOptimization benchmarks completed and saved successfully!")
