# Inference Benchmark Report
**Pre-Delinquency Risk Engine — Full Hybrid Pipeline**

---

## 1. Pipeline Components

The benchmark covers the **complete end-to-end hybrid inference pipeline**, including:

| Component | Description |
|-----------|-------------|
| **Feature Engineering** | Rolling statistics, slope computation, lag features |
| **LightGBM Scorer** | Gradient-boosted tree model for tabular risk features |
| **PyTorch LSTM** | Sequential model processing monthly behaviour history |
| **Platt Calibration** | Sigmoid calibration layer applied to raw model scores |
| **Score Fusion** | Weighted combination of LightGBM + LSTM outputs |

No component is excluded. The latency figures represent the wall-clock time a production API or batch job would observe.

---

## 2. Methodology

- **Tool:** Python `time.perf_counter` with a custom benchmarking harness
- **Repetitions:** 100 timed runs per batch size
- **Reported statistic:** Median latency (robust to occasional GC pauses)
- **Warm-up:** 5 un-timed warm-up iterations before measurement begins, to populate caches and JIT paths
- **Environment:** Single-threaded inference; no parallelism or batched GPU acceleration
- **Data:** Synthetic but representative customer feature vectors matching production schema

---

## 3. Results

| Batch Size | Total Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Latency / Customer (ms) | Throughput (cust/sec) |
|------------|-------------------|---------------------|------------------|-------------------------|-----------------------|
| 1 | 1.51 | 1.25 | 2.24 | 1.512 | 661.2 |
| 10 | 1.60 | 1.53 | 1.85 | 0.160 | 6,259.3 |
| 50 | 2.02 | 2.00 | 2.45 | 0.040 | 24,773.1 |
| 100 | 2.25 | 2.02 | 3.03 | 0.022 | 44,541.2 |
| 500 | 4.61 | 4.69 | 5.64 | 0.009 | 108,346.0 |
| 1,000 | 4.71 | 4.61 | 5.34 | 0.005 | 212,168.3 |
| 2,250 | 9.42 | 9.14 | 10.40 | 0.004 | 238,897.6 |

> **Metric definitions**
> - *Total Latency* — median wall-clock time for one complete pipeline call at that batch size
> - *P95 Latency* — 95th-percentile latency across 100 repetitions
> - *Latency / Customer* — Total Latency ÷ Batch Size
> - *Throughput* — Batch Size ÷ (Total Latency in seconds)

---

## 4. Scaling Analysis

### 4.1 Sub-linear Latency Growth

Latency does **not** scale linearly with batch size. A naive linear extrapolation would predict:

```
Expected latency for batch=2,250 (linear): 2,250 × 1.51 ms = 3,397.5 ms
Actual latency for batch=2,250:                               9.42 ms
Ratio:                                                        360× faster than linear
```

This sub-linear behaviour arises from vectorised matrix operations in both the LightGBM leaf-assignment and the PyTorch LSTM forward pass, which amortise Python call overhead, memory allocation, and CPU cache misses across the entire batch.

### 4.2 Throughput Gain (Batch vs. Single)

| Mode | Throughput |
|------|-----------|
| Single-customer (batch=1) | 661.2 cust/sec |
| Maximum batch (batch=2,250) | 238,897.6 cust/sec |
| **Efficiency gain** | **~361×** |

### 4.3 Throughput Inflection

Throughput grows rapidly up to batch=1,000 (212,168 cust/sec) and plateaus modestly beyond that. This suggests CPU vector-unit saturation around batch sizes of 1,000–2,250. For production scheduling, **batch=1,000** is the practical sweet spot — near-peak throughput with lower memory overhead.

### 4.4 P95 Real-Time Suitability

| Batch Size | P95 Latency | Suitable for |
|------------|-------------|--------------|
| 1 | 2.24 ms | Real-time API (< 50 ms SLA) ✅ |
| 10 | 1.85 ms | Real-time micro-batch ✅ |
| 1,000 | 5.34 ms | Near-real-time streaming ✅ |

All measured P95 values comfortably satisfy a 50 ms API latency SLA.

---

## 5. Operational Recommendations

### Real-Time Scoring API
- Use **batch size = 1** (or up to 10 for marginal gain)
- P95 latency of **2.24 ms** leaves ample headroom for network + serialisation overhead within a 50 ms SLA
- Suitable for: mobile app credit checks, POS decisioning, inbound call-centre risk lookup

### Nightly Portfolio Scoring
- Use **batch size = 1,000** for optimal throughput/memory balance
- **212,168 cust/sec** → a 1-million-customer portfolio scored in under **5 seconds**
- Suitable for: overnight risk refresh, regulatory reporting runs, early-warning alert generation

### Streaming / Near-Real-Time
- Use **batch size = 100–500**
- Throughput range: 44,541–108,346 cust/sec with P95 < 6 ms
- Suitable for: Kafka consumer pipelines, event-driven delinquency triggers

---

## 6. Hardware Disclosure

| Item | Specification |
|------|---------------|
| CPU | Intel Core i7 |
| Cores / Threads | 14 cores / 14 threads |
| RAM | 15.43 GB |
| GPU / CUDA | None (CPU-only inference) |
| Operating System | Windows 11 |
| Python Version | 3.10.0 |
| Inference Mode | Single-threaded, CPU-only |

> [!NOTE]
> All results are CPU-only. GPU acceleration (e.g., CUDA-enabled PyTorch) would further improve LSTM throughput for large batch sizes. Results on different hardware will vary; this benchmark establishes a **conservative baseline** on commodity laptop-grade hardware.
