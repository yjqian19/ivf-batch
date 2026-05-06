# Query Scheduling Optimization in IVF-based Vector Search

Yujia Qian — [yjqian19@mit.edu](mailto:yjqian19@mit.edu) · Xiangyu Guan — [xiang949@mit.edu](mailto:xiang949@mit.edu)

---

## Abstract

Vector search is a core component of modern AI systems, yet most optimization efforts focus on index structure rather than query execution. This paper investigates whether throughput can be improved by batching concurrent queries over an IVF index to exploit overlap in the inverted lists they probe, without modifying the index itself. We compare two batching strategies, Batch(MV) and Batch(MM), against a sequential baseline under random and clustered query workloads. Experiments on SIFT1M confirm that batching consistently improves throughput over sequential execution: Batch(MV) outperforms Sequential by up to 19% on random queries, while Batch(MM) achieves up to 40% gains on clustered queries, with all schedulers preserving identical recall. The preferred scan mode shifts with workload structure, and batch size further modulates the relative performance between MV and MM. A microbenchmark isolates the mechanism underlying these differences and supports practical guidance on applying batch scheduling in IVF-based vector search.

---

## 1. Introduction

Vector search is central to a wide range of AI applications, including retrieval-augmented generation, semantic search, and recommendation systems. Given a query vector, the system must efficiently return the k nearest neighbors from a large corpus of stored vectors. At scale, such systems must sustain high query throughput while keeping tail latency bounded.

The dominant approach to improving vector search performance is to optimize the index structure. Hierarchical Navigable Small World graphs (HNSW), product quantization, and IVF variants have each offered significant gains in recall-efficiency trade-offs. By contrast, the query execution layer — how individual queries are dispatched and processed at runtime — has received comparatively less attention.

This work takes the index as fixed and asks: **can throughput be improved by reorganizing how concurrent queries are executed?** 

We focus on IVF (Inverted File Index), which partitions the vector space into clusters, each maintaining an inverted list of assigned vectors. To answer a query, the engine identifies the nearest centroids and scans the corresponding lists. Crucially, when multiple queries are processed together, many of them probe overlapping lists. If the engine exploits this overlap — loading each list once and computing distances for all queries that need it — the total number of memory loads decreases substantially.

To evaluate this approach, we design experiments under two workload conditions: random queries and clustered queries. We expect the degree of list overlap — and thus the benefit of batching — to differ substantially between the two, making them a natural test of when the approach succeeds and when it does not. We further consider two scan modes: Batch(MV), which computes distances for each query individually via matrix-vector multiplication, and Batch(MM), which stacks all queries sharing a list into a single matrix-matrix multiplication. We examine how the choice of scan mode and batch size affects throughput under each workload.

Our experiments reveal clear performance differences across workloads and scan modes, and offer practical insights into how batch scheduling strategies can be applied to improve throughput in IVF-based vector search.

---

## 2. Method

### 2.1 System Design

#### 2.1.1 IVF Engine

Our engine follows the IVF design from Faiss: base vectors are partitioned into 256 clusters via k-means, each cluster maintaining an inverted list. We use Faiss for centroid training but replace `index.search()` with a custom single-threaded implementation. Faiss's search is a stateless black box that processes each query independently, making cross-query optimizations invisible. Our engine exposes centroid lookup and list scanning as separate interfaces, giving the scheduler direct control over execution order and batch composition.

#### 2.1.2 Per-List Batch Scan

When a batch of queries is dispatched together, multiple queries often probe the same inverted list. The per-list batch scan exploits this by iterating over inverted lists in list-major order: for each list, distances are computed for all queries that probe it before moving to the next. Each list is therefore loaded from memory exactly once per batch, regardless of how many queries share it.

We define **L** as the number of queries within a batch that probe the same inverted list. Under sequential execution, L = 1 always. Under time-window batching, L grows with batch size and workload locality, and directly determines which scan mode is more efficient.

#### 2.1.3 Scan Modes

Two implementations of the per-list distance computation are provided, differing in how they handle the L queries sharing each list:

**Batch(MV).** Distances are computed for each of the L queries individually via a matrix-vector multiply (`vecs @ q`). Arithmetic intensity is constant at 0.5 FLOP/byte, independent of L.

**Batch(MM).** All L queries are stacked into a query matrix and distances are computed via a single matrix-matrix multiply (`vecs @ Q.T`). Arithmetic intensity scales as L/2 FLOP/byte, making MM increasingly efficient as L grows. However, GEMM carries non-trivial setup overhead at small L, degrading performance relative to MV. The crossover point is hardware-specific and is measured empirically in Section 3.3.

#### 2.1.4 Schedulers

Three configurations are evaluated:


| Scheduler  | Scan mode               | Batching policy                            |
| ---------- | ----------------------- | ------------------------------------------ |
| Sequential | Single-query GEMV       | None — queries are processed one at a time |
| Batch(MV)  | Per-query GEMV per list | Time-window, dual-trigger flush            |
| Batch(MM)  | Per-list GEMM           | Time-window, dual-trigger flush            |


Time-window batching accumulates arriving queries until either a time limit Δt elapses or a maximum batch size MaxBS is reached, then dispatches all accumulated queries as a single batch. The dual trigger bounds both waiting time at low load and memory pressure at high load.

### 2.2 Experiment Design

#### 2.2.1 Dataset

All experiments use the SIFT1M benchmark: one million 128-dimensional SIFT descriptors, evaluated under L2 distance, with 10,000 held-out test queries and precomputed ground truth. All vectors reside in memory; no disk I/O occurs during search. The dataset provides a standard reference point for IVF evaluation, with well-separated cluster geometry that makes it amenable to the inverted-file access pattern studied here.

#### 2.2.2 Workloads

Query arrivals are simulated at a target rate of 2,000 QPS. Two workloads are evaluated:

**Random.** All 10,000 SIFT test queries are used in their original order. Each query probes a largely disjoint set of 8 inverted lists, keeping L small. This represents an adversarial case for batching: list-reuse benefit is low, and any overhead from batch management directly reduces throughput.

**Clustered.** Queries are filtered to those whose nearest centroid falls among the 10 most-queried centroids, yielding 791 unique queries. These are sorted by primary centroid and then tiled (round-robin repetition) to reach 10,000 arrivals, matching the random workload in volume. Sorting by centroid ensures that consecutive arrivals within each time window belong to the same cluster, maximising L per list per batch. All 791 queries are real SIFT test vectors with valid ground truth, so Recall@10 is well-defined.

#### 2.2.3 Device

All experiments are conducted on a MacBook Pro with an Apple M3 Pro chip and 18 GB unified memory. The engine runs single-threaded throughout to isolate scheduling effects from multi-core parallelism.

#### 2.2.4 Experimental Protocol

Three experiments are designed to address the following research questions:

- **RQ1.** Does time-window batching improve throughput over sequential execution, under both random and clustered workloads?
- **RQ2.** How does scan mode choice (MV vs MM) affect performance under random versus clustered query distributions?
- **RQ3.** How does batch size affect the performance of batch schedulers?
- **RQ4.** What is the underlying mechanism that governs the relative performance of the two scan modes?

**Experiment 1 — Main Experiment (RQ1, RQ2).** All three schedulers are evaluated on both workloads under fixed parameters (n_clusters = 256, nprobe = 8, k = 10, Δt = 5 ms, MaxBS = 128) across 5 independent runs. The index is built once and reused across all runs to eliminate index-level variance. Per-run statistics include throughput (QPS), latency (average, P95, P99), unique list loads per query, and per-list sharing count L distribution. Summary statistics are reported as mean ± std; P99 latency is reported as the median across runs due to sensitivity to garbage collection pauses.

**Experiment 2 — Batch Size Parameter Sweep (RQ3).** A grid search over Δt ∈ {0.5, 1, 2, 5, 10, 20, 50} ms and MaxBS ∈ {32, 64, 128, 256} is conducted once on both workloads to examine how batch configuration affects throughput and the relative performance of the three schedulers.

**Experiment 3 — Microbenchmark (RQ4).** The MV and MM kernels are evaluated in isolation on a synthetic inverted list across a range of L values, with 50 repetitions per point. This removes scheduler noise and directly measures kernel-level performance as a function of L.

---

## 3. Results

We present results in three stages. First, the main experiment demonstrates the core throughput claims under fixed scheduler parameters. Second, the batch size parameter sweep shows how batch size modulates the relative performance of MV and MM. Third, the microbenchmark isolates L as the underlying mechanism explaining both.

### 3.1 Main Experiment

All three schedulers are evaluated at fixed parameters (Δt = 5 ms, MaxBS = 128) across 5 independent runs. Results are reported as mean ± std; P99 latency is the median across runs.

#### 3.1.1 Performance Overview

**Table 1. Throughput and Recall@10 (mean ± std, 5 runs). Δt = 5 ms, MaxBS = 128.**

**Random workload:**


| Scheduler  | QPS (mean ± std) | Avg Lat (ms) | P95 Lat (ms) | P99 Lat (median) | Recall@10     |
| ---------- | ---------------- | ------------ | ------------ | ---------------- | ------------- |
| Sequential | 1,854 ± 58       | 0.54         | 0.74         | 0.81             | 0.957 ± 0.000 |
| Batch(MV)  | **2,202 ± 23**   | 74.34        | 106.79       | 114.67           | 0.957 ± 0.000 |
| Batch(MM)  | 1,791 ± 87       | 104.69       | 134.60       | 152.89           | 0.957 ± 0.000 |


**Clustered workload:**


| Scheduler  | QPS (mean ± std) | Avg Lat (ms) | P95 Lat (ms) | P99 Lat (median) | Recall@10     |
| ---------- | ---------------- | ------------ | ------------ | ---------------- | ------------- |
| Sequential | 1,650 ± 33       | 0.61         | 0.78         | 0.88             | 0.981 ± 0.000 |
| Batch(MV)  | 1,933 ± 39       | 96.19        | 127.30       | 139.06           | 0.981 ± 0.000 |
| Batch(MM)  | **2,303 ± 111**  | 58.09        | 87.15        | 106.98           | 0.981 ± 0.000 |


Two findings stand out. First, batching consistently outperforms Sequential when the workload is favorable: Batch(MV) achieves +19% on random, and Batch(MM) achieves +40% on clustered. Second, the relative ordering of MV and MM is reversed across workloads — MV leads on random while MM leads on clustered — indicating that query distribution determines which scan mode is more efficient. The throughput gains of batching come at the cost of higher per-query latency due to queueing: average latency rises from ~0.5 ms (Sequential) to 58–105 ms for batch schedulers. Notably, Batch(MM) on clustered achieves both higher QPS and lower latency than Batch(MV), as faster batch execution reduces queue occupancy. Recall@10 is identical across all schedulers within each workload, confirming that the gains arise from execution reorganization rather than approximation.

#### 3.1.2 List Sharing (L)

**Table 2. Mean per-list reuse metrics across 5 runs.**

**Random workload:**


| Scheduler  | Lists/query | L (mean) | L (P50) | L (P95) |
| ---------- | ----------- | -------- | ------- | ------- |
| Sequential | 8.00        | 1.0      | 1       | 1       |
| Batch(MV)  | 2.23        | 3.6      | 3       | 8       |
| Batch(MM)  | 1.92        | 4.2      | 4       | 10      |


**Clustered workload:**


| Scheduler  | Lists/query | L (mean) | L (P50) | L (P95) |
| ---------- | ----------- | -------- | ------- | ------- |
| Sequential | 8.00        | 1.0      | 1       | 1       |
| Batch(MV)  | 0.55        | 14.6     | 5       | 64      |
| Batch(MM)  | 0.65        | 12.4     | 5       | 52      |


Sequential always loads 8 lists per query with L = 1. On the random workload, batching reduces unique list loads per query to 2.23 (3.6× reduction), with mean L = 3.6–4.2. On the clustered workload, this reduction reaches 0.55 lists per query, with mean L = 12.4–14.6. The contrast in L across the two workloads points to the underlying mechanism distinguishing MV and MM performance — a question addressed directly by the microbenchmark.

### 3.2 Batch Size Parameter Sweep

We sweep Δt and MaxBS on both workloads to examine how batch size affects the MV–MM trade-off. Four representative configurations are selected per workload. The sequential baseline achieves 1,937 QPS on random and 1,650 QPS on clustered.

#### 3.2.1 Random Workload

**Table 4. Selected (Δt, MaxBS) configurations on random workload.**


| Δt (ms) | MaxBS   | AvgBS (MV/MM)   | MV QPS    | MM QPS    | L mean (MV/MM) | Avg Lat (MV) | Δ (MM − MV) |
| ------- | ------- | --------------- | --------- | --------- | -------------- | ------------ | ----------- |
| 0.5     | 32      | 21.2 / 31.4     | 2,098     | 1,300     | 1.5 / 1.7      | 17.5ms       | −38%        |
| 5       | 64      | 60.2 / 63.3     | 2,230     | 1,446     | 2.5 / 2.5      | 42.3ms       | −35%        |
| **5**   | **128** | **82.6 / 98.0** | **2,277** | **2,227** | **3.1 / 3.5**  | **60.1ms**   | **−2%**     |
| 20      | 256     | 222.2 / 175.4   | 2,322     | 2,562     | 7.3 / 5.8      | 160.8ms      | +10%        |


As Δt and MaxBS grow, AvgBS and L both increase, and MM's disadvantage narrows and eventually reverses. At the smallest configuration (Δt = 0.5 ms, MaxBS = 32), L ≈ 1.5–1.7 places both schedulers well below the hardware crossover, and MM's GEMM setup overhead dominates (−38%). At the main experiment parameters (Δt = 5 ms, MaxBS = 128), L rises to 3.1–3.5 and the gap nearly closes (−2%). At Δt = 20 ms and MaxBS = 256, L reaches 5.8–7.3 and MM overtakes MV by 10%. The throughput gains come at a direct latency cost: MV average latency grows from 17.5 ms to 160.8 ms across the same range — driven by queueing — while the sequential baseline maintains ~0.5 ms.

#### 3.2.2 Clustered Workload

**Table 5. Selected (Δt, MaxBS) configurations on clustered workload.**


| Δt (ms) | MaxBS   | AvgBS (MV/MM)    | MV QPS    | MM QPS    | L mean (MV/MM)  | Avg Lat (MV) | Δ (MM − MV) |
| ------- | ------- | ---------------- | --------- | --------- | --------------- | ------------ | ----------- |
| 0.5     | 32      | 30.5 / 31.5      | 1,987     | 1,884     | 7.5 / 7.6       | 23.2ms       | −5%         |
| 5       | 64      | 62.9 / 61.7      | 1,934     | 2,156     | 11.1 / 11.0     | 48.7ms       | +11%        |
| **5**   | **128** | **120.5 / 71.4** | **1,952** | **2,324** | **14.6 / 11.9** | **95.4ms**   | **+19%**    |
| 20      | 256     | 238.1 / 208.3    | 1,948     | 2,424     | 17.8 / 17.3     | 192.3ms      | +24%        |


In contrast to the random workload, MM wins from small-to-medium batch sizes because clustered queries concentrate on overlapping lists, keeping L above the crossover even in small batches. At Δt = 0.5 ms and MaxBS = 32, L ≈ 7.5–7.6 sits right at the hardware crossover and MM barely loses (−5%). As batch size grows, L rises well above the crossover and MM's advantage increases steadily to +24%. Notably, at the main experiment parameters MM achieves AvgBS = 71 versus MV's 121 — faster execution drains the queue earlier — yet L remains at 11.9, well above the crossover. As with the random workload, MV latency grows with batch size (23 ms to 192 ms), reflecting the same throughput–latency trade-off.

### 3.3 Microbenchmark

To isolate the effect of L from scheduler dynamics, the MV and MM kernels are benchmarked directly on a synthetic inverted list (n = 4,000 vectors, d = 128) for L ∈ {1, …, 256}, with 50 repetitions per point.

**Table 5. Kernel throughput as a function of L. Metric: ns/(query × vector), lower is better.**


| L     | MV (ns/q·v) | MM (ns/q·v) | MM/MV speedup         |
| ----- | ----------- | ----------- | --------------------- |
| 1     | 5.515       | 5.583       | 0.99×                 |
| 2     | 5.474       | 15.852      | 0.35×                 |
| 4     | 5.279       | 7.961       | 0.66×                 |
| **8** | **5.214**   | **4.016**   | **1.30× ← crossover** |
| 16    | 5.154       | 3.009       | 1.71×                 |
| 32    | 5.127       | 1.508       | 3.40×                 |
| 64    | 5.178       | 1.425       | 3.63×                 |
| 256   | 5.114       | 1.355       | 3.77×                 |


MV throughput is approximately constant across all values of L (5.1–5.5 ns/q·v), as each query is processed independently. MM throughput improves sharply with L, crossing the MV baseline at L = 8 and reaching a 3.77× speedup at L = 256. The crossover corresponds to the point where GEMM arithmetic intensity (L/2 FLOP/byte) exceeds the GEMV baseline (0.5 FLOP/byte). In the main experiment, mean L ≈ 3.6–4.2 on random (below the crossover) and mean L ≈ 12.4–14.6 on clustered (above it), fully accounting for the reversed MV–MM ordering observed in Table 1.

---

## 4. Discussion

**RQ1.** Batching can meaningfully improve throughput over sequential execution, though the degree depends on both the scan mode and the workload. Batch(MV) outperforms Sequential on both workloads; Batch(MM) does so only on the clustered workload, where it achieves +40%. On the random workload, Batch(MM) falls 3% below Sequential — the queueing cost of batching is not recovered when per-batch scan overhead is high. The primary source of gain is list reuse: batching reduces unique list loads per query from 8 to 2.23 on random and 0.55 on clustered, a reduction of up to 14.5× relative to sequential processing.

**RQ2.** Query distribution substantially affects the ranking of all three schedulers. On the random workload, Batch(MV) is the clear winner while Batch(MM) falls below even Sequential; on the clustered workload, Batch(MM) takes the lead and Sequential is last. Neither batch mode dominates unconditionally — the workload determines which strategy is most effective. On the clustered workload, MM's faster execution also reduces queue occupancy, further lowering per-query latency.

**RQ3.** Larger batch sizes raise L across both workloads. On the random workload, this narrows the gap between MV and Sequential but does not invert it. On the clustered workload, MM's lead over both MV and Sequential is robust across all batch sizes. Batch size therefore modulates the magnitude of the performance differences but does not alter their direction given a fixed query distribution.

**RQ4.** The microbenchmark isolates L as the governing variable. MV throughput is approximately constant regardless of L, while MM throughput improves sharply with L, crossing the MV baseline at L = 8. This threshold corresponds to the point at which GEMM arithmetic intensity (L/2 FLOP/byte) exceeds that of GEMV (0.5 FLOP/byte) on the experimental hardware. In the main experiment, mean L ≈ 3.6–4.2 on random places Batch(MM) below this crossover, while mean L ≈ 12.4–14.6 on clustered places it well above, fully accounting for the reversed ordering in Table 1.

**Summary.** Time-window batching is a viable strategy for improving IVF throughput without modifying the index or compromising recall. In practice, the choice of scan mode and batch parameters should be guided by the expected query distribution: workloads with high spatial locality favour Batch(MM) with larger batch sizes, while low-locality workloads favour Batch(MV). The L distribution of a target workload provides a principled basis for this configuration. The gains, however, come at the cost of higher per-query latency due to queueing, and the throughput–latency trade-off must be weighed against the application's tail-latency requirements.

**Limitations and Future Work.** The engine is deliberately single-threaded to isolate scheduling effects; whether the observed gains survive in a multi-core deployment remains an open question. The current design assigns each scheduler a fixed scan mode — an adaptive strategy that switches between MV and MM per batch based on observed L would eliminate the regression on random workloads while preserving MM's advantage on clustered ones. All results are reported on SIFT1M; validation on datasets with different dimensionalities or cluster geometries would strengthen the generalizability of the findings.