# Reference Notes

## 1. Related Works

### IVF Indexing

Most vector search systems build on IVF, which partitions the vector space into clusters via k-means and assigns each vector to its nearest centroid's inverted list. At query time, only `nprobe` lists are scanned, trading recall for speed. The nprobe setting is the primary knob governing the latency–recall tradeoff and directly determines how many inverted lists queries overlap — the key factor enabling cluster-based batching.

**[1] Billion-Scale Similarity Search with GPUs**
Jeff Johnson, Matthijs Douze, Hervé Jégou (Meta FAIR) — IEEE Transactions on Big Data, 2019
https://arxiv.org/abs/1702.08734
The foundational Faiss paper. Defines the IVF index architecture (centroid assignment, inverted lists, nprobe-driven cluster probing) and GPU-accelerated k-selection. Essential baseline for understanding the index structure all batching strategies operate on.

---

### Batch-Aware Vector Search *(main focus)*

Batching concurrent queries is the primary mechanism for improving throughput in vector search. The key insight is that multiple queries frequently probe overlapping inverted lists, so executing them together allows shared data loads, better cache utilization, and vectorized distance computation across queries. This direction spans production systems, GPU-accelerated indexes, and system-level design of concurrent query execution.

**[2] Milvus: A Purpose-Built Vector Data Management System**
Jianguo Wang et al. (Purdue / Zilliz) — ACM SIGMOD 2021
https://www.cs.purdue.edu/homes/csjgwang/pubs/SIGMOD21_Milvus.pdf
Describes how Milvus partitions concurrent queries into cache-fitting query blocks and dispatches them with multi-threading. The query block design — grouping queries so their combined working set fits in cache — is a direct structural parallel to this project's cluster-based batching strategy.

**[3] Manu: A Cloud Native Vector Database Management System**
Rentong Guo et al. (Zilliz) — PVLDB Vol. 15, 2022
https://www.vldb.org/pvldb/vol15/p3548-yan.pdf
Extends Milvus to a cloud-native architecture with MVCC and delta consistency. Benchmarks concurrent query throughput under varying workloads and evaluates the IVF query dispatch pipeline at scale, providing a production reference for the throughput metrics targeted in this project.

**[4] GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ**
NTU / NVIDIA cuVS team — arXiv 2025
https://arxiv.org/abs/2602.23999
Explicitly demonstrates that batching queries together while probing IVF clusters exposes regular computation patterns ideal for GEMM-style execution and enables coalesced memory access within lists. Directly validates the core hypothesis of cluster-based batching: co-scheduling queries that share probe lists drives throughput gains.

**[5] The Design and Implementation of a Real Time Visual Search System on JD E-commerce Platform**
Jie Li et al. (JD.com) — arXiv 2019
https://arxiv.org/abs/1908.07389
Describes Vearch/Gamma, a production vector search engine built on Faiss with lock-free concurrent indexing and querying. Provides a concrete case study of handling high-concurrency IVF queries at e-commerce scale, including how batching behavior emerges from request queuing under load.

---

### Query Scheduling in Databases

Classical multi-query optimization research addresses the same fundamental problem: when multiple queries access overlapping data, scheduling them to share I/O and computation reduces redundant work. Cooperative scan techniques in column stores — grouping concurrent queries to share a single sequential pass — are the direct conceptual predecessor to time-window and cluster-based batching over IVF inverted lists.

**[6] Cooperative Scans: Dynamic Bandwidth Sharing in a DBMS**
Marcin Zukowski, Sándor Héman, Niels Nes, Peter Boncz — VLDB 2007
https://15721.courses.cs.cmu.edu/spring2016/papers/p723-zukowski.pdf
Canonical work on sharing scan I/O across concurrent queries in column stores. Introduces dynamic batching of queries that touch the same data pages, with analysis of memory bandwidth savings. The "load once, serve many" principle maps directly to cluster-based batching over IVF inverted lists.

**[7] Shared Workload Optimization**
Georgios Giannikis, Darko Makreshanski, Gustavo Alonso, Donald Kossmann (ETH Zurich) — PVLDB Vol. 7, 2014
https://vldb.org/pvldb/vol7/p429-giannikis.pdf
Develops a global optimizer that batches concurrent queries to share hash joins and scans. The stochastic knapsack formulation for grouping queries by shared access patterns is a close theoretical analog to the batch formation problem in cluster-based batching.

---

### SIMD-Optimized Distance Computation

Vectorized distance kernels are what make batching computationally beneficial beyond just reducing I/O. When multiple queries scan the same inverted list together, SIMD instructions can process several query–vector distance computations in a single instruction, multiplying the throughput benefit of batching.

**[8] Cache Locality Is Not Enough: High-Performance Nearest Neighbor Search with Product Quantization Fast Scan**
Fabien André, Anne-Marie Kermarrec, Nicolas Le Scouarnec (Inria) — PVLDB 2015
http://www.vldb.org/pvldb/vol9/p288-andre.pdf
Introduces PQ Fast Scan, which fits PQ lookup tables into SIMD registers (SSE/AVX) for 4–6× speedup. Adopted directly into Faiss's `IndexIVFPQFastScan`. Demonstrates that the inner distance-computation loop within an IVF cluster is the critical path and is amenable to SIMD batching.

**[9] Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)**
Ruiqi Guo et al. (Google Research) — ICML 2020
http://proceedings.mlr.press/v119/guo20h/guo20h.pdf
Introduces anisotropic quantization loss and AVX-accelerated distance kernels. ScaNN's SIMD design informed Faiss's FastScan implementation and is a key reference for the hardware-level throughput gains achievable when distance computation is properly vectorized across batched queries.

---

## 2. Datasets & Benchmarks

### Recommended Datasets

| Dataset | Dimensions | Base vectors | Query vectors | Source | Notes |
|---------|-----------|-------------|--------------|--------|-------|
| **SIFT1M** | 128 | 1,000,000 | 10,000 | [INRIA](http://corpus-texmex.irisa.fr/) | Classic ANN benchmark; well-understood baselines. Primary recommendation. |
| **SIFT10K** | 128 | 10,000 | 100 | Same as above | Smaller subset; useful for fast debugging and unit tests. |
| **GloVe-100** | 100 | 1,183,514 | 10,000 | [ANN-Benchmarks](https://ann-benchmarks.com/) | Real word embeddings; natural cluster structure differs from SIFT. Good for generalizability. |
| **Fashion-MNIST** | 784 | 60,000 | 10,000 | [ANN-Benchmarks](https://ann-benchmarks.com/) | Higher-dimensional; tests whether batching benefits hold in different regimes. |
| **NYTimes** | 256 | 290,000 | 10,000 | [ANN-Benchmarks](https://ann-benchmarks.com/) | Sparse-ish real data; another distribution to test against. |

### Suggestion

Use **SIFT1M** as the primary dataset (reviewers can contextualize your numbers easily) and **GloVe-100** as a secondary dataset for generalizability (different cluster structure).

### Query Workloads

Design at least two workloads:

- **Random queries** — queries sampled uniformly; low overlap between probe lists across queries. This is the worst case for cluster-based batching.
- **Clustered queries** — queries drawn from a few regions of the vector space; high overlap. This is the best case for cluster-based batching.

Varying the workload lets you show *when* each strategy wins, not just *whether* it wins.

---

## 3. Core Engine Implementation

The core engine is the single-query IVF search baseline that all three schedulers run on top of. Keep it as simple as possible — correctness and measurability matter more than optimization at this stage. **Use Faiss directly.** It is the reference implementation for IVF, is battle-tested, and provides ground-truth results you can compare against.

### Library

```
pip install faiss-cpu        # CPU build, sufficient for this project
# or
pip install faiss-gpu        # if you have CUDA
```

Faiss is maintained by Meta FAIR (the authors of reference [1]) and implements exactly the IVF architecture described in the proposal.

### Minimal Implementation

The entire core engine — build index, load data, single-query search — is ~30 lines:

```python
import numpy as np
import faiss

# ── 1. Build the index ────────────────────────────────────────────────────────
def build_ivf_index(vectors: np.ndarray, n_clusters: int = 256) -> faiss.IndexIVFFlat:
    """Train an IVFFlat index on the given float32 vectors."""
    d = vectors.shape[1]                          # dimension
    quantizer = faiss.IndexFlatL2(d)              # exact L2 centroid search
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)
    index.train(vectors)                          # k-means, one-time cost
    index.add(vectors)                            # assign vectors to lists
    return index

# ── 2. Single-query search ────────────────────────────────────────────────────
def search(index: faiss.IndexIVFFlat, query: np.ndarray,
           k: int = 10, nprobe: int = 8):
    """Return top-k neighbor IDs and distances for one query vector."""
    index.nprobe = nprobe                         # lists to scan per query
    distances, ids = index.search(query[None, :], k)  # shape (1, d) → (1, k)
    return ids[0], distances[0]

# ── 3. Batch search (baseline sequential) ────────────────────────────────────
def search_batch(index: faiss.IndexIVFFlat, queries: np.ndarray,
                 k: int = 10, nprobe: int = 8):
    """Search a batch of queries sequentially (the baseline scheduler)."""
    index.nprobe = nprobe
    distances, ids = index.search(queries, k)     # Faiss handles the loop
    return ids, distances
```

All vectors must be `np.float32` (Faiss requirement). Call `vectors.astype(np.float32)` when loading.

### Loading SIFT1M

SIFT1M is distributed as raw binary `.fvecs` / `.ivecs` files. A standard loader:

```python
def read_fvecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()

def read_ivecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].copy()

# Usage
base    = read_fvecs("sift/sift_base.fvecs")        # (1_000_000, 128)
queries = read_fvecs("sift/sift_query.fvecs")        # (10_000, 128)
gt      = read_ivecs("sift/sift_groundtruth.ivecs")  # (10_000, 100) true neighbors
```

### Verifying Correctness (Recall@k)

Before building any scheduler, confirm the core engine reaches expected recall against the ground truth:

```python
def recall_at_k(predicted_ids: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Fraction of queries where the true nearest neighbor is in top-k results."""
    hits = sum(gt[0] in pred[:k] for pred, gt in zip(predicted_ids, ground_truth))
    return hits / len(predicted_ids)

ids, _ = search_batch(index, queries, k=10, nprobe=8)
print(f"Recall@10: {recall_at_k(ids, gt, k=10):.3f}")   # expect ~0.90 with nprobe=8
```

A recall of ≥0.90 on SIFT1M with `nprobe=8` is a reasonable sanity check. Once confirmed, **this recall value becomes the fixed target** — all three schedulers must hit the same number to ensure fair throughput comparison (see Section 3 / Evaluation Advice).

### Key Parameters

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| `n_clusters` | 256 (small) – 4096 (large) | More clusters → shorter lists → faster scan, lower recall |
| `nprobe` | 4–64 | More probes → higher recall, higher latency, more list overlap between queries |
| `k` | 10–100 | Top-k neighbors to return |

For this project, fix `n_clusters=256` and `k=10`, then vary `nprobe` to control the recall–speed tradeoff and the amount of inter-query cluster overlap (which directly affects the gain from cluster-based batching).

### What the Schedulers Sit On Top Of

The schedulers do **not** replace Faiss — they change *when* and *how many* queries are sent to Faiss:

- **Sequential:** call `index.search(q[None,:], k)` once per query, in arrival order.
- **Time-window:** collect queries for Δt ms, then call `index.search(batch, k)` once.
- **Cluster-based:** inspect each query's top-`nprobe` centroids (via `quantizer.search`), group queries by centroid overlap, then call `index.search` per group.

All three paths return identical results for the same queries; only throughput and latency differ.

---

## 4. Time-Window Batching — Design Notes

### Core Trade-off

A larger window produces a larger batch, which improves hardware utilization (better SIMD fill, fewer redundant cache loads), but increases the waiting time each individual query experiences before execution begins.

### Dual-trigger Flush Mechanism

The window uses two trigger conditions, whichever fires first:

1. **Time trigger:** once Δt has elapsed since the window opened, the current batch is flushed immediately regardless of its size.
2. **Size trigger:** once the batch accumulates `max_batch_size` queries, it is flushed immediately regardless of remaining time.

This is a standard pattern in systems that buffer work (e.g., Kafka producer batching, database write buffers). Using only a time trigger wastes resources at low QPS (tiny batches) and risks memory pressure at high QPS (unbounded batches). Using only a size trigger causes unacceptable latency at low QPS (a single query may wait indefinitely).

### Parameter Ranges

Since a single IVF query takes roughly 0.1–10 ms depending on dataset size and `nprobe`, setting Δt too small (e.g., 0.1 ms) collapses to sequential execution, while setting it too large (e.g., 100 ms) adds unacceptable latency. Planned sweep:

- Δt ∈ {0.5, 1, 2, 5, 10, 20, 50} ms
- `max_batch_size` ∈ {32, 64, 128, 256}

### Simulating Concurrent Arrivals

Because this is not a live serving system, query arrivals are simulated using a Poisson process: given a target QPS λ, inter-arrival times are drawn from Exp(1/λ). The resulting timestamp sequence determines how queries fall into each time window. Varying λ explores the spectrum from low load (small batches, little batching benefit) to high load (large batches, high throughput but growing queue delay).

### Latency Accounting

A query's total latency is decomposed as:

> *total latency = queue delay + batch execution time*

where **queue delay** is the time spent waiting for the window to close, and **batch execution time** is the wall-clock time of the entire batch (not divided by batch size, since each query must wait for the full batch to complete). Reporting this decomposition explicitly shows how much latency comes from scheduling overhead versus actual compute.

---

## 5. Proposal Assessment

### What's Strong

**The framing is correct.** "We don't change the index, only the execution layer" is a clean, defensible contribution scope. It's narrow enough to execute in 5 weeks.

**Time-window batching is well-specified.** The dual-trigger flush, Poisson arrival simulation, and latency decomposition (queue delay + execution time) show clear thinking. References [6] (Cooperative Scans) and [7] (Shared Workload Optimization) independently validate this "buffer and share" principle in classical database systems.

**The hypothesis is testable and falsifiable.** Cluster-based batching is predicted to win on clustered workloads and lose on random ones. References [2] (Milvus query blocks) and [4] (IVF-RaBitQ GPU batching) both confirm the effect is real — this is not a long shot.

---

### Risks

#### Risk 1 — Faiss already batches internally

`index.search(queries, k)` where `queries` is a matrix does not loop sequentially — Faiss uses OpenMP to parallelize across queries internally. The gain measured by time-window batching could be Faiss's internal threading rather than the scheduling logic. Two ways to handle this:

- **Disable Faiss threading** to isolate the scheduler's effect:
  ```python
  faiss.omp_set_num_threads(1)
  ```
- **Acknowledge it explicitly** and frame the contribution as "how to form the input batch", not "how to parallelize within it."

Either is valid, but the choice must be made deliberately and stated in the paper.

#### Risk 2 — Cluster-based batching is underspecified

The time-window section has concrete parameters (Δt, `max_batch_size`, sweep ranges). The cluster-based section is still vague. Before implementation, define:

- **Overlap metric:** exact same probe lists? Jaccard threshold? minimum shared list count?
- **Grouping algorithm:** greedy assignment? graph coloring by shared lists?
- **Latency of the grouping step:** the centroid-lookup pass (`quantizer.search`) adds overhead that must be accounted for in the latency decomposition.

Without this, it is hard to implement or evaluate fairly.

#### Risk 3 — Effect size may be small on in-memory CPU data

The batching benefit is largest when data loading is the bottleneck (disk I/O, DRAM bandwidth). With everything in DRAM and Faiss's SIMD already engaged, the scheduler's marginal gain may be 10–20% rather than 2×. Reference [4] shows dramatic gains on GPU precisely because memory bandwidth is the bottleneck there; the in-memory CPU setup is a harder case.

This is not fatal — a 10–20% throughput gain at the same recall is a valid result — but expectations in the introduction should be tempered and large speedups should not be promised.

---

### Advice

1. **Fill in the introduction's related works gap.** The `(need to search similar works)` note should cite [2], [4], and [6] directly. Those works are already documented in Section 1.

2. **Add implementation detail for cluster-based batching** comparable to the time-window section: overlap metric, grouping algorithm, and handling for queries that don't fit any group.

3. **State the null hypothesis explicitly.** "If batching provides no benefit on in-memory CPU data, we expect all three strategies to perform similarly." Acknowledging this upfront makes a negative result a valid finding rather than a failure.

4. **Run a microbenchmark early.** Before full experiments, measure cache miss rate with `perf stat -e cache-misses,cache-references` on a batched vs. sequential run. If cache misses drop measurably, the throughput story will follow.

---

## 5. Others

### Evaluation Advice

#### What is throughput?

**Throughput** = the number of queries the system can process per unit of time, typically measured in **queries per second (QPS)**. It answers: "how much work can the system handle?"

This is distinct from **latency**, which measures how long a single query takes. A system can have high throughput but also high latency (e.g., large batches finish many queries at once, but each query waits longer before the batch starts executing).

The fundamental trade-off in this project: **batching increases throughput (more queries share work) but may increase per-query latency (queries wait to form a batch).**

#### Recommended Metrics

| Metric | What it measures | How to collect |
|--------|-----------------|----------------|
| **Average latency** | Typical query response time | Time from query arrival to result return, averaged over all queries |
| **P95 / P99 latency** | Tail latency; worst-case user experience | Sort all query latencies, take the 95th/99th percentile |
| **Throughput (QPS)** | System capacity | Total queries completed / total wall-clock time |
| **Recall@k** | Search accuracy | Fraction of true top-k neighbors found; must stay constant across strategies to ensure fair comparison |
| **Cache miss rate** | Validates cache locality claim | Collect via `perf stat -e cache-misses,cache-references` on Linux |
| **Instructions per query** | Validates SIMD utilization | Collect via `perf stat -e instructions` |

#### Tips

- **Fix recall first.** Make sure all three strategies produce the same recall@k. If batching changes result quality, the throughput comparison is unfair.
- **Sweep parameters.** For time-window batching, vary Δt. For cluster-based batching, vary the grouping threshold. Plot throughput vs. latency as a Pareto curve.
- **Warm up.** Run several throwaway iterations before measuring to fill caches and stabilize JIT/branch predictors.
- **Report variance.** Run each experiment multiple times and report mean ± std, not just a single number.
