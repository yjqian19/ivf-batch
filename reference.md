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

## 3. Others

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
