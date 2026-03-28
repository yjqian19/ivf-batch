# Reference Notes

## 1. Related Works

> TODO: search and fill in related papers / systems.

**Directions to explore:**

- **IVF indexing**: Faiss (Johnson et al., 2019) — the standard IVF implementation. Understand how `nprobe` affects recall/latency trade-off.
- **Batch-aware vector search**: look for papers on batched ANN queries, e.g., query batching in Faiss GPU, or Milvus/Vearch system papers that discuss concurrent query execution.
- **Query scheduling in databases**: classical work on shared scans (e.g., cooperative scans in column stores) is conceptually related — multiple queries sharing a single pass over data.
- **SIMD-optimized distance computation**: papers on vectorized distance kernels (e.g., SIMD-based inner product / L2 in Faiss, ScaNN).
- **Cache-aware algorithms**: work on cache-oblivious or cache-conscious data access patterns in nearest-neighbor search.

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
