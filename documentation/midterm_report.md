# Mid-Term Report: Query Scheduling Optimization in IVF-based Vector Search

Yujia Qian — yjqian19@mit.edu\
Xiangyu Guan — xiang949@mit.edu

---

## 1. Project Status

**Tasks**

| Task | Complete | Pending |
|---|---|---|
| Core engine | IVF index, data loading, single-query search | — |
| Schedulers | Sequential, Time-Window, Cluster-Based | Jaccard benchmark not yet run |
| Evaluation | SIFT1M parameter sweep, workload comparison (random vs. clustered) | GloVe-100, cache-miss measurement, multi-run statistics |

**Deliverables**

| Deliverable | Complete | Pending |
|---|---|---|
| A working system | all | — |
| Measured performance differences | First-round results on SIFT1M | GloVe-100, mean ± std |
| Final report and video | — | all |

---

## 2. Experiment Setup

- **Dataset:** SIFT1M (1M × 128-d, L2 distance)
- **Device:** MacBook Pro, Apple M3 Pro, 18 GB RAM
- **Parameters:** n_clusters=256, nprobe=8, k=10
- **Queries:** 10K queries from SIFT1M, arrival rate 2000 QPS (Poisson), tested under two workloads:
  - *Random* — queries sampled uniformly, probing largely disjoint inverted lists
  - *Clustered* — queries drawn from 10 spatial regions, with high centroid overlap across queries

  We test both because the benefit of cluster-based batching depends entirely on centroid overlap between concurrent queries — random is the worst case, clustered is the best case.

## 3. Results

### Random Queries

| Scheduler | Recall@10 | QPS | Avg Lat | P95 Lat |
|---|---|---|---|---|
| Sequential | 0.956 | 1994 | 0.5 ms | 0.7 ms |
| Time-Window | 0.956 | 2361 (+18%) | 154.4 ms (×309) | 216.7 ms (×310) |
| Cluster-Batch | 0.956 | 2023 (+1%) | 166.1 ms (×332) | 242.1 ms (×346) |

<sub>Peak-QPS configuration shown: Time-Window at Δt=20ms, MaxBS=256; Cluster-Batch at Δt=5ms, MaxBS=256.</sub>

All three schedulers achieve identical Recall@10=0.956, confirming correctness. Time-Window delivers the highest throughput at +18% over Sequential, but at a steep latency cost — average latency increases 309× from 0.5 ms to 154 ms. Cluster-Batch offers only marginal throughput gain (+1%) while incurring even higher latency, as primary-centroid grouping produces near-singleton groups (AvgGrp 1.1–1.7) on random queries, meaning grouping overhead accumulates with little sharing benefit.

### Clustered Queries

| Scheduler | Recall@10 | QPS | Avg Lat |
|---|---|---|---|
| Sequential | 0.956 | 2007 | 0.5 ms |
| Time-Window | 0.956 | 2329 (+16%) | 56.6 ms (×113) |
| Cluster-Batch | 0.956 | 2346 (+17%) | — |

<sub>Fixed config: Δt=5ms, MaxBS=128.</sub>

Under clustered queries, Cluster-Batch's AvgGrp jumps from 1.3 to **63.7** and group count drops from 7436 to **157**, nearly eliminating grouping overhead and bringing it to 2346 QPS — slightly edging Time-Window (2329 QPS). This confirms the core hypothesis: cluster-based batching pays off when queries share spatial locality, and the two strategies converge in performance when batching conditions are favorable.

Full parameter sweep results are in `documentation/parameter_sweep.md`.

---

## 4. Potential Problems

**Cluster-based grouping degenerates on random workloads.**
Primary-centroid grouping produces near-singleton groups (AvgGrp 1.1–1.7) on random SIFT queries, making Scheduler 3 slower than baseline. We plan to benchmark the already-implemented Jaccard-similarity grouping to test whether partial overlap relaxes this. If it does not help, we will document this as a fundamental workload-sensitivity result.

**Severe latency cost at peak throughput for Time-Window.**
Peak QPS (2361) requires average latency of 154 ms — a 300× increase over sequential. We will identify Pareto-optimal configurations (e.g., Δt=0.5ms, MaxBS=128 gives 2125 QPS at 17 ms) and frame the sweep as an operating-point selection guide.

**Cache-miss measurement unavailable on macOS.**
`perf stat` is Linux-only. We will use timing ratios (per-list scan time vs. batch size) as a proxy for cache reuse, or note this as a limitation if direct measurement is infeasible before the deadline.

**Single-run results.**
All current numbers are from one run. We will add a repeat loop (3–5 runs) before the final report to report mean ± std.

---

## 5. Updated Timeline

| Phase | Dates | Tasks |
|---|---|---|
| Phase 1 | Now → Apr 14 ✓ | Core engine, data loading, single-query validation |
| Phase 2 | Apr 14 → May 3 | Jaccard grouping benchmark; GloVe-100 experiments; cache-miss proxy measurement; multi-run statistics; latency decomposition for Scheduler 3 |
| Phase 3 | May 3 → May 7 | Report and video |

---

## 6. Division of Work

*(To be filled in.)*

---

## 7. Questions
