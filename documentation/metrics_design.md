# Additional Metrics Design

## 1. Goal

The midterm report makes **two** claims, not one:

1. **Batch (MV or MM) beats Sequential** — by reusing each inverted list across multiple queries, batching cuts memory traffic and amortizes BLAS call overhead.
2. **MM beats MV at large m** — once enough queries share a list, GEMM's higher arithmetic intensity overtakes GEMV.

The current results only report QPS, recall, and a couple of latency percentiles, which show *that* both effects exist but not *why*. This document picks the smallest set of additional metrics that explain the mechanism — for both comparisons — and that are realistic to run on a MacBook Pro M3.

---

## 2. Theory: why we expect what we observe

Before picking metrics, it helps to be explicit about *why* batching should help and *why* MM crosses over MV. The metrics in §4–§5 are then chosen to test these mechanisms one-to-one.

### 2.1 Why Batch beats Sequential — per-list reuse

In Sequential, each query independently traverses its `nprobe` inverted lists. If queries Q₁ and Q₂ both probe list L, list L is read from DRAM **twice** — once per query. In Batch, the scheduler groups queries by list, so L is read **once per batch** regardless of how many queries probe it.

The reason this matters:

- **IVF list scans are memory-bandwidth bound on CPU.** Each distance computation reads one 128-dim float32 vector (512 B) and does ≈256 FLOPs — arithmetic intensity ≈ 0.5 FLOP/byte, well below the M3's roofline ridge (≈0.7 FLOP/byte). The Faiss paper and "Bang for the Buck" both make this point explicitly.
- **Therefore per-query latency tracks bytes loaded per query.** If batching cuts list loads per query from `nprobe` to `nprobe / m` (where `m` = average queries sharing each list in a batch), throughput should improve by roughly factor m.

Secondary effect: each BLAS call has fixed dispatch overhead (kernel pick, threading, Python boundary). Fewer, larger calls amortize that overhead — a smaller win than the bandwidth one, but not zero.

### 2.2 Why MM beats MV at large m, loses to MV at small m

Both MV and MM operate on identical per-list batched data; they differ only in the kernel:

- **MV** — loop over the m queries, compute m independent GEMVs per list (`vecs @ q` for each q).
- **MM** — stack the m queries into a (m × d) matrix Q, compute one GEMM (`vecs @ Q.T`).

The arithmetic intensities differ in a critical way:

| Kernel | FLOPs | Bytes loaded (list of n vectors, dim d) | AI (FLOP/byte) |
|---|---|---|---|
| Per-query GEMV | 2·n·d | ≈ n·d·4 | **0.5** (constant in m) |
| Per-list GEMM | 2·n·d·m | ≈ n·d·4 (matrix dominates when n ≫ m) | **m / 2** (linear in m) |

GEMM's AI grows linearly in m: as m increases, the same loaded-once list does more arithmetic. In theory, GEMM crosses the M3 roofline ridge at m ≈ 1.4. In practice the crossover is much later (we observed AvgBS ≈ 120) because:

- **Materialization cost.** MM has to gather m query vectors into a contiguous (m × d) buffer per list. MV streams them from one persistent query buffer.
- **GEMM dispatch overhead.** BLAS GEMM has heavier per-call overhead than GEMV (tile selection, optional threading). At small m the overhead exceeds the AI advantage.

So **at small m, MM does the same arithmetic as MV with extra overhead → MV wins**. At large m, AI scaling dominates overhead → MM wins. Random queries give small m (≈4 in our run), clustered queries give large m — which is exactly what the §4 results show.

### 2.3 What this implies for metrics

The theory says observable throughput depends on three quantities. Each one becomes a metric in §4:

| Theoretical quantity | Predicts | Metric |
|---|---|---|
| **Per-list reuse factor m** | Batch-vs-Sequential gap, MM-vs-MV crossover point | A2 (lists loaded per query), A3 (m distribution) |
| **Per-list scan time** | Where in the pipeline batching's wins land | A1 (latency decomposition) |
| **Arithmetic intensity vs. m** | The kernel-level mechanism that drives MM > MV at large m | A4 (scan-throughput-vs-m microbench), A5 (analytical AI) |

This is why the candidate metrics below are chosen the way they are — each one tests a specific link in the theory, not just "more numbers."

---

## 3. What similar work measures

A quick survey of recent ANNS / vector-search papers and benchmarks, grouped by what's actually reported:

| Source | Setting | Metrics reported |
|---|---|---|
| **ANN-Benchmarks** ([docs](https://docs.weaviate.io/weaviate/benchmarks/ann)) | Standard CPU benchmark suite | QPS, recall@k, build time, index size; QPS-vs-recall Pareto curves |
| **Big-ANN-Benchmarks** ([NeurIPS'21 track](https://big-ann-benchmarks.com/neurips21.html)) | Billion-scale | QPS at fixed recall, hardware cost ($/QPS), memory footprint |
| **BANG** ([arXiv 2401.11324](https://arxiv.org/pdf/2401.11324)) | Billion-scale ANNS, single GPU | QPS, recall@k, latency, batch-size sweep |
| **AverSearch** ([arXiv 2504.20461](https://arxiv.org/abs/2504.20461)) | Graph-based ANNS, low-latency | QPS, P50/P95/P99 latency, intra-query parallelism breakdown |
| **"Bang for the Buck"** ([arXiv 2505.07621](https://arxiv.org/html/2505.07621v1)) | IVF/HNSW on cloud CPUs | QPS, $/QPS, AMD Zen4 vs Intel Sapphire Rapids — explicitly notes IVF is bandwidth-bound |
| **Faiss paper** ([arXiv 2401.08281](https://arxiv.org/html/2401.08281v2)) | Library overview | QPS, recall, batch-size sensitivity; states most Faiss indices are memory-bandwidth-limited, not compute-limited |
| **Elasticsearch simdvec** ([blog](https://www.elastic.co/search-labs/blog/elasticsearch-vector-search-simdvec-engine)) | CPU SIMD distance kernel | Speedup factor (1.7–4×) over Faiss baseline; calls out bulk scoring + prefetching as the mechanism |
| **NVIDIA cuVS IVF-PQ** ([blog](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/)) | GPU IVF | QPS, latency, batch-size sweep — frames small batches as GEMV, large batches as GEMM |

Two takeaways:

- **Hardware counters (cache misses, memory bandwidth) are not standard in ANNS papers.** None of the surveyed work runs `perf stat` as the headline argument. They report wall-clock QPS, latency, and recall, and explain the mechanism with batch-size sweeps and analytical reasoning. Our midterm's `perf stat` mention is therefore overkill — we can drop it without weakening the paper.
- **The standard "explain the mechanism" tool is a parameter sweep**, not a counter dump. Plotting QPS or scan-time vs. batch size (or vs. m, in our case) is what cuVS, simdvec, and the Faiss paper all do.

---

## 4. Candidate metrics

### Tier A — software-only, no special tooling

| # | Metric | Which claim it supports | How to measure |
|---|---|---|---|
| A1 | **Latency decomposition** (queue / scan / topk) per query | Both — separates "queueing cost" from "scan cost", which is essential when comparing Sequential (no queueing) to Batch (large queueing) | Add 4 `time.perf_counter()` markers in scheduler; aggregate per-query |
| A2 | **Lists loaded per query** (= total list scans / num queries) | Batch vs Sequential — direct measure of per-list reuse. Sequential ≈ nprobe; Batch ≈ nprobe / m | Counter inside `search_batch_per_list`; divide by query count |
| A3 | **Avg / P50 / P95 m per list per batch** | Both — m is the lever the whole story turns on | Log `len(query_ids)` for every list scan |
| A4 | **Scan throughput vs. m microbench** | MM vs MV — produces the crossover plot directly, free of scheduler noise | Standalone script: fix list size, vary m ∈ {1,2,4,…,256}, time MV and MM scans, plot ns / (query·vector) |
| A5 | **Arithmetic intensity (FLOP/byte)** | Both — compact analytical statement of why batching helps. AI = m·d·2 / (d·4) = m/2, so AI grows linearly with m | Analytical + measured GFLOP/s from A4 |
| A6 | **P99 latency** (already report P95) | Both — community convention; tail latency is what user-facing systems care about | Trivial extension of existing logging |

### Tier B — hardware counters

Realistic only as a sanity check, **not** as a headline result.

- **macOS native**: `xctrace record --template 'CPU Counters'` (Instruments CLI) reads M3 PMU counters via Apple's private `kperf` framework. Output is a `.trace` bundle, not plain text — needs scripted post-processing. Counter names differ from Linux `perf`. Realistic for an isolated microbench (the A4 kernels), painful for a full scheduler run.
- **Linux in Docker on Mac**: ❌ Does **not** work for hardware counters. Docker Desktop's Linux VM does not pass M3 PMU counters through, so `perf stat` inside reports software counters only (page-faults, context-switches). Confirmed by Apple Silicon dev community.
- **Linux on a real machine** (cloud VM, lab workstation): Works fully. This is what most ANNS systems papers do — they run on Linux servers and report wall-clock QPS, occasionally `perf stat` for kernel-level claims. None of the ANNS papers in §3 above relies on hardware counters as their main evidence.

**Conclusion on Linux**: not worth the setup. The argument we want to make is at the QPS / scan-time / arithmetic-intensity level, all of which run fine on the M3. Hardware counters would be needed only if a reviewer specifically demanded cache-miss numbers — then `xctrace` on the A4 microbench is enough, no Linux required.

### Tier C — skip

- Energy / power per query — our claim isn't about energy
- Per-core utilization — single-threaded by design
- Build time / index size — same index across all three schedulers, so it's a constant
- $/QPS — a cloud-cost framing that doesn't apply to a laptop study

---

## 5. The shortlist (pick these)

Metrics split into two groups by where they run.

### Primary experiment — instrumentation added to the existing scheduler run

These four require only software timers and counters inside the scheduler. No new scripts.

| Metric | Supports | Effort | Goes in report as |
|---|---|---|---|
| **A1 — latency decomposition** (queue / scan / topk) | Both | ~1 hr | Stacked-bar plot per scheduler per workload. Shows whether MM's latency win comes from shorter scans or less queueing. |
| **A2 — lists loaded per query** | Batch vs Sequential | <1 hr | One extra row in §4 results tables. Direct count of per-list reuse: Sequential ≈ nprobe, Batch ≈ nprobe / m. |
| **A3 — avg / P50 / P95 m per list per batch** | Both | <1 hr | One sentence per workload in §4.3. Connects A4's theoretical crossover to actual observed m — without this the reader cannot verify the clustered workload lands in MM's advantage zone. |
| **A6 — P99 latency** | Both | <15 min | Extra column next to P95 in §4 tables. Standard ANNS practice. |

### Secondary experiment — standalone microbenchmark script (decoupled from scheduler)

One script: fix list size n, sweep m ∈ {1, 2, 4, …, 256}, time MV and MM kernels repeatedly, take medians.

| Metric | Supports | Effort | Goes in report as |
|---|---|---|---|
| **A4 — scan throughput vs. m** | MM vs MV | ~2 hr | Line plot: ns/(query·vector) vs. m for MV and MM. Headline figure for the §4.3 crossover discussion. |

Note: A5 (arithmetic intensity) is dropped as a measured metric. True AI = FLOPs / Bytes requires actual bytes loaded from DRAM, which needs hardware counters (Tier B). The theoretical formula AI = m/2 FLOP/byte is a one-line analytical statement written into the report prose — no experiment.

Total estimated effort: primary ≈ 3 hr, secondary ≈ 2 hr.

---

## 6. What to change in the midterm report

In §5.5 (potential problems), replace:

> *Cache miss rate (`perf stat`) would directly validate the per-list reuse effect. Latency decomposition (queue delay vs. scan time) would show whether throughput gains come from better compute utilization or simply larger batches.*

with something like:

> *Three additional metrics will be added: (i) inverted-list loads per query, which directly counts the per-list reuse rate that distinguishes Batch from Sequential; (ii) a queue / scan / topk latency decomposition, which separates queueing cost from scan cost across schedulers; (iii) a scan-throughput-vs-m microbenchmark, which gives the MV-vs-MM crossover free of scheduler noise. Hardware counters (`perf stat` on Linux, `xctrace` on macOS) are deliberately omitted — recent ANNS work (Faiss, cuVS, Big-ANN-Benchmarks, "Bang for the Buck") explains mechanisms via batch-size sweeps and analytical arithmetic intensity, not PMU dumps.*
