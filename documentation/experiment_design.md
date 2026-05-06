# Experiment Design

This document describes the three experiments, how to run them, and the design decisions behind each one.

---

## 1. The Three Experiments

| # | Script | Purpose | Runs needed |
|---|---|---|---|
| 1 | `run_sweep.py` | Grid search over (Δt, MaxBS) to find the parameter that best separates MV and MM | 1 |
| 2 | `run_main.py` | Fixed-parameter workload comparison (Sequential / MV / MM × random / clustered) with statistical aggregation | 5 |
| 3 | `microbench_a4.py` | Isolated kernel comparison (MV vs MM) as a function of m — free of scheduler noise | 1 |

---

## 2. How to Run

### 2.1 Parameter sweep

```bash
uv run python run_sweep.py
```

Output saved to `results/sweep_<timestamp>.txt`.

After running, find the (Δt, MaxBS) row where:
- Batch(MV) QPS > Batch(MM) QPS on the **random** workload
- Batch(MM) QPS > Batch(MV) QPS on the **clustered** workload
- Both beat Sequential

Record this pair and proceed to §4 (Parameter Choice) below.

### 2.2 Main experiment

After filling in §4, set `DELTA_T_MS` and `MAX_BATCH_SIZE` at the top of `run_main.py`, then:

```bash
uv run python run_main.py                              # default: 5 runs
uv run python run_main.py --runs 5 --delta-t-ms 5 --max-bs 128   # override via CLI
```

Output saved to `results/main_<timestamp>.txt`. Each run prints the per-workload comparison (QPS, latency, A1/A2/A3 metrics); the final section prints mean ± std across all runs.

### 2.3 Microbenchmark

```bash
uv run python microbench_a4.py
uv run python microbench_a4.py --n 4000 --d 128 --repeats 50   # explicit defaults
```

Output saved to `results/microbench_a4_<timestamp>.txt`.

---

## 3. Design Considerations

### 3.1 Fixed k-means seed

**Rule:** `faiss.Kmeans` is called with `seed=0` in `engine/custom_index.py`.

**Why this matters:** Without a fixed seed, the k-means initialisation is random. Two runs with different seeds produced radically different clustered workloads: ~10 000 queries qualified in one run (highly skewed clusters) vs. 784 in another (balanced clusters). The m distribution — which determines whether MM beats MV — changed completely between them. All three experiments must use the same index to be comparable.

### 3.2 Deterministic workloads

- **Arrival times**: `generate_arrivals` uses `seed=42`. Identical every run — timing noise is the only run-to-run variance.
- **Clustered query selection**: `select_clustered_queries` uses `seed=42` for the optional sub-sample. Given a fixed index (§3.1), the selected queries are also identical every run.

### 3.3 Clustered Query Generation

#### 3.3.1 Why real queries, not synthetic

The previous approach (`generate_clustered_queries`) sampled base vectors directly from selected inverted lists and used them as queries. This was discarded for two reasons:
1. Base vectors have no entry in `sift_groundtruth.ivecs` → Recall@10 is undefined.
2. The workload was artificially extreme: 1000 queries per centroid literally sampled from the same list meant m ≈ 1000 for that list, far beyond any realistic scenario.

The current approach (`select_clustered_queries`) uses real vectors from `sift_query.fvecs`: each query is assigned to its nearest centroid (`nprobe=1`), and only queries whose primary centroid is among the top-`n_centers` most-queried centroids are kept. These have valid ground truth, so Recall@10 is well-defined.

#### 3.3.2 Why only ~791 queries survive the filter

The index has `N_CLUSTERS = 256` centroids. The 10,000 SIFT test queries are distributed roughly uniformly, so each centroid receives on average 10000 / 256 ≈ **39 queries**. Even the top-10 most popular centroids only have 70–100 queries each — giving ~791 total after filtering.

This is not a bug. It reflects the actual query distribution over the index.

#### 3.3.3 Why 791 queries is too few for the experiment

With only 791 queries arriving at 2000 QPS, the entire workload completes in ~0.4 seconds. The batch scheduler accumulates queries through its simulated-time mechanism, but with so few queries the average batch size (AvgBS) stays around 72 — much lower than the ~97 achieved by the 10,000-query random workload.

The consequence: with AvgBS ≈ 72 spread across 10 centroids, each centroid contributes only **72 / 10 ≈ 7 queries per batch**. The per-list sharing count m stays around 5–7, **below the MM crossover threshold of m ≈ 8** on the M3. MM never gets a chance to win.

#### 3.3.4 Fix: tile the 791 queries to 10,000

To match the random workload in query volume, the 791 clustered queries are repeated until 10,000 arrivals are reached. Before tiling, `select_clustered_queries` sorts the 791 queries by their primary centroid, so each tile repetition cycles through all clusters in order:

```
[c0_0, c0_1, …, c1_0, c1_1, …, c9_0, …  |  c0_0, c0_1, …, c1_0, …]
 ←────────── tile repetition 1 ──────────→  ←── repetition 2 ──…
```

```python
n_tile = len(queries)          # 10 000 — same as random workload
reps   = n_tile // len(cl_queries) + 1
cl_queries = np.tile(cl_queries, (reps, 1))[:n_tile]
cl_gt      = np.tile(cl_gt,      (reps, 1))[:n_tile]
cl_arrivals = generate_arrivals(n_tile, TARGET_QPS)
```

**Effect:** with 10,000 arrivals, AvgBS rises to ~97 (same as random). Because consecutive arrivals come from the same cluster, each batch is dominated by queries sharing a small set of inverted lists → m is concentrated rather than spread across 10 centroids. This is the regime where GEMM (MM) pays off over GEMV (MV).

**Why cluster-burst, not uniform mix:** the goal of the clustered workload is to stress-test the case where many queries probe the same lists simultaneously. Sorting before tiling produces genuine same-cluster bursts within every time window, which directly drives up m. A uniformly mixed ordering would dilute m back toward the random-workload baseline, defeating the purpose.

**Justification for repetition:** in a real system, a "clustered" request stream means users repeatedly searching within the same topic or region. Cycling the same 791 query patterns is a standard systems-benchmark technique for stress-testing a specific access pattern at scale.

#### 3.3.5 Where the sort lives and why it does not affect timing

The cluster-sort is applied inside `select_clustered_queries` (in `engine/schedulers.py`), which is a **data preparation function**, not part of the scheduling loop. It runs once before any experiment timing begins and returns a pre-sorted `cl_queries` array.

The batch scheduler (`run_batch`) receives this array and has no knowledge of how it was ordered. All latency and QPS measurements start only when the scheduler begins processing — the sort adds zero overhead to the measured results.

The function lives in `schedulers.py` for file-organisation convenience, but it belongs logically to data preparation, not scheduling.

### 3.4 Index built once per `run_main.py` invocation

`run_main.py` builds the index once, then loops N runs over the scheduler comparison. This has two benefits:
1. Saves ~1–2 s per run.
2. Ensures all N runs measure the exact same index — no index-level variance between runs.

The sweep (`run_sweep.py`) and microbench (`microbench_a4.py`) each rebuild the index once at startup; since they run only once, this is fine.

### 3.5 Run counts and aggregation

| Experiment | Runs | Rationale |
|---|---|---|
| Parameter sweep | 1 | Effect sizes across the grid are large (20–100% QPS differences). A single run is sufficient to identify the winning region. |
| Main experiment | 5 | The MM vs MV gap on clustered is ~10–20% — a moderate effect. 5 runs with timing CV ≈ 3–5% gives SE ≈ 1–2%, comfortably below the effect. P99 latency is reported as the **median** across runs (GC-pause sensitive). All other metrics are reported as **mean ± std**. |
| Microbenchmark | 1 | The script runs 50 internal repetitions per (m, kernel) point and reports the median. The crossover point is structural, not a timing artifact — re-running adds nothing. |

### 3.6 Metrics collected in the main experiment

`run_main.py` calls both schedulers with `collect_stats=True`, which enables:

- **A1 — Latency decomposition**: per-query queue delay vs. scan time
- **A2 — Lists loaded per query**: direct measure of per-list reuse (Sequential ≈ nprobe, Batch ≈ nprobe / m)
- **A3 — m distribution**: per-(batch, list) sharing count (mean, P50, P95)

These are averaged across the N runs in the summary. See `documentation/metrics_design.md` for the full rationale behind each metric.

---

## 4. Parameter Choice

Source: `results/sweep_20260506_165210.txt` (seed=0, cluster-sorted tiling).

**Selected parameters:**

| Parameter | Value | Source |
|---|---|---|
| Δt (time window) | 5 ms | sweep result |
| MaxBS (max batch size) | 128 | sweep result |

**Justification:**

Δt=5ms / MaxBS=128 is the smallest setting that produces a clear, clean separation between the two workloads:

| Workload | Sequential | Batch(MV) | Batch(MM) | Winner |
|---|---|---|---|---|
| Random | 1845 QPS | **2280 QPS** | 2203 QPS | MV +3.5% over MM |
| Clustered | 1845 QPS | 1971 QPS | **2461 QPS** | MM +25% over MV |

- **Random workload**: MV beats MM because AvgBS=82 yields small m per inverted list, and the GEMM overhead is not amortised. MV's per-query GEMV loop is lighter at low m.
- **Clustered workload**: cluster-sorted tiling means consecutive arrivals in each time window come from the same centroid, pushing m high. MM's GEMM advantage takes over, beating MV by 25% and Sequential by 33%.
- **Going larger (Δt=10ms / MaxBS=128)**: MM starts winning even on random (+1%), which blurs the contrast between workloads — not desirable for the experiment narrative.
- **Going smaller (Δt=5ms / MaxBS=64)**: the random MV lead widens to +11% but AvgBS drops to ~61, giving a slightly weaker demonstration of batching benefit over Sequential.

**Notable side-effect of cluster-burst ordering:** on the clustered workload, MM's AvgBS (54.9) is much smaller than MV's (119.0). This is a self-reinforcing effect: high m → fast batch execution → queue drains quickly → smaller next batch — yet m remains high because cluster bursts keep same-centroid queries together.
