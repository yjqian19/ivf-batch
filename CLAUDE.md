# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project comparing three query scheduling strategies for IVF-based vector search without modifying the index structure:
1. **Sequential** — baseline, one query at a time
2. **Time-window batching** — group queries by arrival time using a dual-trigger flush (time Δt OR max batch size)
3. **Cluster-based batching** — group queries by shared inverted lists they probe

## Setup

```bash
uv init                        # initialize project (if not already done)
uv add faiss-cpu numpy         # or faiss-gpu if CUDA available
uv run python main.py          # run scripts
```

Download datasets:
- **SIFT1M** (primary): http://corpus-texmex.irisa.fr/ — `sift_base.fvecs`, `sift_query.fvecs`, `sift_groundtruth.ivecs`
- **GloVe-100** (secondary): https://ann-benchmarks.com/

## Key Implementation Details

### Core Engine
- Use `faiss.IndexIVFFlat` with `n_clusters=256`, `k=10`; vary `nprobe` to control recall-speed tradeoff
- All vectors must be `np.float32`
- Fix `faiss.omp_set_num_threads(1)` when benchmarking schedulers to isolate scheduling effects from Faiss's internal parallelism
- Data loaders for `.fvecs`/`.ivecs` format are documented in `reference.md` Section 3

### Scheduler Architecture
Schedulers sit **on top of** Faiss — they change *when* and *how many* queries are sent to `index.search`, not what happens inside it. All three paths must return identical results for the same queries.

### Time-Window Batching
- Dual-trigger flush: whichever fires first — time limit Δt OR `max_batch_size`
- Simulate arrivals with Poisson process at target QPS λ (inter-arrival times ~ Exp(1/λ))
- Latency = queue delay (waiting for window) + batch execution time
- Sweep: Δt ∈ {0.5, 1, 2, 5, 10, 20, 50} ms, `max_batch_size` ∈ {32, 64, 128, 256}

### Cluster-Based Batching
- Inspect each query's top-`nprobe` centroids via `quantizer.search` (the quantizer sub-index inside `IndexIVFFlat`)
- Group queries by centroid overlap (define overlap metric: e.g., Jaccard threshold or minimum shared list count)
- Account for the centroid-lookup overhead in latency decomposition

## Evaluation
- **Fix recall first**: all three strategies must achieve the same Recall@10 (target ≥0.90 on SIFT1M with `nprobe=8`) before comparing throughput
- Metrics: average latency, P95/P99 latency, throughput (QPS), Recall@k, cache miss rate (`perf stat -e cache-misses,cache-references` on Linux)
- Two workloads: **random queries** (worst case for cluster-based batching) and **clustered queries** (best case)
- Warm up before measuring; report mean ± std over multiple runs

## Timeline
- Phase 1 (now → Apr 14): Core engine + data loading + single-query validation
- Phase 2 (Apr 14 → May 3): Implement all 3 schedulers, run experiments
- Phase 3 (May 3 → May 7): Report + video

## Reference Files
- `documentation/reference.md` — detailed implementation notes, dataset info, risk analysis, evaluation advice
- `README.md` — abstract, methodology, evaluation plan
