# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research project comparing three query scheduling strategies for IVF-based vector search on a single machine, without modifying the index structure:

1. **Sequential** — baseline, one query at a time
2. **Time-window batching** — group queries by arrival time, dual-trigger flush (time Δt OR max batch size)
3. **Cluster-based batching** — group queries by shared inverted lists they probe

**Goal:** improve throughput and cache efficiency by reorganizing query execution, not the index.

## Setup

```bash
uv init                        # initialize project (if not already done)
uv add faiss-cpu numpy         # or faiss-gpu if CUDA available
uv run python run_experiments.py
```

Download datasets:
- **SIFT1M** (primary): http://corpus-texmex.irisa.fr/ — `sift_base.fvecs`, `sift_query.fvecs`, `sift_groundtruth.ivecs`
- **GloVe-100** (secondary): https://ann-benchmarks.com/

## Evaluation Targets

- Metrics: throughput (QPS), average/P95/P99 latency, Recall@10
- All three schedulers must achieve the same Recall@10 before comparing throughput
- Two workloads: **random queries** (worst case for cluster-based) and **clustered queries** (best case)

## Timeline

- Phase 1 (→ Apr 14): Core engine + data loading + single-query validation ✓
- Phase 2 (Apr 14 → May 3): Implement all 3 schedulers, run experiments
- Phase 3 (May 3 → May 7): Report + video

## Reference Files

- `documentation/project_proposal/project_proposal.md` — original proposal
- `documentation/reference.md` — implementation notes, dataset info, evaluation advice
- `documentation/yujia_recap.md` — current engine and scheduler implementation details
- `documentation/custom_engine_rationale.md` — rationale for custom engine over Faiss
