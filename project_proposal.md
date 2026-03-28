# Query Scheduling Optimization in IVF-based Vector Search

Yujia Qian — yjqian19@mit.edu\
Xiangyu Guan — xiang949@mit.edu

## Abstract

Vector search has become a foundational building block for AI applications such as retrieval-augmented generation and semantic search. To serve these workloads efficiently, most optimizations target the index structure. In contrast, we focus on the execution layer: given an IVF index, can we improve throughput by reorganizing how concurrent queries are batched and dispatched? We study three strategies—sequential execution, time-window batching, and cluster-based batching—and compare their trade-offs in latency, throughput, and hardware efficiency on a single machine.

## Introduction
**(need to search similar works & hypothesis)**

Vector search is a core component of modern AI applications. Most optimization efforts target indexing structures (e.g., HNSW, IVF variants), but less attention has been paid to how queries are scheduled at runtime.

This project asks: **can we improve performance by reorganizing query execution, without changing the index?**

We build on IVF (Inverted File Index), which partitions vectors into clusters (inverted lists). Each query probes only a few lists, and different queries often probe overlapping ones. This means queries can share loaded data if executed together.

Batching does not reduce total computation — the work stays at roughly `number of queries × vectors scanned`. But it improves how that work is organized:

- **Less redundant data access** — shared inverted lists are loaded once
- **Better cache locality** — data stays in cache across queries
- **SIMD parallelism** — multiple distance computations are vectorized together

We compare three strategies: sequential execution (baseline), time-window batching (group by arrival time), and cluster-based batching (group by shared inverted lists), evaluating latency, throughput, and efficiency on a single machine.

## Methodology

### Sequential Execution
Each query is processed independently: it finds the nearest centroids, scans the corresponding inverted lists, computes distances, and returns the top-k results. This serves as the baseline with no cross-query optimization.

### Time-window Batching
Queries arriving within a fixed time window Δt are collected into a batch and executed together. This allows multiple queries to share vectorized computation, improving hardware utilization without requiring any knowledge of query content.

### Cluster-based Batching
Queries are grouped by the inverted lists they access. Queries that probe overlapping clusters are placed in the same batch so that each list is loaded and scanned once, then reused across all queries in the group. This strategy is more selective than time-window batching but requires an extra grouping step.

## Evaluation

- Latency (average and tail, e.g. P95)
- Throughput (queries per second)
- Precision / recall@*k*

We expect cluster-based batching to yield the highest throughput under clustered query workloads where probe-list overlap is high, while time-window batching should offer a better latency-throughput trade-off under random workloads where the grouping overhead outweighs the sharing benefit.

## Data
**(need to search data & benchmark)**

We use a standard vector search benchmark dataset:
- 10k–100k base vectors
- Corresponding query set

Workloads include:
- Random queries
- Clustered queries

All data is assumed to reside in memory.

## Tasks

- **Core engine:** IVF index, in-memory data, and correct single-query search as the baseline.
- **Schedulers:** Sequential, time-window, and cluster-based execution paths over the same engine.
- **Evaluation:** Benchmark workloads, metrics, and comparison plots.

## Timeline

### Phase 1: Now → Apr 14, Mid-term (2 weeks)
- Build IVF-based retrieval system
- Load dataset and benchmark
- Validate single-query execution

### Phase 2: Apr 14 → May 3 (3 weeks)
- Implement scheduling strategies
- Iteratively refine
- Run experiments and compare performance

### Phase 3: May 3 → May 7 (1 week)
- Compose the report and prepare the video

## Deliverables
- A working system
- Measured performance differences
- A clear explanation of results
