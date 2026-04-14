# From Faiss to Custom Engine: Why Scheduling Was Invisible

## 1. The Problem: Three Schedulers, Zero Difference

We implemented three query scheduling strategies on top of Faiss's `IndexIVFFlat`:

- **Sequential**: process one query at a time
- **Time-window batching**: accumulate queries within a time window, flush as a batch
- **Cluster-based batching**: group queries by centroid overlap before execution

All three strategies called `faiss.IndexIVFFlat.search()` with `omp_set_num_threads(1)` to isolate scheduling effects. The result: **no measurable performance difference** between the three strategies.

## 2. Root Cause: Faiss Neutralizes Scheduling

### Faiss's query-major, stateless architecture

Each call to `index.search(1_query, k)` is a completely independent operation:

```
index.search(query_A, k):
    find nprobe nearest centroids → load & scan those lists → return top-k → done

index.search(query_B, k):
    find nprobe nearest centroids → load & scan those lists → return top-k → done
```

There is no state carried between calls. Faiss does not know that it just scanned list #42 for query A when processing query B. It has no concept of cross-query context.

### The information loss boundary

The scheduler may know that queries A, B, and C all need list #42 and deliberately send them consecutively. But this information is **lost at the `index.search()` API boundary** — Faiss's interface accepts a single query (or a batch it processes query-by-query internally) and returns results. The "these queries share lists" signal that the scheduler carefully constructed simply does not pass through.

```
Scheduler layer:  "A, B, C share list #42 — send them together!"
        │
        ▼
   index.search()  ← information boundary
        │
Faiss internals:   process each query independently, scan lists from scratch
```

No matter how cleverly we arrange queries at the scheduler level, Faiss treats each one as if it were the first query ever seen.

### Why not just use batch calls?

Calling `index.search(batch_of_100_queries, k)` does not help either. With single-threaded execution, Faiss internally loops over each query in the batch and scans lists independently per query. The batch call is essentially a convenience wrapper around N independent single-query searches — it does not merge or share list accesses across queries.

## 3. Attempt: Isolating the Scheduling Variable

### Motivation

The Phase 1 result raises a question: is scheduling truly irrelevant, or is its effect being masked by the engine? To test this, we need an engine where the scheduler's decisions have a **direct, guaranteed** path to influence execution — rather than relying on implicit CPU cache effects that may or may not materialize.

Our approach: build a custom IVF engine that changes the scanning order from query-major to list-major, so that the scheduler's grouping decisions directly determine how many redundant list loads are eliminated.

### Implementation: `CustomIVFIndex`

We built `engine/custom_index.py` with two search methods:

**`search_one(query, k, nprobe)`** — per-query scanning (used by sequential scheduler):

```python
for each of the query's nprobe lists:
    load list → compute distances → collect candidates
return top-k
```

**`search_batch_per_list(queries, centroid_ids, k)`** — per-list scanning (used by batching schedulers):

```python
# Build inverted mapping: list_id → [query indices that need it]
for list_id, q_indices in list_to_queries.items():
    vecs = inverted_lists[list_id]          # load list ONCE
    for q_idx in q_indices:                 # list data stays in cache
        compute distances(vecs, queries[q_idx])
return top-k for each query
```

The key difference: in `search_batch_per_list`, each inverted list is loaded exactly once and used for all queries that need it. The scheduler's grouping decisions directly determine how many queries share each list load.

### What the custom engine changes

| | Faiss `index.search()` | Custom `search_batch_per_list()` |
|---|---|---|
| Input | single query (or loop of singles) | batch of queries + their centroid assignments |
| Cross-query info | lost | preserved |
| List scanning | per-query, independent | per-list, shared across queries |
| Scheduling effect | neutralized | directly impacts list reuse |

### Limitations and open questions

**The engine is not a pure "isolation" of the scheduling variable.** By switching from query-major to list-major scanning, we changed both the scheduling strategy and the scanning algorithm simultaneously. The observed performance difference is a joint effect of the two — we cannot cleanly attribute it to scheduling alone.

**This engine is not a production design.** List-major scanning is not used in mainstream vector databases (Faiss, Milvus, ScaNN, etc.). It sacrifices parallelism and per-query latency for batch-level list reuse. We use it here as an experimental apparatus to amplify the scheduling effect, not as a proposed replacement for query-major engines.

**All three strategies now share the same code path.** Sequential, time-window, and cluster-based all route through `quantizer_search` + `search_batch_per_list`, with sequential degenerating to batch_size=1. This ensures the only variable is the scheduling strategy, not the underlying search implementation.

### An unresolved question: why does query-major show no difference?

In principle, scheduling order should matter even in query-major engines through implicit CPU cache effects. If queries A and B both probe list #42, processing them consecutively should keep list #42 in L3 cache for query B. Our hypothesis for why this did not show up on SIFT1M:

- Each inverted list ≈ 2MB, nprobe=8 → per-query working set ≈ 16MB
- Modern L3 cache ≈ 16–32MB → the working set largely fits in cache regardless of query order
- The cache reuse signal is real but too weak to measure at this dataset scale

**This remains unverified.** We have not run experiments on larger datasets (e.g., 10M+ vectors) where cache pressure would be higher, nor have we collected hardware-level cache miss statistics (`perf stat`) to confirm whether scheduling order affects cache behavior on SIFT1M. These would be valuable follow-up experiments.

## 4. How Scheduling Strategies Map to the Custom Engine

| Scheduler | Search method | List reuse |
|---|---|---|
| Sequential | `search_batch_per_list(1 query)` | Degenerate batch of 1 — no cross-query sharing |
| Time-window | `search_batch_per_list(batch)` | Batch members share list loads |
| Cluster-based | `search_batch_per_list(group)` | Grouped by centroid overlap → maximized sharing |

All three strategies use the same custom engine and the same code path (`quantizer_search` + `search_batch_per_list`) — no Faiss `index.search()` calls remain. The only variable is how the scheduler assembles the set of queries passed to `search_batch_per_list`; sequential simply passes one query at a time.

## 5. Experimental Narrative

The two-phase experiment tells a complete story:

### Phase 1: Faiss baseline (batch_size=1 for all strategies)

- All three schedulers call `index.search(1 query, k)` independently
- Result: no performance difference
- Conclusion: **Faiss's stateless, query-major architecture makes external scheduling invisible**

### Phase 2: Custom engine

- All three strategies use `search_batch_per_list()` through the same code path
- Sequential passes one query at a time (batch_size=1, no cross-query sharing)
- Batching strategies pass larger batches (cross-query list sharing)
- Result: batching strategies outperform sequential; cluster-based grouping further improves over time-window
- Conclusion: **when the engine is batch-aware, scheduling strategy directly impacts performance through list reuse**

### Combined conclusion

Scheduling strategy and engine architecture are coupled. The benefit of intelligent scheduling only materializes when the engine is designed to exploit cross-query structure. This motivates co-designing schedulers and search engines rather than treating them as independent layers.

## 6. Why Faiss Doesn't Use List-Major Scanning

Faiss's query-major design is not a mistake — it optimizes for different goals:

| | Query-major (Faiss) | List-major (our engine) |
|---|---|---|
| Parallelism | Each query independent → trivial OpenMP parallelization | Queries share state → requires synchronization |
| Latency | Single query returns immediately | Must wait for entire batch to complete |
| Memory | One top-k heap per query, discard on completion | Accumulation buffers for all queries in batch |
| Best for | Online serving, multi-threaded, GPU | Single-threaded batched workloads with query overlap |

Our engine demonstrates the scheduling effect in a controlled single-threaded setting. In production, the tradeoff between parallelism and list reuse remains an open design question.
