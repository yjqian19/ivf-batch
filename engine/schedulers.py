import numpy as np
import time
from collections import defaultdict


def generate_arrivals(n_queries, qps, seed=42):
    """Simulate Poisson arrivals at target QPS. Returns cumulative arrival times in seconds."""
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(1.0 / qps, size=n_queries)
    return np.cumsum(inter_arrivals)


# ── Sequential ───────────────────────────────────────────────────────────────

def run_sequential(index, queries, k=10, nprobe=8, collect_stats=False):
    """One query at a time — batch_size=1, MV scan.

    If collect_stats=True, the returned stats dict additionally contains:
      centroid_times : (n,) float64  — per-query centroid lookup time (s)
      scan_times     : (n,) float64  — per-query scan+topk time (s)
      queue_delays   : (n,) zeros    — no queueing in sequential
      list_loads_per_query : float   — always ≈ nprobe
      m_values       : (n*nprobe,) int  — always 1 (one query per list)
    """
    n = len(queries)
    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    query_times = np.empty(n)
    centroid_times = np.empty(n)
    scan_times = np.empty(n)

    total_list_loads = 0
    all_m_values = []

    t_wall = time.perf_counter()
    for i in range(n):
        q = queries[i:i+1]

        t0 = time.perf_counter()
        centroid_ids = index.quantizer_search(q, nprobe)
        t1 = time.perf_counter()

        batch_stats = {} if collect_stats else None
        D, I = index.search_batch_per_list(q, centroid_ids, k, mode="mv",
                                           _stats=batch_stats)
        t2 = time.perf_counter()

        centroid_times[i] = t1 - t0
        scan_times[i] = t2 - t1
        query_times[i] = t2 - t0
        all_ids[i] = I[0]
        all_dists[i] = D[0]

        if collect_stats and batch_stats:
            total_list_loads += batch_stats["list_loads"]
            all_m_values.extend(batch_stats["m_values"])

    wall_time = time.perf_counter() - t_wall

    stats = {
        "query_times": query_times,
        "centroid_times": centroid_times,
        "scan_times": scan_times,
        "queue_delays": np.zeros(n),
        "wall_time": wall_time,
        "qps": n / wall_time,
    }
    if collect_stats:
        stats["list_loads_per_query"] = total_list_loads / n
        stats["m_values"] = np.array(all_m_values, dtype=np.int64)
    return all_ids, all_dists, stats


# ── Batch(MV) / Batch(MM) — Time-Window Batching ────────────────────────────

def run_batch(index, queries, arrival_times, delta_t_ms, max_batch_size,
              k=10, nprobe=8, scan_mode="mv", collect_stats=False):
    """Time-window batching with dual-trigger flush.

    scan_mode : "mv"  — per-query GEMV inside search_batch_per_list  [Batch(MV)]
                "mm"  — per-list GEMM inside search_batch_per_list   [Batch(MM)]

    If collect_stats=True, the returned stats dict additionally contains:
      centroid_times       : (n,) float64  — centroid time each query "paid"
                             (all queries in a batch share the same value)
      scan_times           : (n,) float64  — scan+topk time each query "paid"
      list_loads_per_query : float  — avg unique list loads per query
      m_values             : array of per-(batch,list) sharing counts
    """
    n = len(queries)
    delta_t = delta_t_ms / 1000.0

    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    latencies = np.empty(n)
    queue_delays = np.empty(n)
    centroid_times = np.empty(n)
    scan_times = np.empty(n)
    batch_sizes = []

    total_list_loads = 0
    all_m_values = []

    sim_time = 0.0
    i = 0
    t_wall = time.perf_counter()

    while i < n:
        sim_time = max(sim_time, arrival_times[i])
        window_open = sim_time
        deadline = window_open + delta_t
        batch_idx = []

        while i < n and len(batch_idx) < max_batch_size:
            if arrival_times[i] <= deadline:
                batch_idx.append(i)
                i += 1
            else:
                break

        flush_time = arrival_times[batch_idx[-1]] if len(batch_idx) >= max_batch_size else deadline
        sim_time = max(sim_time, flush_time)

        batch_q = queries[batch_idx]

        t0 = time.perf_counter()
        centroid_ids = index.quantizer_search(batch_q, nprobe)
        t1 = time.perf_counter()

        batch_stats = {} if collect_stats else None
        D, I = index.search_batch_per_list(batch_q, centroid_ids, k, mode=scan_mode,
                                           _stats=batch_stats)
        t2 = time.perf_counter()

        batch_centroid_time = t1 - t0
        batch_scan_time = t2 - t1
        exec_time = t2 - t0
        sim_time += exec_time
        batch_sizes.append(len(batch_idx))

        if collect_stats and batch_stats:
            total_list_loads += batch_stats["list_loads"]
            all_m_values.extend(batch_stats["m_values"])

        for j, idx in enumerate(batch_idx):
            all_ids[idx] = I[j]
            all_dists[idx] = D[j]
            queue_delays[idx] = flush_time - arrival_times[idx]
            latencies[idx] = queue_delays[idx] + exec_time
            centroid_times[idx] = batch_centroid_time
            scan_times[idx] = batch_scan_time

    wall_time = time.perf_counter() - t_wall

    stats = {
        "latencies": latencies,
        "queue_delays": queue_delays,
        "centroid_times": centroid_times,
        "scan_times": scan_times,
        "batch_sizes": np.array(batch_sizes),
        "wall_time": wall_time,
        "sim_time": sim_time,
        "qps": n / wall_time,
    }
    if collect_stats:
        stats["list_loads_per_query"] = total_list_loads / n
        stats["m_values"] = np.array(all_m_values, dtype=np.int64)
    return all_ids, all_dists, stats


# ── Workload generation ──────────────────────────────────────────────────────

def generate_clustered_queries(index, n_queries=10000, n_centers=10, seed=42):
    """Create a clustered workload by sampling from selected inverted lists."""
    rng = np.random.default_rng(seed)
    selected = rng.choice(index.n_clusters, n_centers, replace=False)
    per_center = n_queries // n_centers

    parts = []
    for c_id in selected:
        vecs = index.inverted_lists[c_id]
        if len(vecs) >= per_center:
            idx = rng.choice(len(vecs), per_center, replace=False)
            parts.append(vecs[idx].copy())
        else:
            parts.append(vecs.copy())

    return np.vstack(parts).astype(np.float32)
