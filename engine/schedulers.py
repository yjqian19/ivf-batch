import numpy as np
import time
from collections import defaultdict


def generate_arrivals(n_queries, qps, seed=42):
    """Simulate Poisson arrivals at target QPS. Returns cumulative arrival times in seconds."""
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(1.0 / qps, size=n_queries)
    return np.cumsum(inter_arrivals)


# ── Sequential ───────────────────────────────────────────────────────────────

def run_sequential(index, queries, k=10, nprobe=8):
    """One query at a time — batch_size=1, MV scan."""
    n = len(queries)
    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    query_times = np.empty(n)

    t_wall = time.perf_counter()
    for i in range(n):
        q = queries[i:i+1]
        t0 = time.perf_counter()
        centroid_ids = index.quantizer_search(q, nprobe)
        D, I = index.search_batch_per_list(q, centroid_ids, k, mode="mv")
        query_times[i] = time.perf_counter() - t0
        all_ids[i] = I[0]
        all_dists[i] = D[0]
    wall_time = time.perf_counter() - t_wall

    return all_ids, all_dists, {
        "query_times": query_times,
        "wall_time": wall_time,
        "qps": n / wall_time,
    }


# ── Batch(MV) / Batch(MM) — Time-Window Batching ────────────────────────────

def run_batch(index, queries, arrival_times, delta_t_ms, max_batch_size,
              k=10, nprobe=8, scan_mode="mv"):
    """Time-window batching with dual-trigger flush.

    scan_mode : "mv"  — per-query GEMV inside search_batch_per_list  [Batch(MV)]
                "mm"  — per-list GEMM inside search_batch_per_list   [Batch(MM)]
    """
    n = len(queries)
    delta_t = delta_t_ms / 1000.0

    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    latencies = np.empty(n)
    queue_delays = np.empty(n)
    batch_sizes = []

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
        D, I = index.search_batch_per_list(batch_q, centroid_ids, k, mode=scan_mode)
        exec_time = time.perf_counter() - t0
        sim_time += exec_time
        batch_sizes.append(len(batch_idx))

        for j, idx in enumerate(batch_idx):
            all_ids[idx] = I[j]
            all_dists[idx] = D[j]
            queue_delays[idx] = flush_time - arrival_times[idx]
            latencies[idx] = queue_delays[idx] + exec_time

    wall_time = time.perf_counter() - t_wall

    return all_ids, all_dists, {
        "latencies": latencies,
        "queue_delays": queue_delays,
        "batch_sizes": np.array(batch_sizes),
        "wall_time": wall_time,
        "sim_time": sim_time,
        "qps": n / wall_time,
    }


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
