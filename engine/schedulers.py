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

def select_clustered_queries(index, real_queries, gt=None, n_centers=10,
                             target_n=None, seed=42):
    """Filter real queries to those whose primary centroid is one of n_centers
    most-populous centroids. Returns (queries, gt_or_None, info).

    Unlike a synthetic clustered workload, every returned query is a real,
    held-out SIFT query with valid groundtruth — recall is well-defined.

    Parameters
    ----------
    index        : custom IVF index
    real_queries : (Nq, d) the held-out query set (e.g. sift_query.fvecs)
    gt           : (Nq, 100) ground-truth IDs aligned with real_queries, or None
    n_centers    : how many hot centroids to keep — fewer = more clustered
    target_n     : optional sub-sample to this size (for fixed workload size)
    """
    rng = np.random.default_rng(seed)
    primary = index.quantizer_search(real_queries, nprobe=1)[:, 0]
    counts = np.bincount(primary, minlength=index.n_clusters)
    selected = np.argsort(counts)[-n_centers:][::-1]
    mask = np.isin(primary, selected)
    idx = np.where(mask)[0]
    if target_n is not None and len(idx) > target_n:
        idx = np.sort(rng.choice(idx, target_n, replace=False))
    return (
        real_queries[idx].astype(np.float32),
        gt[idx] if gt is not None else None,
        {
            "n_queries": int(len(idx)),
            "selected_centroids": selected.tolist(),
            "queries_per_centroid": counts[selected].tolist(),
        },
    )
