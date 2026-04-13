import numpy as np
import time
import faiss
from collections import defaultdict


def generate_arrivals(n_queries, qps, seed=42):
    """Simulate Poisson arrivals at target QPS. Returns cumulative arrival times in seconds."""
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(1.0 / qps, size=n_queries)
    return np.cumsum(inter_arrivals)


# ── Scheduler 1: Sequential ──────────────────────────────────────────────────

def run_sequential(index, queries, k=10, nprobe=8):
    """One query at a time — true sequential baseline."""
    index.nprobe = nprobe
    n = len(queries)
    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    query_times = np.empty(n)

    t_wall = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        D, I = index.search(queries[i : i + 1], k)
        query_times[i] = time.perf_counter() - t0
        all_ids[i] = I[0]
        all_dists[i] = D[0]
    wall_time = time.perf_counter() - t_wall

    return all_ids, all_dists, {
        "query_times": query_times,
        "wall_time": wall_time,
        "qps": n / wall_time,
    }


# ── Scheduler 2: Time-Window Batching ────────────────────────────────────────

def run_time_window(index, queries, arrival_times, delta_t_ms, max_batch_size,
                    k=10, nprobe=8):
    """Dual-trigger flush: whichever fires first — Δt elapsed OR batch full.

    Simulates a single-threaded server processing batched requests.
    sim_time tracks the virtual clock; wall_time tracks real execution.
    """
    index.nprobe = nprobe
    n = len(queries)
    delta_t = delta_t_ms / 1000.0

    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    latencies = np.empty(n)
    queue_delays = np.empty(n)
    exec_times_per_q = np.empty(n)
    batch_sizes = []

    sim_time = 0.0
    i = 0
    t_wall = time.perf_counter()

    while i < n:
        # Wait for first query of this window
        sim_time = max(sim_time, arrival_times[i])
        window_open = sim_time
        deadline = window_open + delta_t
        batch_idx = []

        # Collect until a trigger fires
        while i < n and len(batch_idx) < max_batch_size:
            if arrival_times[i] <= deadline:
                batch_idx.append(i)
                i += 1
            else:
                break  # time trigger — no more queries before deadline

        # Determine flush time
        if len(batch_idx) >= max_batch_size:
            flush_time = arrival_times[batch_idx[-1]]   # size trigger
        else:
            flush_time = deadline                        # time trigger

        sim_time = max(sim_time, flush_time)

        # Execute batch
        batch_q = queries[batch_idx]
        t0 = time.perf_counter()
        D, I = index.search(batch_q, k)
        exec_time = time.perf_counter() - t0
        sim_time += exec_time
        batch_sizes.append(len(batch_idx))

        # Record per-query results
        for j, idx in enumerate(batch_idx):
            all_ids[idx] = I[j]
            all_dists[idx] = D[j]
            queue_delays[idx] = flush_time - arrival_times[idx]
            exec_times_per_q[idx] = exec_time
            latencies[idx] = queue_delays[idx] + exec_time

    wall_time = time.perf_counter() - t_wall

    return all_ids, all_dists, {
        "latencies": latencies,
        "queue_delays": queue_delays,
        "exec_times": exec_times_per_q,
        "batch_sizes": np.array(batch_sizes),
        "wall_time": wall_time,
        "sim_time": sim_time,
        "qps": n / wall_time,
    }


# ── Scheduler 3: Cluster-Based Batching ──────────────────────────────────────

def _group_by_primary(centroid_ids):
    """Group queries that share the same nearest centroid."""
    groups = defaultdict(list)
    for i in range(len(centroid_ids)):
        groups[centroid_ids[i, 0]].append(i)
    return list(groups.values())


def _group_by_jaccard(centroid_ids, threshold=0.5):
    """Greedy grouping: a query joins the current group if its probe-set
    Jaccard similarity with the group seed exceeds `threshold`."""
    n = len(centroid_ids)
    probe_sets = [set(centroid_ids[i].tolist()) for i in range(n)]

    # Inverted index: centroid → queries that probe it (fast candidate lookup)
    c2q = defaultdict(set)
    for i in range(n):
        for c in probe_sets[i]:
            c2q[c].add(i)

    assigned = np.zeros(n, dtype=bool)
    groups = []

    for i in range(n):
        if assigned[i]:
            continue
        # Candidates: queries sharing ≥1 centroid with query i
        candidates = set()
        for c in probe_sets[i]:
            candidates.update(c2q[c])

        group = [i]
        assigned[i] = True
        seed_set = probe_sets[i]

        for j in sorted(candidates):
            if j == i or assigned[j]:
                continue
            shared = len(seed_set & probe_sets[j])
            union = len(seed_set | probe_sets[j])
            if union > 0 and shared / union >= threshold:
                group.append(j)
                assigned[j] = True

        groups.append(group)

    return groups


def run_cluster_batch(index, queries, k=10, nprobe=8,
                      grouping="primary", jaccard_threshold=0.5):
    """Group queries by centroid overlap, then search per group.

    grouping="primary"  → group by nearest centroid (fast)
    grouping="jaccard"  → greedy Jaccard-threshold grouping
    """
    index.nprobe = nprobe
    n = len(queries)
    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)

    # Step 1: quantizer lookup — find each query's top-nprobe centroids
    quantizer = faiss.downcast_index(index.quantizer)
    t0 = time.perf_counter()
    _, centroid_ids = quantizer.search(queries, nprobe)
    quantizer_time = time.perf_counter() - t0

    # Step 2: form groups
    t0 = time.perf_counter()
    if grouping == "primary":
        groups = _group_by_primary(centroid_ids)
    elif grouping == "jaccard":
        groups = _group_by_jaccard(centroid_ids, jaccard_threshold)
    else:
        raise ValueError(f"Unknown grouping: {grouping}")
    grouping_time = time.perf_counter() - t0

    # Step 3: execute each group
    group_exec_times = []
    query_exec_time = np.empty(n)

    t_exec = time.perf_counter()
    for group in groups:
        group_q = queries[group]
        t0 = time.perf_counter()
        D, I = index.search(group_q, k)
        et = time.perf_counter() - t0
        group_exec_times.append(et)

        for j, idx in enumerate(group):
            all_ids[idx] = I[j]
            all_dists[idx] = D[j]
            query_exec_time[idx] = et
    total_exec_time = time.perf_counter() - t_exec

    wall_time = quantizer_time + grouping_time + total_exec_time
    group_sizes = np.array([len(g) for g in groups])

    return all_ids, all_dists, {
        "quantizer_time": quantizer_time,
        "grouping_time": grouping_time,
        "total_exec_time": total_exec_time,
        "wall_time": wall_time,
        "qps": n / wall_time,
        "n_groups": len(groups),
        "group_sizes": group_sizes,
        "group_exec_times": np.array(group_exec_times),
        "query_exec_time": query_exec_time,
    }


def generate_clustered_queries(index, base, n_queries=10000, n_centers=10, seed=42):
    """Create a clustered workload: queries drawn from a few regions of vector space.
    Best case for cluster-based batching (high centroid overlap)."""
    rng = np.random.default_rng(seed)
    quantizer = faiss.downcast_index(index.quantizer)
    n_centroids = quantizer.ntotal
    d = base.shape[1]

    # Extract all centroids as a numpy array
    centroids = quantizer.reconstruct_n(0, n_centroids)

    selected = rng.choice(n_centroids, n_centers, replace=False)
    per_center = n_queries // n_centers

    parts = []
    for c_id in selected:
        centroid = centroids[c_id : c_id + 1].copy()
        _, nn_ids = index.search(centroid, per_center)
        parts.append(base[nn_ids[0]])

    return np.vstack(parts).astype(np.float32)
