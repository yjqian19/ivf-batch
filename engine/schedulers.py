import numpy as np
import time
from collections import defaultdict


def generate_arrivals(n_queries, qps, seed=42):
    """Simulate Poisson arrivals at target QPS. Returns cumulative arrival times in seconds."""
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(1.0 / qps, size=n_queries)
    return np.cumsum(inter_arrivals)


# ── Scheduler 1: Sequential ──────────────────────────────────────────────────

def run_sequential(index, queries, k=10, nprobe=8):
    """One query at a time — per-query scanning baseline."""
    n = len(queries)
    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    query_times = np.empty(n)

    t_wall = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        D, I = index.search_one(queries[i], k, nprobe)
        query_times[i] = time.perf_counter() - t0
        all_ids[i] = I
        all_dists[i] = D
    wall_time = time.perf_counter() - t_wall

    return all_ids, all_dists, {
        "query_times": query_times,
        "wall_time": wall_time,
        "qps": n / wall_time,
    }


# ── Scheduler 2: Time-Window Batching ────────────────────────────────────────

def run_time_window(index, queries, arrival_times, delta_t_ms, max_batch_size,
                    k=10, nprobe=8):
    """Dual-trigger flush with per-list batch scanning."""
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

        if len(batch_idx) >= max_batch_size:
            flush_time = arrival_times[batch_idx[-1]]
        else:
            flush_time = deadline

        sim_time = max(sim_time, flush_time)

        # Execute batch: quantizer search + per-list scanning
        batch_q = queries[batch_idx]
        t0 = time.perf_counter()
        centroid_ids = index.quantizer_search(batch_q, nprobe)
        D, I = index.search_batch_per_list(batch_q, centroid_ids, k)
        exec_time = time.perf_counter() - t0
        sim_time += exec_time
        batch_sizes.append(len(batch_idx))

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
    """Greedy grouping by Jaccard similarity of probe sets."""
    n = len(centroid_ids)
    probe_sets = [set(centroid_ids[i].tolist()) for i in range(n)]

    c2q = defaultdict(set)
    for i in range(n):
        for c in probe_sets[i]:
            c2q[c].add(i)

    assigned = np.zeros(n, dtype=bool)
    groups = []

    for i in range(n):
        if assigned[i]:
            continue
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


def run_cluster_batch(index, queries, arrival_times, delta_t_ms, max_batch_size,
                      k=10, nprobe=8,
                      grouping="primary", jaccard_threshold=0.5):
    """Time-window collection + intra-batch cluster grouping + per-list scanning."""
    n = len(queries)
    delta_t = delta_t_ms / 1000.0

    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    latencies = np.empty(n)
    queue_delays = np.empty(n)

    all_batch_sizes = []
    all_group_sizes = []
    total_quantizer_time = 0.0
    total_grouping_time = 0.0

    sim_time = 0.0
    i = 0
    t_wall = time.perf_counter()

    while i < n:
        # ── Collect batch (same dual-trigger logic) ──
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

        if len(batch_idx) >= max_batch_size:
            flush_time = arrival_times[batch_idx[-1]]
        else:
            flush_time = deadline

        sim_time = max(sim_time, flush_time)
        all_batch_sizes.append(len(batch_idx))

        batch_q = queries[batch_idx]

        # ── Quantizer lookup ──
        t0 = time.perf_counter()
        centroid_ids = index.quantizer_search(batch_q, nprobe)
        q_time = time.perf_counter() - t0
        total_quantizer_time += q_time
        sim_time += q_time

        # ── Group within batch ──
        t0 = time.perf_counter()
        if grouping == "primary":
            groups = _group_by_primary(centroid_ids)
        elif grouping == "jaccard":
            groups = _group_by_jaccard(centroid_ids, jaccard_threshold)
        else:
            raise ValueError(f"Unknown grouping: {grouping}")
        g_time = time.perf_counter() - t0
        total_grouping_time += g_time
        sim_time += g_time

        for g in groups:
            all_group_sizes.append(len(g))

        # ── Execute each sub-group with per-list scanning ──
        t0 = time.perf_counter()
        for group in groups:
            group_global = [batch_idx[j] for j in group]
            group_q = queries[group_global]
            group_cids = centroid_ids[group]
            D, I = index.search_batch_per_list(group_q, group_cids, k)
            for j, idx in enumerate(group_global):
                all_ids[idx] = I[j]
                all_dists[idx] = D[j]
        exec_time = time.perf_counter() - t0
        sim_time += exec_time

        for idx in batch_idx:
            queue_delays[idx] = flush_time - arrival_times[idx]
            latencies[idx] = queue_delays[idx] + q_time + g_time + exec_time

    wall_time = time.perf_counter() - t_wall

    return all_ids, all_dists, {
        "latencies": latencies,
        "queue_delays": queue_delays,
        "wall_time": wall_time,
        "qps": n / wall_time,
        "sim_time": sim_time,
        "batch_sizes": np.array(all_batch_sizes),
        "group_sizes": np.array(all_group_sizes),
        "n_groups": len(all_group_sizes),
        "quantizer_time": total_quantizer_time,
        "grouping_time": total_grouping_time,
    }


def generate_clustered_queries(index, n_queries=10000, n_centers=10, seed=42):
    """Create a clustered workload from vectors in selected inverted lists."""
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
