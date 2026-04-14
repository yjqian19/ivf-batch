"""Run all three schedulers and compare throughput / latency / recall."""

import time
import sys
import os
from datetime import datetime
import numpy as np
from engine.data import read_fvecs, read_ivecs
from engine.custom_index import build_custom_index
from engine.metrics import recall_at_k
from engine.schedulers import (
    generate_arrivals,
    run_sequential,
    run_time_window,
    run_cluster_batch,
    generate_clustered_queries,
)

DATA_DIR = "data/sift"
N_CLUSTERS = 256
NPROBE = 8
K = 10


def fmt_lat(times):
    """Format latency stats in ms."""
    return (f"avg={np.mean(times)*1000:.3f}ms  "
            f"p95={np.percentile(times, 95)*1000:.3f}ms  "
            f"p99={np.percentile(times, 99)*1000:.3f}ms")


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def section(title):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def main():
    print("=" * 70)
    print("  IVF Batch Scheduling Experiments")
    print("=" * 70)

    # ── Load data & build index ───────────────────────────────────────────
    print("\nLoading data …")
    base = read_fvecs(f"{DATA_DIR}/sift_base.fvecs")
    queries = read_fvecs(f"{DATA_DIR}/sift_query.fvecs")
    gt = read_ivecs(f"{DATA_DIR}/sift_groundtruth.ivecs")
    print(f"  base={base.shape}  queries={queries.shape}  gt={gt.shape}")

    print(f"\nBuilding custom IVF index (n_clusters={N_CLUSTERS}) …")
    t0 = time.perf_counter()
    index = build_custom_index(base, n_clusters=N_CLUSTERS)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    # ── 1  Sequential baseline ────────────────────────────────────────────
    section("Scheduler 1 — Sequential (one query at a time)")

    ids_seq, _, stats_seq = run_sequential(index, queries, k=K, nprobe=NPROBE)
    recall_seq = recall_at_k(ids_seq, gt, k=K)
    print(f"  Recall@{K}:  {recall_seq:.3f}")
    print(f"  QPS:       {stats_seq['qps']:.0f}")
    print(f"  Wall time: {stats_seq['wall_time']:.3f}s")
    print(f"  Latency:   {fmt_lat(stats_seq['query_times'])}")

    # ── 2  Time-window batching — parameter sweep ─────────────────────────
    section("Scheduler 2 — Time-Window Batching  (sweep Δt × max_batch_size)")

    delta_ts = [0.5, 1, 2, 5, 10, 20, 50]       # ms
    max_bss = [32, 64, 128, 256]
    target_qps = 2000
    arrivals = generate_arrivals(len(queries), target_qps)

    header = (f"  {'Δt_ms':>6} {'MaxBS':>6} {'AvgBS':>6} "
              f"{'QPS':>8} {'Recall':>7} "
              f"{'AvgLat':>10} {'P95Lat':>10} {'AvgQD':>10}")
    print(f"\n  Arrival QPS = {target_qps}\n")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for dt in delta_ts:
        for mbs in max_bss:
            ids, _, s = run_time_window(
                index, queries, arrivals, dt, mbs, k=K, nprobe=NPROBE,
            )
            r = recall_at_k(ids, gt, k=K)
            lat = s["latencies"]
            qd = s["queue_delays"]
            avg_bs = np.mean(s["batch_sizes"])
            print(f"  {dt:>6.1f} {mbs:>6} {avg_bs:>6.1f} "
                  f"{s['qps']:>8.0f} {r:>7.3f} "
                  f"{np.mean(lat)*1000:>9.3f}ms "
                  f"{np.percentile(lat, 95)*1000:>9.3f}ms "
                  f"{np.mean(qd)*1000:>9.3f}ms")

    # ── 3  Cluster-based batching (same sweep as time-window) ──────────────
    section("Scheduler 3 — Cluster-Based Batching  (sweep Δt × max_batch_size)")

    print(f"\n  Arrival QPS = {target_qps}")

    for label, gmode, jt in [
        ("primary centroid", "primary", None),
    ]:
        print(f"\n  Grouping: {label}")
        header3 = (f"  {'Δt_ms':>6} {'MaxBS':>6} {'AvgBS':>6} "
                   f"{'QPS':>8} {'Recall':>7} "
                   f"{'AvgLat':>10} {'P95Lat':>10} {'AvgQD':>10} "
                   f"{'Groups':>7} {'AvgGrp':>7}")
        print(header3)
        print("  " + "-" * (len(header3) - 2))

        for dt in delta_ts:
            for mbs in max_bss:
                kw = {"grouping": gmode}
                if jt is not None:
                    kw["jaccard_threshold"] = jt
                ids, _, s = run_cluster_batch(
                    index, queries, arrivals, dt, mbs,
                    k=K, nprobe=NPROBE, **kw,
                )
                r = recall_at_k(ids, gt, k=K)
                lat = s["latencies"]
                qd = s["queue_delays"]
                avg_bs = np.mean(s["batch_sizes"])
                gs = s["group_sizes"]
                print(f"  {dt:>6.1f} {mbs:>6} {avg_bs:>6.1f} "
                      f"{s['qps']:>8.0f} {r:>7.3f} "
                      f"{np.mean(lat)*1000:>9.3f}ms "
                      f"{np.percentile(lat, 95)*1000:>9.3f}ms "
                      f"{np.mean(qd)*1000:>9.3f}ms "
                      f"{s['n_groups']:>7} {np.mean(gs):>7.1f}")

    # ── 4  Workload comparison: random vs clustered ───────────────────────
    section("Workload Comparison — Random vs Clustered Queries")

    clustered = generate_clustered_queries(index, n_queries=10000, seed=42)
    arr = generate_arrivals(10000, target_qps)

    for wl_name, wl_q in [("Random (sift_query)", queries),
                           ("Clustered (10 regions)", clustered)]:
        print(f"\n  Workload: {wl_name}")

        _, _, s1 = run_sequential(index, wl_q, k=K, nprobe=NPROBE)
        lat1 = s1["query_times"]
        print(f"    Sequential:    {s1['qps']:>8.0f} QPS  "
              f"wall={s1['wall_time']:.3f}s  "
              f"avg_lat={np.mean(lat1)*1000:.3f}ms  "
              f"p95_lat={np.percentile(lat1, 95)*1000:.3f}ms  "
              f"p99_lat={np.percentile(lat1, 99)*1000:.3f}ms")

        _, _, s2 = run_time_window(
            index, wl_q, arr, delta_t_ms=5, max_batch_size=128,
            k=K, nprobe=NPROBE,
        )
        lat2 = s2["latencies"]
        print(f"    Time-window:   {s2['qps']:>8.0f} QPS  "
              f"wall={s2['wall_time']:.3f}s  "
              f"avg_lat={np.mean(lat2)*1000:.3f}ms  "
              f"p95_lat={np.percentile(lat2, 95)*1000:.3f}ms  "
              f"p99_lat={np.percentile(lat2, 99)*1000:.3f}ms  "
              f"avg_qd={np.mean(s2['queue_delays'])*1000:.3f}ms  "
              f"avg_bs={np.mean(s2['batch_sizes']):.1f}")

        _, _, s3 = run_cluster_batch(
            index, wl_q, arr, delta_t_ms=5, max_batch_size=128,
            k=K, nprobe=NPROBE, grouping="primary",
        )
        lat3 = s3["latencies"]
        print(f"    Cluster-batch: {s3['qps']:>8.0f} QPS  "
              f"wall={s3['wall_time']:.3f}s  "
              f"avg_lat={np.mean(lat3)*1000:.3f}ms  "
              f"p95_lat={np.percentile(lat3, 95)*1000:.3f}ms  "
              f"p99_lat={np.percentile(lat3, 99)*1000:.3f}ms  "
              f"avg_qd={np.mean(s3['queue_delays'])*1000:.3f}ms  "
              f"avg_bs={np.mean(s3['batch_sizes']):.1f}  "
              f"groups={s3['n_groups']}  "
              f"avg_group={np.mean(s3['group_sizes']):.1f}")

    print()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/experiment_{timestamp}.txt"
    tee = Tee(result_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nResults saved to {result_path}")
