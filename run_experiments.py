"""Run experiments: Sequential vs Batch(MV) vs Batch(MM)."""

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
    run_batch,
    generate_clustered_queries,
)

DATA_DIR = "data/sift"
N_CLUSTERS = 256
NPROBE = 8
K = 10
TARGET_QPS = 2000
DELTA_TS = [0.5, 1, 2, 5, 10, 20, 50]   # ms
MAX_BSS = [32, 64, 128, 256]


def fmt_lat(times_sec):
    return (f"avg={np.mean(times_sec)*1000:.3f}ms  "
            f"p95={np.percentile(times_sec, 95)*1000:.3f}ms  "
            f"p99={np.percentile(times_sec, 99)*1000:.3f}ms")


class Tee:
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


def print_sweep_header():
    header = (f"  {'Δt_ms':>6} {'MaxBS':>6} {'AvgBS':>6} "
              f"{'QPS':>8} {'Recall':>7} "
              f"{'AvgLat':>10} {'P95Lat':>10} {'AvgQD':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    return header


def run_sweep(index, queries, gt, arrivals, scan_mode):
    print_sweep_header()
    for dt in DELTA_TS:
        for mbs in MAX_BSS:
            ids, _, s = run_batch(
                index, queries, arrivals, dt, mbs,
                k=K, nprobe=NPROBE, scan_mode=scan_mode,
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

    arrivals = generate_arrivals(len(queries), TARGET_QPS)

    # ── 1  Sequential baseline ────────────────────────────────────────────
    section("Sequential — one query at a time")

    ids_seq, _, stats_seq = run_sequential(index, queries, k=K, nprobe=NPROBE)
    recall_seq = recall_at_k(ids_seq, gt, k=K)
    print(f"  Recall@{K}:  {recall_seq:.3f}")
    print(f"  QPS:       {stats_seq['qps']:.0f}")
    print(f"  Wall time: {stats_seq['wall_time']:.3f}s")
    print(f"  Latency:   {fmt_lat(stats_seq['query_times'])}")

    # ── 2  Batch(MV) — parameter sweep ───────────────────────────────────
    section("Batch(MV) — time-window batching, per-list MV scan")
    print(f"\n  Arrival QPS = {TARGET_QPS}\n")
    run_sweep(index, queries, gt, arrivals, scan_mode="mv")

    # ── 3  Batch(MM) — parameter sweep ───────────────────────────────────
    section("Batch(MM) — time-window batching, per-list MM scan")
    print(f"\n  Arrival QPS = {TARGET_QPS}\n")
    run_sweep(index, queries, gt, arrivals, scan_mode="mm")

    # ── 4  Workload comparison: random vs clustered ───────────────────────
    section("Workload Comparison — Random vs Clustered Queries")

    clustered = generate_clustered_queries(index, n_queries=10000, seed=42)
    arr = generate_arrivals(10000, TARGET_QPS)

    for wl_name, wl_q in [("Random (sift_query)", queries),
                           ("Clustered (10 regions)", clustered)]:
        print(f"\n  Workload: {wl_name}")

        _, _, s1 = run_sequential(index, wl_q, k=K, nprobe=NPROBE, collect_stats=True)
        t1 = s1["query_times"]
        print(f"    Sequential: {s1['qps']:>8.0f} QPS  wall={s1['wall_time']:.3f}s  "
              f"avg_lat={np.mean(t1)*1000:.3f}ms  "
              f"p95_lat={np.percentile(t1, 95)*1000:.3f}ms  "
              f"p99_lat={np.percentile(t1, 99)*1000:.3f}ms")

        batch_results = {}
        for label, mode in [("Batch(MV)", "mv"), ("Batch(MM)", "mm")]:
            _, _, s = run_batch(
                index, wl_q, arr, delta_t_ms=5, max_batch_size=128,
                k=K, nprobe=NPROBE, scan_mode=mode, collect_stats=True,
            )
            lat = s["latencies"]
            print(f"    {label}:   {s['qps']:>8.0f} QPS  wall={s['wall_time']:.3f}s  "
                  f"avg_lat={np.mean(lat)*1000:.3f}ms  "
                  f"p95_lat={np.percentile(lat, 95)*1000:.3f}ms  "
                  f"p99_lat={np.percentile(lat, 99)*1000:.3f}ms  "
                  f"avg_qd={np.mean(s['queue_delays'])*1000:.3f}ms  "
                  f"avg_bs={np.mean(s['batch_sizes']):.1f}")
            batch_results[label] = s

        # ── A1: Latency decomposition ─────────────────────────────────────
        print(f"\n    A1 — Latency decomposition (avg per query):")
        print(f"    {'Scheduler':<14} {'Queue (ms)':>11} {'Scan (ms)':>10} {'Total (ms)':>11}")
        print(f"    {'-'*48}")
        q_delay = np.mean(s1['queue_delays']) * 1000
        scan    = np.mean(s1['scan_times']) * 1000
        print(f"    {'Sequential':<14} {q_delay:>10.2f}  {scan:>9.2f}  {q_delay+scan:>10.2f}")
        for label, s in batch_results.items():
            q_delay = np.mean(s['queue_delays']) * 1000
            scan    = np.mean(s['scan_times']) * 1000
            print(f"    {label:<14} {q_delay:>10.2f}  {scan:>9.2f}  {q_delay+scan:>10.2f}")

        # ── A2 + A3: Per-list reuse and m distribution ────────────────────
        print(f"\n    A2+A3 — Per-list reuse and m distribution:")
        print(f"    {'Scheduler':<14} {'Lists/query':>12} {'m_mean':>7} {'m_P50':>6} {'m_P95':>6}")
        print(f"    {'-'*48}")
        mv = s1["m_values"]
        print(f"    {'Sequential':<14} {s1['list_loads_per_query']:>12.1f} "
              f"{np.mean(mv):>7.1f} {np.percentile(mv,50):>6.0f} {np.percentile(mv,95):>6.0f}")
        for label, s in batch_results.items():
            mv = s["m_values"]
            print(f"    {label:<14} {s['list_loads_per_query']:>12.1f} "
                  f"{np.mean(mv):>7.1f} {np.percentile(mv,50):>6.0f} {np.percentile(mv,95):>6.0f}")

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
