"""Parameter sweep: find the best (Δt, MaxBS) for the main experiment.

Run once after fixing the k-means seed. Look for the (Δt, MaxBS) pair where:
  - Batch(MV) > Batch(MM) on random   (m too small, GEMM overhead dominates)
  - Batch(MM) > Batch(MV) on clustered (m large enough, GEMM AI advantage wins)
  - Both beat Sequential

Then set DELTA_T_MS and MAX_BATCH_SIZE in run_main.py.
"""

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
    select_clustered_queries,
)

DATA_DIR   = "data/sift"
N_CLUSTERS = 256
NPROBE     = 8
K          = 10
TARGET_QPS = 2000
N_CENTERS  = 10
DELTA_TS   = [0.5, 1, 2, 5, 10, 20, 50]   # ms
MAX_BSS    = [32, 64, 128, 256]


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
    print(f"\n{'─' * 70}\n  {title}\n{'─' * 70}")


def fmt_lat(times_sec):
    return (f"avg={np.mean(times_sec)*1000:.3f}ms  "
            f"p95={np.percentile(times_sec, 95)*1000:.3f}ms  "
            f"p99={np.percentile(times_sec, 99)*1000:.3f}ms")


def print_sweep_header():
    header = (f"  {'Δt_ms':>6} {'MaxBS':>6} {'AvgBS':>6} "
              f"{'QPS':>8} {'Recall':>7} "
              f"{'AvgLat':>10} {'P95Lat':>10} {'AvgQD':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))


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
            qd  = s["queue_delays"]
            print(f"  {dt:>6.1f} {mbs:>6} {np.mean(s['batch_sizes']):>6.1f} "
                  f"{s['qps']:>8.0f} {r:>7.3f} "
                  f"{np.mean(lat)*1000:>9.3f}ms "
                  f"{np.percentile(lat, 95)*1000:>9.3f}ms "
                  f"{np.mean(qd)*1000:>9.3f}ms")


def main():
    print("=" * 70)
    print("  IVF Batch — Parameter Sweep")
    print("=" * 70)

    print("\nLoading data …")
    base    = read_fvecs(f"{DATA_DIR}/sift_base.fvecs")
    queries = read_fvecs(f"{DATA_DIR}/sift_query.fvecs")
    gt      = read_ivecs(f"{DATA_DIR}/sift_groundtruth.ivecs")
    print(f"  base={base.shape}  queries={queries.shape}  gt={gt.shape}")

    print(f"\nBuilding index (n_clusters={N_CLUSTERS}, seed=0) …")
    t0 = time.perf_counter()
    index = build_custom_index(base, n_clusters=N_CLUSTERS)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    arrivals = generate_arrivals(len(queries), TARGET_QPS)

    section("Sequential — baseline")
    ids_seq, _, s_seq = run_sequential(index, queries, k=K, nprobe=NPROBE)
    print(f"  Recall@{K}: {recall_at_k(ids_seq, gt, K):.3f}  "
          f"QPS: {s_seq['qps']:.0f}  Wall: {s_seq['wall_time']:.3f}s")
    print(f"  Latency: {fmt_lat(s_seq['query_times'])}")

    section("Batch(MV) — random workload sweep")
    print(f"\n  Arrival QPS = {TARGET_QPS}\n")
    run_sweep(index, queries, gt, arrivals, scan_mode="mv")

    section("Batch(MM) — random workload sweep")
    print(f"\n  Arrival QPS = {TARGET_QPS}\n")
    run_sweep(index, queries, gt, arrivals, scan_mode="mm")

    section("Clustered Workload — selected from sift_query.fvecs")
    cl_queries, cl_gt, cl_info = select_clustered_queries(
        index, queries, gt, n_centers=N_CENTERS, seed=42,
    )
    print(f"  selected {cl_info['n_queries']} unique queries in "
          f"{len(cl_info['selected_centroids'])} centroids")
    print(f"  per-centroid counts: {cl_info['queries_per_centroid']}")

    # Tile to match random workload size so AvgBS is comparable.
    # With only ~791 queries, batches would be too small for MM to cross its
    # advantage threshold (m ≈ 8). Round-robin tiling preserves centroid
    # distribution within every time window.
    n_tile = len(queries)
    reps = n_tile // len(cl_queries) + 1
    cl_queries = np.tile(cl_queries, (reps, 1))[:n_tile]
    cl_gt      = np.tile(cl_gt,      (reps, 1))[:n_tile]
    cl_arrivals = generate_arrivals(n_tile, TARGET_QPS)
    print(f"  tiled to {n_tile} arrivals ({reps - 1}–{reps}x repetition)")

    section("Batch(MV) — clustered workload sweep")
    print(f"\n  Arrival QPS = {TARGET_QPS}\n")
    run_sweep(index, cl_queries, cl_gt, cl_arrivals, scan_mode="mv")

    section("Batch(MM) — clustered workload sweep")
    print(f"\n  Arrival QPS = {TARGET_QPS}\n")
    run_sweep(index, cl_queries, cl_gt, cl_arrivals, scan_mode="mm")

    print()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/sweep_{timestamp}.txt"
    tee = Tee(result_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nResults saved to {result_path}")
