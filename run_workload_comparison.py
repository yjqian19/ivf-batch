"""Run only the workload comparison section (random vs clustered queries)."""

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
TARGET_QPS = 2000
DELTA_T_MS = 5
MAX_BATCH_SIZE = 128


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


def main():
    print("=" * 70)
    print("  IVF Batch Scheduling — Workload Comparison")
    print(f"  Config: Δt={DELTA_T_MS}ms, MaxBS={MAX_BATCH_SIZE}, "
          f"nprobe={NPROBE}, k={K}")
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

    # ── Generate clustered queries ────────────────────────────────────────
    print("\nGenerating clustered queries (10 regions) …")
    clustered = generate_clustered_queries(index, n_queries=10000, seed=42)
    arr = generate_arrivals(10000, TARGET_QPS)

    # ── Workload comparison ───────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Workload Comparison — Random vs Clustered Queries")
    print(f"{'─' * 70}")

    for wl_name, wl_q, has_gt in [("Random (sift_query)", queries, True),
                                   ("Clustered (10 regions)", clustered, False)]:
        print(f"\n  Workload: {wl_name}")

        ids1, _, s1 = run_sequential(index, wl_q, k=K, nprobe=NPROBE)
        lat1 = s1["query_times"]
        recall_str = f"recall={recall_at_k(ids1, gt, k=K):.3f}  " if has_gt else ""
        print(f"    Sequential:    {s1['qps']:>8.0f} QPS  "
              f"wall={s1['wall_time']:.3f}s  "
              f"{recall_str}"
              f"avg_lat={np.mean(lat1)*1000:.3f}ms  "
              f"p95_lat={np.percentile(lat1, 95)*1000:.3f}ms  "
              f"p99_lat={np.percentile(lat1, 99)*1000:.3f}ms")

        ids2, _, s2 = run_time_window(
            index, wl_q, arr, delta_t_ms=DELTA_T_MS,
            max_batch_size=MAX_BATCH_SIZE, k=K, nprobe=NPROBE,
        )
        lat2 = s2["latencies"]
        recall_str = f"recall={recall_at_k(ids2, gt, k=K):.3f}  " if has_gt else ""
        print(f"    Time-window:   {s2['qps']:>8.0f} QPS  "
              f"wall={s2['wall_time']:.3f}s  "
              f"{recall_str}"
              f"avg_lat={np.mean(lat2)*1000:.3f}ms  "
              f"p95_lat={np.percentile(lat2, 95)*1000:.3f}ms  "
              f"p99_lat={np.percentile(lat2, 99)*1000:.3f}ms  "
              f"avg_qd={np.mean(s2['queue_delays'])*1000:.3f}ms  "
              f"avg_bs={np.mean(s2['batch_sizes']):.1f}")

        ids3, _, s3 = run_cluster_batch(
            index, wl_q, arr, delta_t_ms=DELTA_T_MS,
            max_batch_size=MAX_BATCH_SIZE, k=K, nprobe=NPROBE,
            grouping="primary",
        )
        lat3 = s3["latencies"]
        recall_str = f"recall={recall_at_k(ids3, gt, k=K):.3f}  " if has_gt else ""
        print(f"    Cluster-batch: {s3['qps']:>8.0f} QPS  "
              f"wall={s3['wall_time']:.3f}s  "
              f"{recall_str}"
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
    result_path = f"results/workload_comparison_{timestamp}.txt"
    tee = Tee(result_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nResults saved to {result_path}")
