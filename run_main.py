"""Main experiment: Sequential vs Batch(MV) vs Batch(MM), multi-run.

Set DELTA_T_MS and MAX_BATCH_SIZE based on run_sweep.py results, then run:
  uv run python run_main.py                     # 5 runs, default params
  uv run python run_main.py --runs 5 --delta-t-ms 5 --max-bs 128
"""

import argparse
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

# ── Set after run_sweep.py (sweep_20260506_165210.txt) ───────────────────────
DELTA_T_MS     = 5.0
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


def section(title):
    print(f"\n{'─' * 70}\n  {title}\n{'─' * 70}")


# ── Single-run logic ──────────────────────────────────────────────────────────

def collect_run(index, queries, gt, arrivals,
                cl_queries, cl_gt, cl_arrivals,
                delta_t_ms, max_bs):
    """One complete Sequential/MV/MM comparison on both workloads.
    Returns dict[workload_key][scheduler_key] = metrics dict.
    """
    out = {}
    for wl_key, wl_q, wl_gt, wl_arr in [
        ("random",    queries,    gt,    arrivals),
        ("clustered", cl_queries, cl_gt, cl_arrivals),
    ]:
        wl = {}

        ids_s, _, s = run_sequential(index, wl_q, k=K, nprobe=NPROBE, collect_stats=True)
        qt = s["query_times"]
        wl["seq"] = dict(
            qps=s["qps"],
            recall=recall_at_k(ids_s, wl_gt, K),
            avg_lat=np.mean(qt) * 1e3,
            p95_lat=np.percentile(qt, 95) * 1e3,
            p99_lat=np.percentile(qt, 99) * 1e3,
            queue_ms=0.0,
            scan_ms=np.mean(s["scan_times"]) * 1e3,
            lists_per_q=s["list_loads_per_query"],
            m_mean=float(np.mean(s["m_values"])),
            m_p50=float(np.percentile(s["m_values"], 50)),
            m_p95=float(np.percentile(s["m_values"], 95)),
        )

        for key, mode in [("mv", "mv"), ("mm", "mm")]:
            ids_b, _, s = run_batch(
                index, wl_q, wl_arr,
                delta_t_ms=delta_t_ms, max_batch_size=max_bs,
                k=K, nprobe=NPROBE, scan_mode=mode, collect_stats=True,
            )
            lat = s["latencies"]
            wl[key] = dict(
                qps=s["qps"],
                recall=recall_at_k(ids_b, wl_gt, K),
                avg_lat=np.mean(lat) * 1e3,
                p95_lat=np.percentile(lat, 95) * 1e3,
                p99_lat=np.percentile(lat, 99) * 1e3,
                avg_bs=float(np.mean(s["batch_sizes"])),
                queue_ms=np.mean(s["queue_delays"]) * 1e3,
                scan_ms=np.mean(s["scan_times"]) * 1e3,
                lists_per_q=s["list_loads_per_query"],
                m_mean=float(np.mean(s["m_values"])),
                m_p50=float(np.percentile(s["m_values"], 50)),
                m_p95=float(np.percentile(s["m_values"], 95)),
            )

        out[wl_key] = wl
    return out


def print_run(run_idx, n_runs, results, cl_info):
    for wl_key, wl_label in [
        ("random",    "Random (10000 queries)"),
        ("clustered", f"Clustered ({cl_info['n_queries']} queries, "
                      f"{len(cl_info['selected_centroids'])} centroids)"),
    ]:
        print(f"\n  Workload: {wl_label}")
        wl = results[wl_key]
        for sched, label in [("seq", "Sequential"), ("mv", "Batch(MV)"), ("mm", "Batch(MM)")]:
            m = wl[sched]
            bs = f"  avg_bs={m['avg_bs']:.1f}" if "avg_bs" in m else ""
            print(f"    {label:<12} {m['qps']:>7.0f} QPS  recall={m['recall']:.3f}  "
                  f"avg={m['avg_lat']:.2f}ms  p95={m['p95_lat']:.2f}ms  "
                  f"p99={m['p99_lat']:.2f}ms{bs}")

        print(f"\n    A1 — Latency decomposition:")
        print(f"    {'Scheduler':<12} {'Queue(ms)':>10} {'Scan(ms)':>10} {'Total(ms)':>10}")
        print(f"    {'-' * 44}")
        for sched, label in [("seq", "Sequential"), ("mv", "Batch(MV)"), ("mm", "Batch(MM)")]:
            m = wl[sched]
            print(f"    {label:<12} {m['queue_ms']:>10.2f} {m['scan_ms']:>10.2f} "
                  f"{m['queue_ms'] + m['scan_ms']:>10.2f}")

        print(f"\n    A2+A3 — Per-list reuse and m distribution:")
        print(f"    {'Scheduler':<12} {'Lists/q':>8} {'m_mean':>7} {'m_P50':>6} {'m_P95':>6}")
        print(f"    {'-' * 44}")
        for sched, label in [("seq", "Sequential"), ("mv", "Batch(MV)"), ("mm", "Batch(MM)")]:
            m = wl[sched]
            print(f"    {label:<12} {m['lists_per_q']:>8.2f} {m['m_mean']:>7.1f} "
                  f"{m['m_p50']:>6.0f} {m['m_p95']:>6.0f}")


# ── Multi-run summary ─────────────────────────────────────────────────────────

def _vals(runs, wl, sched, key):
    return [r[wl][sched][key] for r in runs]


def print_summary(all_runs, delta_t_ms, max_bs, cl_info):
    n = len(all_runs)
    print(f"\n  Parameters: Δt={delta_t_ms}ms  MaxBS={max_bs}  ({n} runs)\n")

    for wl_key, wl_label in [
        ("random",    "Random (10000 queries)"),
        ("clustered", f"Clustered ({cl_info['n_queries']} unique × tiled to {cl_info['n_tiled']})"),
    ]:
        print(f"  {wl_label}:")
        print(f"  {'Scheduler':<12}  {'QPS (mean±std)':>16}  "
              f"{'AvgLat (ms)':>16}  {'P95Lat (ms)':>16}  "
              f"{'P99 median':>12}  {'Recall':>12}")
        print(f"  {'─' * 90}")
        for sched, label in [("seq", "Sequential"), ("mv", "Batch(MV)"), ("mm", "Batch(MM)")]:
            qps = _vals(all_runs, wl_key, sched, "qps")
            avg = _vals(all_runs, wl_key, sched, "avg_lat")
            p95 = _vals(all_runs, wl_key, sched, "p95_lat")
            p99 = _vals(all_runs, wl_key, sched, "p99_lat")
            rec = _vals(all_runs, wl_key, sched, "recall")
            print(f"  {label:<12}  "
                  f"{np.mean(qps):>7.0f} ± {np.std(qps):<6.0f}  "
                  f"{np.mean(avg):>8.2f} ± {np.std(avg):<5.2f}  "
                  f"{np.mean(p95):>8.2f} ± {np.std(p95):<5.2f}  "
                  f"{np.median(p99):>12.2f}  "
                  f"{np.mean(rec):>7.3f} ± {np.std(rec):.3f}")

        print(f"\n  A1 — Latency decomposition (mean across {n} runs):")
        print(f"  {'Scheduler':<12}  {'Queue(ms)':>10}  {'Scan(ms)':>10}  {'Total(ms)':>10}")
        print(f"  {'-' * 48}")
        for sched, label in [("seq", "Sequential"), ("mv", "Batch(MV)"), ("mm", "Batch(MM)")]:
            q  = np.mean(_vals(all_runs, wl_key, sched, "queue_ms"))
            sc = np.mean(_vals(all_runs, wl_key, sched, "scan_ms"))
            print(f"  {label:<12}  {q:>10.2f}  {sc:>10.2f}  {q + sc:>10.2f}")

        print(f"\n  A2+A3 — Per-list reuse (mean across {n} runs):")
        print(f"  {'Scheduler':<12}  {'Lists/q':>8}  {'m_mean':>7}  {'m_P50':>6}  {'m_P95':>6}")
        print(f"  {'-' * 48}")
        for sched, label in [("seq", "Sequential"), ("mv", "Batch(MV)"), ("mm", "Batch(MM)")]:
            print(f"  {label:<12}  "
                  f"{np.mean(_vals(all_runs, wl_key, sched, 'lists_per_q')):>8.2f}  "
                  f"{np.mean(_vals(all_runs, wl_key, sched, 'm_mean')):>7.1f}  "
                  f"{np.mean(_vals(all_runs, wl_key, sched, 'm_p50')):>6.0f}  "
                  f"{np.mean(_vals(all_runs, wl_key, sched, 'm_p95')):>6.0f}")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-t-ms", type=float, default=DELTA_T_MS)
    parser.add_argument("--max-bs",     type=int,   default=MAX_BATCH_SIZE)
    parser.add_argument("--runs",       type=int,   default=5)
    args = parser.parse_args()

    delta_t_ms = args.delta_t_ms
    max_bs     = args.max_bs
    n_runs     = args.runs

    print("=" * 70)
    print(f"  Main Experiment  —  Δt={delta_t_ms}ms  MaxBS={max_bs}  runs={n_runs}")
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

    # Prepare both workloads once — arrivals use seed=42, fully deterministic
    arrivals = generate_arrivals(len(queries), TARGET_QPS)
    cl_queries, cl_gt, cl_info = select_clustered_queries(
        index, queries, gt, n_centers=N_CENTERS, seed=42,
    )
    # Tile clustered queries to match random workload size (see experiment_design.md §3.3)
    n_tile = len(queries)
    reps = n_tile // len(cl_queries) + 1
    cl_queries = np.tile(cl_queries, (reps, 1))[:n_tile]
    cl_gt      = np.tile(cl_gt,      (reps, 1))[:n_tile]
    cl_arrivals = generate_arrivals(n_tile, TARGET_QPS)
    cl_info["n_tiled"] = n_tile
    print(f"\n  Clustered workload: {cl_info['n_queries']} unique queries "
          f"tiled to {n_tile} arrivals, "
          f"{len(cl_info['selected_centroids'])} centroids")
    print(f"  per-centroid counts: {cl_info['queries_per_centroid']}")

    all_runs = []
    for i in range(n_runs):
        section(f"Run {i + 1} / {n_runs}")
        r = collect_run(
            index, queries, gt, arrivals,
            cl_queries, cl_gt, cl_arrivals,
            delta_t_ms, max_bs,
        )
        all_runs.append(r)
        print_run(i + 1, n_runs, r, cl_info)

    section(f"Summary — mean ± std across {n_runs} runs")
    print_summary(all_runs, delta_t_ms, max_bs, cl_info)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/main_{timestamp}.txt"
    tee = Tee(result_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nResults saved to {result_path}")
