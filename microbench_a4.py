"""A4 microbenchmark: scan throughput (ns / query·vector) vs m.

Sweeps m from 1 to 256, comparing:
  MV — m separate GEMV calls  (vecs @ q for each query)
  MM — one GEMM call          (vecs @ Q.T for all m queries)

Uses a synthetic list matching typical SIFT1M statistics:
  n = 4000 vectors (1M / 256 clusters ≈ 3906), d = 128 dimensions.

Results are saved to results/microbench_a4_<timestamp>.txt automatically.

Usage:
  uv run python microbench_a4.py
  uv run python microbench_a4.py --n 4000 --d 128 --repeats 50
"""

import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np


def scan_mv(vecs, queries, v_norms, q_norms):
    """m separate GEMV calls — Batch(MV) kernel."""
    for i in range(len(queries)):
        _ = q_norms[i] + v_norms - 2.0 * (vecs @ queries[i])


def scan_mm(vecs, queries, v_norms, q_norms):
    """One GEMM call for all m queries — Batch(MM) kernel."""
    dots = vecs @ queries.T          # (n_list, m)
    for i in range(len(queries)):
        _ = q_norms[i] + v_norms - 2.0 * dots[:, i]


def bench(fn, vecs, queries, v_norms, q_norms, n_repeat):
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn(vecs, queries, v_norms, q_norms)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000,
                        help="List size (vectors per inverted list)")
    parser.add_argument("--d", type=int, default=128,
                        help="Vector dimension")
    parser.add_argument("--repeats", type=int, default=50,
                        help="Timing repetitions per (m, kernel) combination")
    args = parser.parse_args()

    n, d, n_repeat = args.n, args.d, args.repeats
    m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    v_norms = np.sum(vecs ** 2, axis=1)

    print(f"A4 — Scan throughput vs m")
    print(f"List: n={n} vectors, d={d} dims  |  Repeats: {n_repeat}")
    print(f"Metric: ns / (query × vector)  — lower is better\n")

    header = f"{'m':>5}  {'MV (ns/q·v)':>13}  {'MM (ns/q·v)':>13}  {'MM/MV speedup':>14}"
    print(header)
    print("─" * len(header))

    first_mm_win = None
    rows = []
    for m in m_values:
        queries = rng.standard_normal((m, d)).astype(np.float32)
        q_norms = np.sum(queries ** 2, axis=1)

        mv_sec = bench(scan_mv, vecs, queries, v_norms, q_norms, n_repeat)
        mm_sec = bench(scan_mm, vecs, queries, v_norms, q_norms, n_repeat)

        mv_ns = mv_sec * 1e9 / (m * n)
        mm_ns = mm_sec * 1e9 / (m * n)
        speedup = mv_ns / mm_ns

        rows.append((m, mv_ns, mm_ns, speedup))
        if speedup > 1.0 and first_mm_win is None:
            first_mm_win = m

        marker = " ← MM wins" if speedup >= 1.0 else " ← MV wins"
        print(f"{m:>5}  {mv_ns:>13.3f}  {mm_ns:>13.3f}  {speedup:>13.2f}x{marker}")

    print()
    if first_mm_win is not None:
        print(f"Crossover: MM starts beating MV at m = {first_mm_win}")
    else:
        print("MV wins at all tested m values in this range.")

    print(f"\nTheoretical AI: GEMV = 0.5 FLOP/byte (constant);  "
          f"GEMM = m/2 FLOP/byte (linear in m)")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/microbench_a4_{timestamp}.txt"
    tee = Tee(result_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nResults saved to {result_path}")
