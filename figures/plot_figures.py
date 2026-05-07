"""Generate result figures for the final report."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

_results = Path("results")


def _latest(pattern):
    files = sorted(_results.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching results/{pattern}")
    return json.loads(files[-1].read_text())


MICRO = _latest("microbench_a4_*.json")
SWEEP = _latest("sweep_*.json")
MAIN  = _latest("main_*.json")["summary"]

GRAY   = "#555555"
BLUE   = "#2166ac"
ORANGE = "#d6604d"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

MAIN_EXP_KEY = (5.0, 128)
SEL_KEYS     = {(0.5, 32), (5.0, 64), (5.0, 128), (20.0, 256)}


def _key(r):
    return (r["dt_ms"], r["max_bs"])


def paired_rows(mv_rows, mm_rows, keys=None):
    """Return sorted (avg_bs, mv_qps, mm_qps) tuples, optionally filtered to a key set."""
    mm_by = {_key(r): r for r in mm_rows}
    result = [
        (r["avg_bs"], r["qps"], mm_by[_key(r)]["qps"])
        for r in mv_rows
        if _key(r) in mm_by and (keys is None or _key(r) in keys)
    ]
    return sorted(result)


def delta_pct(rows):
    return [r[0] for r in rows], [(r[2] - r[1]) / r[1] * 100 for r in rows]


# ── Figure 1: Microbenchmark ──────────────────────────────────────────────────

L_vals = [r["L"]            for r in MICRO["rows"]]
mv     = [r["mv_ns_per_qv"] for r in MICRO["rows"]]
mm     = [r["mm_ns_per_qv"] for r in MICRO["rows"]]

crossover = next(L for L, m, g in zip(L_vals, mv, mm) if g < m)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(L_vals, mv, color=BLUE,   marker="o", linewidth=2, markersize=6, label="Batch(MV)")
ax.plot(L_vals, mm, color=ORANGE, marker="s", linewidth=2, markersize=6, label="Batch(MM)")
ax.axvline(x=crossover, color=GRAY, linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(crossover * 1.15, max(mm) * 0.88, f"crossover\nL = {crossover}",
        color=GRAY, fontsize=9, va="top")
ax.set_xscale("log", base=2)
ax.set_xticks(L_vals)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("L  (queries per list per batch)")
ax.set_ylabel("ns / (query × vector)  [lower is better]")
ax.legend()
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("figures/fig_microbench.png", dpi=150)
plt.close()
print("Saved figures/fig_microbench.png")


# ── Figure 2: Parameter Sweep ─────────────────────────────────────────────────

all_r = paired_rows(SWEEP["random_mv"],    SWEEP["random_mm"])
all_c = paired_rows(SWEEP["clustered_mv"], SWEEP["clustered_mm"])
sel_r = paired_rows(SWEEP["random_mv"],    SWEEP["random_mm"],    SEL_KEYS)
sel_c = paired_rows(SWEEP["clustered_mv"], SWEEP["clustered_mm"], SEL_KEYS)

main_r = paired_rows(SWEEP["random_mv"],    SWEEP["random_mm"],    {MAIN_EXP_KEY})[0]
main_c = paired_rows(SWEEP["clustered_mv"], SWEEP["clustered_mm"], {MAIN_EXP_KEY})[0]
mrx, mry = main_r[0], (main_r[2] - main_r[1]) / main_r[1] * 100
mcx, mcy = main_c[0], (main_c[2] - main_c[1]) / main_c[1] * 100

fig, ax = plt.subplots(figsize=(7, 4.5))
xs_r, ys_r = delta_pct(all_r)
xs_c, ys_c = delta_pct(all_c)
ax.scatter(xs_r, ys_r, color=BLUE,   alpha=0.20, s=25, zorder=2)
ax.scatter(xs_c, ys_c, color=ORANGE, alpha=0.20, s=25, zorder=2)
sx_r, sy_r = delta_pct(sel_r)
sx_c, sy_c = delta_pct(sel_c)
ax.plot(sx_r, sy_r, color=BLUE,   marker="o", linewidth=2, markersize=8, label="Random",    zorder=4)
ax.plot(sx_c, sy_c, color=ORANGE, marker="s", linewidth=2, markersize=8, label="Clustered", zorder=4)
ax.scatter([mrx], [mry], color=BLUE,   s=120, zorder=5, edgecolors="white", linewidths=1.5)
ax.scatter([mcx], [mcy], color=ORANGE, s=120, zorder=5, edgecolors="white", linewidths=1.5, marker="s")
ax.annotate("main exp.", xy=(mrx, mry), xytext=(mrx + 12, mry - 10),
            fontsize=8.5, color=GRAY, arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
ax.annotate("main exp.", xy=(mcx, mcy), xytext=(mcx + 12, mcy - 8),
            fontsize=8.5, color=GRAY, arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(max(xs_r + xs_c), 1.5, "breakeven", color=GRAY, fontsize=8.5, ha="right")
ax.set_xlabel("Average Batch Size (MV AvgBS)")
ax.set_ylabel("Δ QPS  (MM − MV)  [%]")
ax.legend(handles=[
    Line2D([0], [0], color=BLUE,   marker="o", linewidth=2, markersize=7, label="Random — selected representative"),
    Line2D([0], [0], color=ORANGE, marker="s", linewidth=2, markersize=7, label="Clustered — selected representative"),
    Line2D([0], [0], linestyle="none", marker="o", color=BLUE,   markersize=5, alpha=0.5, label="Random — all data points"),
    Line2D([0], [0], linestyle="none", marker="o", color=ORANGE, markersize=5, alpha=0.5, label="Clustered — all data points"),
], fontsize=8.5, loc="lower right")
plt.tight_layout()
plt.savefig("figures/fig_sweep.png", dpi=150)
plt.close()
print("Saved figures/fig_sweep.png")


# ── Figure 3: Latency vs Batch Size ──────────────────────────────────────────

mm_by   = {_key(r): r for r in SWEEP["random_mm"]}
all_mv  = SWEEP["random_mv"]
sel_mv  = sorted([r for r in all_mv if _key(r) in SEL_KEYS], key=lambda r: r["avg_bs"])

xs_all      = [r["avg_bs"] for r in all_mv]
lat_mv_all  = [r["avg_lat_ms"] for r in all_mv]
lat_mm_all  = [mm_by[_key(r)]["avg_lat_ms"] for r in all_mv]

sel_xs  = [r["avg_bs"] for r in sel_mv]
sel_lmv = [r["avg_lat_ms"] for r in sel_mv]
sel_lmm = [mm_by[_key(r)]["avg_lat_ms"] for r in sel_mv]

seq_lat = SWEEP["sequential"]["avg_lat_ms"]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(xs_all, lat_mv_all, color=BLUE,   alpha=0.20, s=25, zorder=2)
ax.scatter(xs_all, lat_mm_all, color=ORANGE, alpha=0.20, s=25, zorder=2)
ax.plot(sel_xs, sel_lmv, color=BLUE,   marker="o", linewidth=2, markersize=8, label="Batch(MV)", zorder=4)
ax.plot(sel_xs, sel_lmm, color=ORANGE, marker="s", linewidth=2, markersize=8, label="Batch(MM)", zorder=4)
ax.axhline(seq_lat, color=GRAY, linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(max(xs_all), seq_lat + 2, f"Sequential (~{seq_lat:.1f} ms)",
        color=GRAY, fontsize=8.5, ha="right")
ax.set_xlabel("Average Batch Size (MV AvgBS)")
ax.set_ylabel("Average Latency (ms)")
ax.legend(handles=[
    Line2D([0], [0], color=BLUE,   marker="o", linewidth=2, markersize=7, label="Batch(MV) — selected representative"),
    Line2D([0], [0], color=ORANGE, marker="s", linewidth=2, markersize=7, label="Batch(MM) — selected representative"),
    Line2D([0], [0], linestyle="none", marker="o", color=BLUE,   markersize=5, alpha=0.5, label="Batch(MV) — all data points"),
    Line2D([0], [0], linestyle="none", marker="o", color=ORANGE, markersize=5, alpha=0.5, label="Batch(MM) — all data points"),
], fontsize=8.5)
plt.tight_layout()
plt.savefig("figures/fig_latency.png", dpi=150)
plt.close()
print("Saved figures/fig_latency.png")


# ── Figure 4: Latency vs Batch Size — Clustered Workload ─────────────────────

mm_by_c  = {_key(r): r for r in SWEEP["clustered_mm"]}
all_mv_c = SWEEP["clustered_mv"]
sel_mv_c = sorted([r for r in all_mv_c if _key(r) in SEL_KEYS], key=lambda r: r["avg_bs"])

xs_all_c     = [r["avg_bs"] for r in all_mv_c]
lat_mv_all_c = [r["avg_lat_ms"] for r in all_mv_c]
lat_mm_all_c = [mm_by_c[_key(r)]["avg_lat_ms"] for r in all_mv_c]

sel_xs_c  = [r["avg_bs"] for r in sel_mv_c]
sel_lmv_c = [r["avg_lat_ms"] for r in sel_mv_c]
sel_lmm_c = [mm_by_c[_key(r)]["avg_lat_ms"] for r in sel_mv_c]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(xs_all_c, lat_mv_all_c, color=BLUE,   alpha=0.20, s=25, zorder=2)
ax.scatter(xs_all_c, lat_mm_all_c, color=ORANGE, alpha=0.20, s=25, zorder=2)
ax.plot(sel_xs_c, sel_lmv_c, color=BLUE,   marker="o", linewidth=2, markersize=8, label="Batch(MV)", zorder=4)
ax.plot(sel_xs_c, sel_lmm_c, color=ORANGE, marker="s", linewidth=2, markersize=8, label="Batch(MM)", zorder=4)
ax.axhline(seq_lat, color=GRAY, linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(max(xs_all_c), seq_lat + 2, f"Sequential (~{seq_lat:.1f} ms)",
        color=GRAY, fontsize=8.5, ha="right")
ax.set_xlabel("Average Batch Size (MV AvgBS)")
ax.set_ylabel("Average Latency (ms)")
ax.legend(handles=[
    Line2D([0], [0], color=BLUE,   marker="o", linewidth=2, markersize=7, label="Batch(MV) — selected representative"),
    Line2D([0], [0], color=ORANGE, marker="s", linewidth=2, markersize=7, label="Batch(MM) — selected representative"),
    Line2D([0], [0], linestyle="none", marker="o", color=BLUE,   markersize=5, alpha=0.5, label="Batch(MV) — all data points"),
    Line2D([0], [0], linestyle="none", marker="o", color=ORANGE, markersize=5, alpha=0.5, label="Batch(MM) — all data points"),
], fontsize=8.5)
plt.tight_layout()
plt.savefig("figures/fig_latency_clustered.png", dpi=150)
plt.close()
print("Saved figures/fig_latency_clustered.png")


# ── Figures 5 & 6: Performance Overview (QPS and Latency) ────────────────────

schedulers  = ["Sequential", "Batch(MV)", "Batch(MM)"]
sched_keys  = ["seq", "mv", "mm"]
x           = range(len(schedulers))

rand_qps  = [MAIN["random"][k]["qps_mean"]     for k in sched_keys]
clus_qps  = [MAIN["clustered"][k]["qps_mean"]  for k in sched_keys]
rand_lat  = [MAIN["random"][k]["avg_lat_mean"]  for k in sched_keys]
clus_lat  = [MAIN["clustered"][k]["avg_lat_mean"] for k in sched_keys]

# Figure 5: QPS overview
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, rand_qps, color=BLUE,   marker="o", linewidth=2, markersize=8, label="Random")
ax.plot(x, clus_qps, color=ORANGE, marker="s", linewidth=2, markersize=8, label="Clustered")
for xi, y in zip(x, rand_qps):
    ax.text(xi, y + 25, f"{y:.0f}", ha="center", va="bottom", fontsize=8.5, color=BLUE)
for xi, y in zip(x, clus_qps):
    ax.text(xi, y - 55, f"{y:.0f}", ha="center", va="top",    fontsize=8.5, color=ORANGE)
ax.set_xticks(x)
ax.set_xticklabels(schedulers)
ax.set_ylabel("Throughput (QPS)")
ax.legend()
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("figures/fig_overview_qps.png", dpi=150)
plt.close()
print("Saved figures/fig_overview_qps.png")

# Figure 6: Latency overview
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, rand_lat, color=BLUE,   marker="o", linewidth=2, markersize=8, label="Random")
ax.plot(x, clus_lat, color=ORANGE, marker="s", linewidth=2, markersize=8, label="Clustered")
for xi, y in zip(x, rand_lat):
    ax.text(xi, y + 1.5, f"{y:.1f}", ha="center", va="bottom", fontsize=8.5, color=BLUE)
for xi, y in zip(x, clus_lat):
    ax.text(xi, y - 3,   f"{y:.1f}", ha="center", va="top",    fontsize=8.5, color=ORANGE)
ax.set_xticks(x)
ax.set_xticklabels(schedulers)
ax.set_ylabel("Average Latency (ms)")
ax.legend()
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("figures/fig_overview_latency.png", dpi=150)
plt.close()
print("Saved figures/fig_overview_latency.png")
