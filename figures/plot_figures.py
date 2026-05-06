"""Generate result figures for the final report."""
# ── Sweep data (all 28 configurations, random & clustered) ───────────────────
# Each row: (mv_avgbs, mv_qps, mm_qps, mv_avglat_ms, mm_avglat_ms)
SWEEP_RANDOM = [
    # (mv_avgbs, mv_qps, mm_qps, mv_lat, mm_lat)
    (21.2,2098,1300,17.5,39.3),(24.2,2101,1511,20.9,60.0),(25.6,2096,1763,21.5,103.1),(24.9,2098,1984,21.0,139.9),
    (27.5,2099,1305,20.8,36.9),(30.8,2153,1382,24.3,65.6),(31.0,2152,1730,24.7,110.4),(30.7,2154,2009,24.3,127.1),
    (29.8,2141,1332,21.5,38.2),(44.2,2200,1401,32.9,64.4),(45.5,2204,2104,34.5,60.5),(47.4,2197,2082,36.6,62.5),
    (31.6,2148,1304,22.3,34.8),(60.2,2230,1446,42.3,67.9),(82.6,2277,2227,60.1,70.5),(84.0,2274,2159,61.1,99.7),
    (31.9,2150,1314,22.5,39.2),(63.3,2187,1404,44.7,61.0),(119.0,2232,1742,85.2,109.5),(147.1,2296,2155,107.9,161.0),
    (31.9,2098,1377,22.9,34.6),(63.7,2182,1438,44.9,65.4),(125.0,2244,2062,87.9,92.9),(222.2,2322,2562,160.8,115.6),
    (31.9,2129,1813,22.7,26.3),(63.7,2202,1620,44.7,57.1),(126.6,2243,2010,88.5,95.1),(250.0,2264,2316,174.8,171.9),
]
SWEEP_CLUSTERED = [
    (30.5,1987,1884),(55.2,1965,2097),(86.2,1957,2061),(106.4,1969,2060),
    (31.4,1945,1959),(60.2,1944,2083),(97.1,1964,2098),(142.9,1954,2102),
    (31.7,1945,1995),(61.7,1951,2315),(108.7,1950,2310),(161.3,1952,2220),
    (31.8,1943,2094),(62.9,1934,2156),(120.5,1952,2324),(200.0,1983,2300),
    (31.9,1988,2082),(63.3,1992,2347),(123.5,2006,2564),(217.4,2024,2520),
    (31.9,1981,2060),(63.7,1951,2265),(125.0,1931,2359),(238.1,1948,2424),
    (31.9,1939,2146),(63.7,1949,2251),(126.6,1938,2285),(250.0,1952,2303),
]
# Selected 4 rows per workload (from Tables 4 & 5)
SEL_RANDOM    = [(21.2,2098,1300),(60.2,2230,1446),(82.6,2277,2227),(222.2,2322,2562)]
SEL_CLUSTERED = [(30.5,1987,1884),(62.9,1934,2156),(120.5,1952,2324),(238.1,1948,2424)]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

GRAY   = "#555555"
BLUE   = "#2166ac"
ORANGE = "#d6604d"
GREEN  = "#4dac26"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ── Figure 1: Microbenchmark ──────────────────────────────────────────────────

L_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256]
mv     = [5.515, 5.474, 5.279, 5.214, 5.154, 5.127, 5.178, 5.124, 5.114]
mm     = [5.583, 15.852, 7.961, 4.016, 3.009, 1.508, 1.425, 1.417, 1.355]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(L_vals, mv, color=BLUE,   marker="o", linewidth=2,
        markersize=6, label="Batch(MV)")
ax.plot(L_vals, mm, color=ORANGE, marker="s", linewidth=2,
        markersize=6, label="Batch(MM)")

ax.axvline(x=8, color=GRAY, linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(8.5, 14, "crossover\nL = 8", color=GRAY, fontsize=9, va="top")

ax.set_xscale("log", base=2)
ax.set_xticks(L_vals)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("L  (queries per list per batch)")
ax.set_ylabel("ns / (query × vector)  [lower is better]")
ax.set_title("Microbenchmark: Scan Kernel Throughput vs L", fontsize=12)
ax.legend()
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("figures/fig_microbench.png", dpi=150)
plt.close()
print("Saved figures/fig_microbench.png")


# ── Figure 2: Parameter Sweep ─────────────────────────────────────────────────

def delta_pct(rows):
    xs = [r[0] for r in rows]
    ys = [(r[2] - r[1]) / r[1] * 100 for r in rows]
    return xs, ys

fig, ax = plt.subplots(figsize=(7, 4.5))

# All sweep points — faint background
xs_r, ys_r = delta_pct(SWEEP_RANDOM)
xs_c, ys_c = delta_pct(SWEEP_CLUSTERED)
ax.scatter(xs_r, ys_r, color=BLUE,   alpha=0.20, s=25, zorder=2)
ax.scatter(xs_c, ys_c, color=ORANGE, alpha=0.20, s=25, zorder=2)

# Selected rows — highlighted lines with markers
sx_r, sy_r = delta_pct(SEL_RANDOM)
sx_c, sy_c = delta_pct(SEL_CLUSTERED)
ax.plot(sx_r, sy_r, color=BLUE,   marker="o", linewidth=2,
        markersize=8, label="Random",    zorder=4)
ax.plot(sx_c, sy_c, color=ORANGE, marker="s", linewidth=2,
        markersize=8, label="Clustered", zorder=4)

# Mark main-experiment point (Δt=5ms, MaxBS=128)
ax.scatter([82.6],  [-2.2], color=BLUE,   s=120, zorder=5,
           edgecolors="white", linewidths=1.5)
ax.scatter([120.5], [19.1], color=ORANGE, s=120, zorder=5,
           edgecolors="white", linewidths=1.5, marker="s")
ax.annotate("main exp.", xy=(82.6, -2.2),  xytext=(95, -12),
            fontsize=8.5, color=GRAY,
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
ax.annotate("main exp.", xy=(120.5, 19.1), xytext=(135, 10),
            fontsize=8.5, color=GRAY,
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(255, 1.5, "breakeven", color=GRAY, fontsize=8.5, ha="right")

ax.set_xlabel("Average Batch Size (MV AvgBS)")
ax.set_ylabel("Δ QPS  (MM − MV)  [%]")
ax.set_title("Batch Size Sweep: Relative QPS of MM vs MV", fontsize=12)
ax.legend()

plt.tight_layout()
plt.savefig("figures/fig_sweep.png", dpi=150)
plt.close()
print("Saved figures/fig_sweep.png")


# ── Figure 3: Latency vs Batch Size ──────────────────────────────────────────

xs_mv  = [r[0] for r in SWEEP_RANDOM]
lat_mv = [r[3] for r in SWEEP_RANDOM]
lat_mm = [r[4] for r in SWEEP_RANDOM]

# Selected 4 rows for highlight lines
sel_xs   = [21.2, 60.2, 82.6, 222.2]
sel_lmv  = [17.5, 42.3, 60.1, 160.8]
sel_lmm  = [39.3, 67.9, 70.5, 115.6]

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.scatter(xs_mv, lat_mv, color=BLUE,   alpha=0.20, s=25, zorder=2)
ax.scatter(xs_mv, lat_mm, color=ORANGE, alpha=0.20, s=25, zorder=2)

ax.plot(sel_xs, sel_lmv, color=BLUE,   marker="o", linewidth=2,
        markersize=8, label="Batch(MV)", zorder=4)
ax.plot(sel_xs, sel_lmm, color=ORANGE, marker="s", linewidth=2,
        markersize=8, label="Batch(MM)", zorder=4)

ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(255, 2, "Sequential (~0.5 ms)", color=GRAY, fontsize=8.5, ha="right")

ax.set_xlabel("Average Batch Size (MV AvgBS)")
ax.set_ylabel("Average Latency (ms)")
ax.set_title("Latency Cost of Batching: Random Workload", fontsize=12)
ax.legend()

plt.tight_layout()
plt.savefig("figures/fig_latency.png", dpi=150)
plt.close()
print("Saved figures/fig_latency.png")
