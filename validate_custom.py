"""Quick sanity check: custom engine vs Faiss recall."""
import time
import numpy as np
from engine.data import read_fvecs, read_ivecs
from engine.index import build_ivf_index
from engine.custom_index import build_custom_index
from engine.metrics import recall_at_k

DATA = "data/sift"
K, NPROBE, NC = 10, 8, 256

base = read_fvecs(f"{DATA}/sift_base.fvecs")
queries = read_fvecs(f"{DATA}/sift_query.fvecs")
gt = read_ivecs(f"{DATA}/sift_groundtruth.ivecs")
print(f"base={base.shape}  queries={queries.shape}")

# ── Build custom index ──
print("\nBuilding custom index …")
t0 = time.perf_counter()
cidx = build_custom_index(base, n_clusters=NC)
print(f"  done in {time.perf_counter()-t0:.1f}s")

# ── Per-query search (sequential style) ──
print("\nPer-query search (100 queries) …")
n_test = 100
ids_pq = np.empty((n_test, K), dtype=np.int64)
t0 = time.perf_counter()
for i in range(n_test):
    _, ids_pq[i] = cidx.search_one(queries[i], K, NPROBE)
t_pq = time.perf_counter() - t0
r_pq = recall_at_k(ids_pq, gt[:n_test], K)
print(f"  recall@{K} = {r_pq:.3f}   time = {t_pq:.3f}s   QPS = {n_test/t_pq:.0f}")

# ── Per-list batch search ──
print("\nPer-list batch search (100 queries) …")
batch_q = queries[:n_test]
t0 = time.perf_counter()
c_ids = cidx.quantizer_search(batch_q, NPROBE)
_, ids_pl = cidx.search_batch_per_list(batch_q, c_ids, K)
t_pl = time.perf_counter() - t0
r_pl = recall_at_k(ids_pl, gt[:n_test], K)
print(f"  recall@{K} = {r_pl:.3f}   time = {t_pl:.3f}s   QPS = {n_test/t_pl:.0f}")

# ── Check agreement ──
match = np.mean(np.sort(ids_pq, axis=1) == np.sort(ids_pl, axis=1))
print(f"\nPer-query vs per-list result agreement: {match*100:.1f}%")
