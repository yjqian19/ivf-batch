import time
from engine.data import read_fvecs, read_ivecs
from engine.index import build_ivf_index, search_batch
from engine.metrics import recall_at_k

DATA_DIR = "data/sift"
N_CLUSTERS = 256
NPROBE = 8
K = 10


def main():
    print("Loading data...")
    base    = read_fvecs(f"{DATA_DIR}/sift_base.fvecs")
    queries = read_fvecs(f"{DATA_DIR}/sift_query.fvecs")
    gt      = read_ivecs(f"{DATA_DIR}/sift_groundtruth.ivecs")
    print(f"  base:    {base.shape}")
    print(f"  queries: {queries.shape}")
    print(f"  gt:      {gt.shape}")

    print(f"\nBuilding IVF index (n_clusters={N_CLUSTERS})...")
    t0 = time.perf_counter()
    index = build_ivf_index(base, n_clusters=N_CLUSTERS)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    print(f"\nSearching (nprobe={NPROBE}, k={K})...")
    t0 = time.perf_counter()
    ids, _ = search_batch(index, queries, k=K, nprobe=NPROBE)
    elapsed = time.perf_counter() - t0
    qps = len(queries) / elapsed
    print(f"  done in {elapsed:.3f}s  ({qps:.0f} QPS)")

    r = recall_at_k(ids, gt, k=K)
    print(f"\nRecall@{K}: {r:.3f}  (target >= 0.90)")


if __name__ == "__main__":
    main()
