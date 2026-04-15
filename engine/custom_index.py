"""Custom IVF index with per-list scanning support.

Replaces Faiss's IndexIVFFlat.search() so that schedulers can demonstrate
cache-locality benefits on a single thread.
"""

import numpy as np
from collections import defaultdict
import faiss


class CustomIVFIndex:
    def __init__(self, centroids, inverted_lists, vector_ids):
        self.centroids = centroids.astype(np.float32)        # (n_clusters, d)
        self.inverted_lists = inverted_lists                  # list[np.ndarray (n_c, d)]
        self.vector_ids = vector_ids                          # list[np.ndarray (n_c,)]
        self.n_clusters = len(centroids)
        self.d = centroids.shape[1]
        # Precompute norms for fast L2
        self.centroid_norms = np.sum(centroids ** 2, axis=1)  # (n_clusters,)
        self.list_norms = [
            np.sum(vecs ** 2, axis=1) if len(vecs) > 0 else np.empty(0, dtype=np.float32)
            for vecs in inverted_lists
        ]

    # ── Quantizer ────────────────────────────────────────────────────────────

    def quantizer_search(self, queries, nprobe):
        """Return the nprobe nearest centroid IDs for each query.
        Shape: (n_queries, nprobe) int64.
        """
        q_norms = np.sum(queries ** 2, axis=1, keepdims=True)
        dists = q_norms + self.centroid_norms - 2.0 * queries @ self.centroids.T
        ids = np.argpartition(dists, nprobe, axis=1)[:, :nprobe]
        # sort within the top-nprobe
        for i in range(len(queries)):
            order = np.argsort(dists[i, ids[i]])
            ids[i] = ids[i, order]
        return ids.astype(np.int64)

    # ── Per-query search (sequential baseline) ───────────────────────────────

    def search_one(self, query, k, nprobe):
        """Search a single query. Returns (dists[k], ids[k])."""
        c_ids = self.quantizer_search(query.reshape(1, -1), nprobe)[0]
        q_norm = float(np.sum(query ** 2))

        cand_d, cand_i = [], []
        for c in c_ids:
            vecs = self.inverted_lists[c]
            if len(vecs) == 0:
                continue
            d = q_norm + self.list_norms[c] - 2.0 * (vecs @ query)
            cand_d.append(d)
            cand_i.append(self.vector_ids[c])

        all_d = np.concatenate(cand_d)
        all_i = np.concatenate(cand_i)
        return self._topk(all_d, all_i, k)

    # ── Per-list batch search (core innovation) ──────────────────────────────

    def search_batch_per_list(self, queries, centroid_ids, k):
        """Scan each inverted list once for all queries that need it.

        Parameters
        ----------
        queries      : (n, d) float32
        centroid_ids : (n, nprobe) int64  — from quantizer_search
        k            : int

        Returns
        -------
        (distances, ids)  each (n, k)
        """
        n = len(queries)

        # Build inverted mapping: list_id → [query indices that probe it]
        list_to_queries = defaultdict(list)
        for q_idx in range(n):
            for c in centroid_ids[q_idx]:
                list_to_queries[int(c)].append(q_idx)

        # Accumulate candidates
        cand_d = [[] for _ in range(n)]
        cand_i = [[] for _ in range(n)]
        q_norms = np.sum(queries ** 2, axis=1)  # (n,)

        for list_id, q_indices in list_to_queries.items():
            vecs = self.inverted_lists[list_id]       # (n_c, d) — loaded ONCE
            if len(vecs) == 0:
                continue
            v_ids = self.vector_ids[list_id]           # (n_c,)
            v_norms = self.list_norms[list_id]         # (n_c,)

            # GEMM: compute distances for all queries sharing this list at once
            q_mat = queries[q_indices]                 # (m, d)
            dots = vecs @ q_mat.T                      # (n_c, m)
            for i, q_idx in enumerate(q_indices):
                d = q_norms[q_idx] + v_norms - 2.0 * dots[:, i]
                cand_d[q_idx].append(d)
                cand_i[q_idx].append(v_ids)

        # Extract top-k per query
        out_d = np.empty((n, k), dtype=np.float32)
        out_i = np.empty((n, k), dtype=np.int64)
        for q_idx in range(n):
            ad = np.concatenate(cand_d[q_idx])
            ai = np.concatenate(cand_i[q_idx])
            out_d[q_idx], out_i[q_idx] = self._topk(ad, ai, k)
        return out_d, out_i

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _topk(dists, ids, k):
        if len(dists) <= k:
            pad = k - len(dists)
            order = np.argsort(dists)
            d = np.concatenate([dists[order], np.full(pad, np.inf, dtype=np.float32)])
            i = np.concatenate([ids[order], np.full(pad, -1, dtype=np.int64)])
            return d.astype(np.float32), i.astype(np.int64)
        top = np.argpartition(dists, k)[:k]
        top = top[np.argsort(dists[top])]
        return dists[top].astype(np.float32), ids[top].astype(np.int64)


# ── Build ────────────────────────────────────────────────────────────────────

def build_custom_index(base_vectors, n_clusters=256):
    """Train k-means (via Faiss) and build the custom IVF index."""
    d = base_vectors.shape[1]
    n = len(base_vectors)

    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True)
    kmeans.train(base_vectors)
    centroids = kmeans.centroids.copy()

    # Assign vectors to nearest centroid
    c_norms = np.sum(centroids ** 2, axis=1)
    assignments = np.empty(n, dtype=np.int64)
    bs = 65536
    for s in range(0, n, bs):
        e = min(s + bs, n)
        batch = base_vectors[s:e]
        q_norms = np.sum(batch ** 2, axis=1, keepdims=True)
        dists = q_norms + c_norms - 2.0 * batch @ centroids.T
        assignments[s:e] = np.argmin(dists, axis=1)

    # Build inverted lists
    inverted_lists = []
    vector_ids = []
    for c in range(n_clusters):
        ids = np.where(assignments == c)[0].astype(np.int64)
        inverted_lists.append(base_vectors[ids].copy())
        vector_ids.append(ids)

    return CustomIVFIndex(centroids, inverted_lists, vector_ids)
