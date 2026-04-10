import numpy as np
import faiss

faiss.omp_set_num_threads(1)


def build_ivf_index(vectors: np.ndarray, n_clusters: int = 256) -> faiss.IndexIVFFlat:
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)
    index.train(vectors)
    index.add(vectors)
    return index


def search_batch(
    index: faiss.IndexIVFFlat,
    queries: np.ndarray,
    k: int = 10,
    nprobe: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    index.nprobe = nprobe
    distances, ids = index.search(queries, k)
    return ids, distances
