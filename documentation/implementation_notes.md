# Implementation Notes

## 1. Folder Structure and Core Engine

### Folder Structure

```
ivf-batch/
├── CLAUDE.md
├── README.md                        # project proposal
├── main.py                          # sanity-check: load → build → search → recall
├── pyproject.toml                   # uv-managed dependencies
├── data/                            # datasets (gitignored)
├── documentation/
│   ├── reference.md
│   ├── implementation_notes.md
│   └── project_proposal.pdf
└── engine/
    ├── data.py                      # dataset loaders
    ├── index.py                     # Faiss IVF index build + search
    └── metrics.py                   # recall@k
```

### Core Engine

The engine wraps Faiss `IndexIVFFlat` with three thin modules.

**`engine/index.py`** — builds and searches the index:

```python
faiss.omp_set_num_threads(1)   # disable Faiss internal threading at import time

def build_ivf_index(vectors, n_clusters=256) -> faiss.IndexIVFFlat
def search_batch(index, queries, k=10, nprobe=8) -> (ids, distances)
```

`omp_set_num_threads(1)` is set at module import so that all scheduler benchmarks are isolated from Faiss's own OpenMP parallelism — without this, throughput gains from batching would be conflated with internal multi-threading.

**`engine/metrics.py`**:

```python
def recall_at_k(predicted_ids, ground_truth, k) -> float
```

Measures the fraction of queries where the true nearest neighbor appears in the top-k predictions. Ground truth is taken as `gt[i][0]` (the single closest neighbor).

**`engine/data.py`** — loaders for fvecs/ivecs (INRIA binary format); see Section 2.

### Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_clusters` | 256 | Standard baseline; short lists, fast scan |
| `k` | 10 | Top-10 neighbors |
| `nprobe` | 8 | Tuned to hit ≥0.90 recall on SIFT1M |

### How to Run

```bash
uv run python main.py
```

---

## 2. Dataset: SIFT1M

### Download

Download `sift.tar.gz` manually from http://corpus-texmex.irisa.fr/ (click "SIFT1M"). Move the downloaded file into the `data/` folder, then extract:

```bash
tar -xzf data/sift.tar.gz -C data/
rm data/sift.tar.gz
```

The tarball extracts into `data/sift/` containing:

| File | Shape | Description |
|------|-------|-------------|
| `sift_base.fvecs` | (1,000,000, 128) | Base vectors |
| `sift_query.fvecs` | (10,000, 128) | Query vectors |
| `sift_groundtruth.ivecs` | (10,000, 100) | True neighbor IDs, sorted nearest-first |
| `sift_learn.fvecs` | (100,000, 128) | Training set (unused) |

### fvecs/ivecs Format

INRIA's binary format is simple: each vector is stored as a 4-byte integer dimension `d` followed by `d` × 4-byte values. The loaders in `engine/data.py` read the whole file as `int32`, reshape to `(-1, d+1)`, and slice off the leading dimension column:

```python
def read_fvecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()

def read_ivecs(path: str) -> np.ndarray:  # same, stays int32
    ...
```

`read_fvecs` reinterprets the raw bytes as `float32` via `.view()` after slicing; `read_ivecs` keeps `int32` for the ground-truth index arrays.

### Active Data Path in `main.py`

```python
DATA_DIR = "data/sift"
base    = read_fvecs(f"{DATA_DIR}/sift_base.fvecs")
queries = read_fvecs(f"{DATA_DIR}/sift_query.fvecs")
gt      = read_ivecs(f"{DATA_DIR}/sift_groundtruth.ivecs")
```

---

## 3. How the Engine Works: A Walkthrough of `main.py`

When you run `uv run python main.py`, five things happen in sequence. Here is what each step actually does under the hood.

### Step 1 — Load the dataset

```python
base    = read_fvecs("data/sift/sift_base.fvecs")   # shape: (1_000_000, 128)
queries = read_fvecs("data/sift/sift_query.fvecs")   # shape: (10_000, 128)
gt      = read_ivecs("data/sift/sift_groundtruth.ivecs")  # shape: (10_000, 100)
```

- **`base`** is the database: 1 million SIFT image descriptors, each represented as a 128-dimensional float32 vector. Think of each row as one image patch encoded as a list of 128 numbers.
- **`queries`** are the questions: 10,000 vectors you want to find nearest neighbors for.
- **`gt`** is the answer key: for each of the 10,000 queries, the dataset tells you which 100 base vectors are truly the closest (pre-computed offline). We use only `gt[i][0]` — the single true nearest neighbor — to evaluate correctness.

All three arrays live entirely in RAM. No disk access happens during search.

### Step 2 — Build the IVF index

```python
index = build_ivf_index(base, n_clusters=256)
```

This is a one-time setup that has two sub-steps inside Faiss:

**a) k-means training** — Faiss picks 256 cluster centroids by running k-means over the 1M base vectors. Each centroid is a 128-dimensional point that represents the "center" of a region in the vector space. Think of it as dividing a map into 256 zones.

**b) Assignment** — Every base vector is assigned to its nearest centroid and stored in that centroid's *inverted list* (a bucket). With 1M vectors and 256 clusters, each list holds ~4,000 vectors on average.

After this step, the index knows: "vector #42 lives in cluster #7, vector #1337 lives in cluster #193," and so on.

### Step 3 — Search

```python
ids, distances = search_batch(index, queries, k=10, nprobe=8)
```

For each query vector, Faiss does the following:

1. **Centroid lookup** — find the 8 closest centroids to the query (`nprobe=8`). This narrows the search to 8 out of 256 zones.
2. **Scan inverted lists** — load those 8 buckets (~32,000 vectors total out of 1M) and compute the L2 distance from the query to every vector in them.
3. **Top-k selection** — keep the 10 closest vectors found.

The key insight: instead of comparing the query to all 1M vectors (brute force), we only compare to ~3.2% of them. This is the speed/recall tradeoff — scanning fewer lists is faster but may miss the true nearest neighbor if it lives in a list we skipped.

`nprobe=8` is a tuning knob. Higher nprobe → better recall, slower search. Lower → faster, less accurate.

### Step 4 — Measure throughput

```python
elapsed = time.perf_counter() - t0
qps = len(queries) / elapsed
```

`qps` (queries per second) is the core throughput metric. All 10,000 queries run back-to-back, and we divide the count by the total wall-clock time. This is the **sequential baseline**: one query at a time, no batching tricks. Future schedulers will try to beat this number.

### Step 5 — Verify correctness with Recall@10

```python
r = recall_at_k(ids, gt, k=10)
```

For each query, we check: is the true nearest neighbor (`gt[i][0]`) anywhere in the 10 results we returned? If yes, that query is a "hit". Recall@10 is the fraction of hits across all 10,000 queries.

A value of 1.0 means perfect — we always found the true nearest neighbor. A value of 0.955 means we missed it 4.5% of the time, which is the cost of only scanning 8 out of 256 clusters instead of all of them.

This number must stay the same across all three schedulers. The schedulers reorganize *when* queries are sent to Faiss — they don't change what Faiss computes. So recall should be identical; if it differs, something is wrong.

### Data Flow Summary

```
sift_base.fvecs  ──► read_fvecs() ──► base (1M × 128 float32)
                                           │
                                    build_ivf_index()
                                           │
                                    IndexIVFFlat (256 clusters,
                                    each cluster holds ~4k vectors)
                                           │
sift_query.fvecs ──► read_fvecs() ──► queries (10k × 128 float32)
                                           │
                                    search_batch() ──► ids (10k × 10)
                                           │
sift_groundtruth ──► read_ivecs() ──► gt (10k × 100)
                                           │
                                    recall_at_k() ──► 0.955
```

---

## 4. First Round Evaluation

**Date:** 2026-04-10
**Dataset:** SIFT1M (`sift_base.fvecs` / `sift_query.fvecs` / `sift_groundtruth.ivecs`)
**Parameters:** `n_clusters=256`, `nprobe=8`, `k=10`, single-threaded Faiss (`omp_num_threads=1`)

### Machine

| | |
|-|-|
| Model | MacBook Pro (Mac15,6) |
| Chip | Apple M3 Pro |
| CPU cores | 11 (5 performance + 6 efficiency) |
| Memory | 18 GB |

### Results

| Metric | Value |
|--------|-------|
| Index build time | 0.56 s |
| Search time (10k queries) | 2.692 s |
| Throughput | **3,714 QPS** |
| Recall@10 | **0.955** |

### Interpretation

- Recall@10 of 0.955 comfortably exceeds the 0.90 target. This is the fixed recall floor — all three schedulers must match it to ensure a fair throughput comparison.
- 3,714 QPS is the **sequential baseline** with Faiss threading disabled. This is the number to beat with time-window and cluster-based batching.
- Index build (k-means training + vector assignment) takes 0.56s, a one-time cost.
