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

## 3. First Round Evaluation

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
