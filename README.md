# IVF Batch Query Scheduling

Comparing three query scheduling strategies — sequential, time-window batching, and cluster-based batching — on an IVF vector index, without modifying the index itself.

## Setup

### 1. Install dependencies

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

### 2. Download the dataset

Download `sift.tar.gz` manually from http://corpus-texmex.irisa.fr/ (click "SIFT1M"), then:

```bash
tar -xzf data/sift.tar.gz -C data/
rm data/sift.tar.gz
```

This places the dataset files under `data/sift/`.

## Running

### Baseline sanity check

Builds the IVF index, runs all 10k queries sequentially, and prints throughput and Recall@10:

```bash
uv run python main.py
```

Expected output:

```
Recall@10: 0.955  (target >= 0.90)
~3,700 QPS
```

## Documentation

- [`documentation/project_proposal.md`](documentation/project_proposal.md) — project scope and methodology
- [`documentation/reference.md`](documentation/reference.md) — references, dataset details, evaluation advice
- [`documentation/implementation_notes.md`](documentation/implementation_notes.md) — running notes on the implementation process
