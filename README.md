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

The final experiments are run with three scripts:

```bash
# 1. Sweep batching parameters
uv run python run_sweep.py

# 2. Run the main Sequential / Batch(MV) / Batch(MM) comparison
uv run python run_main.py

# 3. Run the MV-vs-MM scan microbenchmark
uv run python microbench_a4.py
```

`run_main.py` defaults to the final-report setting: 5 runs with
`delta_t_ms=5` and `max_batch_size=128`. To override:

```bash
uv run python run_main.py --runs 5 --delta-t-ms 5 --max-bs 128
```

Each script writes its output automatically under `results/`:

```text
results/sweep_<timestamp>.txt
results/main_<timestamp>.txt
results/microbench_a4_<timestamp>.txt
```

`main.py` and `run_experiments.py` are older entry points kept in the repository.
For the final experiment pipeline, use the three scripts above.

## Results

Experiment outputs are stored under `results/`. Older runs are archived under
`results/archived/`.

The final-report results currently used are:

- [`results/sweep_20260506_182948.txt`](results/sweep_20260506_182948.txt)
- [`results/main_20260506_171102.txt`](results/main_20260506_171102.txt)
- [`results/microbench_a4_20260506_171758.txt`](results/microbench_a4_20260506_171758.txt)

## Documentation

- [`documentation/final_report.md`](documentation/final_report.md) — final write-up and results discussion
- [`documentation/experiment_design.md`](documentation/experiment_design.md) — experiment protocol and how to run the three scripts
- [`documentation/parameter_sweep.md`](documentation/parameter_sweep.md) — parameter-sweep details and interpretation
- [`documentation/metrics_design.md`](documentation/metrics_design.md) — metric rationale and measurement design
- [`documentation/MetricDesignCN.md`](documentation/MetricDesignCN.md) — Chinese version of the metric-design notes
- [`documentation/custom_engine_rationale.md`](documentation/custom_engine_rationale.md) — rationale for the custom IVF engine over Faiss search
- [`documentation/custom_engine_rationale_zh.md`](documentation/custom_engine_rationale_zh.md) — Chinese version of the custom-engine rationale
- [`documentation/midterm_report.md`](documentation/midterm_report.md) — midterm report source
- [`documentation/project_proposal/project_proposal.md`](documentation/project_proposal/project_proposal.md) — original project proposal
- [`documentation/project_proposal/reference.md`](documentation/project_proposal/reference.md) — dataset and background references
- [`documentation/midterm_process_outdated/`](documentation/midterm_process_outdated/) — archived midterm process notes
