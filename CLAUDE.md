# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research project comparing three query execution strategies for IVF-based vector search on a single machine, without modifying the index structure:

1. **Sequential** — baseline, one query at a time
2. **Batch(MV)** — time-window batching, per-query GEMV scan inside each inverted list
3. **Batch(MM)** — time-window batching, per-list GEMM scan (matrix × matrix across all queries sharing a list)

**Goal:** improve throughput by reorganizing query execution, not the index. The key variable is m — the number of queries sharing each inverted list per batch — which determines whether GEMM pays off over GEMV.

## Setup

```bash
uv init                        # initialize project (if not already done)
uv add faiss-cpu numpy         # or faiss-gpu if CUDA available
uv run python run_experiments.py
```

Download datasets:
- **SIFT1M** (primary): http://corpus-texmex.irisa.fr/ — `sift_base.fvecs`, `sift_query.fvecs`, `sift_groundtruth.ivecs`
- **GloVe-100** (secondary): https://ann-benchmarks.com/

## Evaluation Targets

- Metrics: throughput (QPS), average/P95/P99 latency, Recall@10
- All three schedulers must achieve the same Recall@10 before comparing throughput
- Two workloads: **random queries** (small m, Batch(MM) disadvantaged) and **clustered queries** (large m, Batch(MM) advantaged)

## Timeline

- Phase 1 (→ Apr 14): Core engine + data loading + single-query validation ✓
- Phase 2 (Apr 14 → May 3): Refine schedulers and scan modes, run experiments
- Phase 3 (May 3 → May 7): Report + video

## Report HTML Style

`documentation/midterm_report.html` uses these style preferences:
- `hr` — `display: none` (no horizontal rules rendered)
- inline code — green (`#3a8a3a`), not the default yellow
- `h2` — `margin-top: 2em` for section spacing

When regenerating the HTML from Markdown, reapply these overrides in the CSS block.

To export to A4 PDF using Chrome headless:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new \
  --no-sandbox \
  --print-to-pdf="documentation/midterm_report.pdf" \
  --print-to-pdf-no-header \
  --no-pdf-header-footer \
  --paper-size=A4 \
  "file:///Volumes/yjbolt/projects/ivf-batch/documentation/midterm_report.html"
```

## Reference Files

- `documentation/project_proposal/project_proposal.md` — original proposal
- `documentation/reference.md` — implementation notes, dataset info, evaluation advice
- `documentation/yujia_recap.md` — current engine and scheduler implementation details
- `documentation/custom_engine_rationale.md` — rationale for custom engine over Faiss
