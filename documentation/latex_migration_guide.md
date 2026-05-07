# LaTeX Migration Guide — ACM sigconf

## 0. Target Layout

All LaTeX files live in `documentation/latex/` to keep the tree clean:

```
documentation/
  latex/
    final_report_acm.tex
    references.bib
    *.aux  *.log  *.pdf  (build artifacts)
figures/
  fig_overview_qps.png
  fig_sweep.png
  fig_latency.png
  fig_latency_clustered.png
  fig_microbench.png
```

---

## 1. Install Tools (one-time)

```bash
brew install --cask mactex-no-gui   # full TeX Live (~3 GB)
brew install pandoc
```

Install **LaTeX Workshop** in VS Code for live PDF preview.

---

## 2. Bootstrap with Pandoc

Run once to get a rough conversion of the body text:

```bash
pandoc documentation/final_report.md -o documentation/latex/final_report_acm.tex
```

The output won't be ACM-formatted, but it converts sections, bold, italic, tables, and inline code correctly. Use it as the body to paste into the ACM template below — don't use it as-is.

---

## 3. ACM sigconf Preamble

Create `documentation/latex/final_report_acm.tex` with this shell:

```latex
\documentclass[sigconf, nonacm]{acmart}
% nonacm removes the ACM copyright block for course submissions

\usepackage{booktabs}
\usepackage{graphicx}
\graphicspath{{../../figures/}}   % figures/ is two levels up from latex/

\title{Query Scheduling Optimization in IVF-based Vector Search}

\author{Yujia Qian}
\email{yjqian19@mit.edu}
\affiliation{\institution{MIT}}

\author{Xiangyu Guan}
\email{xiang949@mit.edu}
\affiliation{\institution{MIT}}

\begin{document}
\maketitle

\begin{abstract}
% paste abstract here (drop the surrounding --- lines)
\end{abstract}

\keywords{vector search, IVF, batch scheduling, GEMM, query execution}

% ── paste body sections here ──

\bibliographystyle{ACM-Reference-Format}
\bibliography{references}

\end{document}
```

---

## 4. Markdown → LaTeX Translations

### Sections

```
# Title           →  (handled by \title{})
## 1. Intro       →  \section{Introduction}
### 2.1 Design    →  \subsection{System Design}
#### 2.1.1 IVF   →  \subsubsection{IVF Engine}
```

The author line `Yujia Qian — yjqian19@mit.edu · Xiangyu Guan — ...` at the top of the Markdown is **not a section** — delete it entirely (it's in the ACM metadata above).

The `---` horizontal-rule separators between sections are **not needed** — delete them.

**Section numbering fix:** The Markdown jumps from §3.2 to §3.4 (§3.3 is missing). Renumber to §3.3 in the LaTeX body.

### Text formatting

```
**bold**          →  \textbf{bold}
*italic*          →  \textit{italic}
`inline code`     →  \texttt{inline code}
```

Standalone bold paragraphs like `**Batch(MV).**` and `**RQ1.**` use `\textbf{...}` as-is within `\paragraph{}` or inline.

### Special characters in this paper

Every occurrence of these must be replaced with LaTeX math:

| Markdown | LaTeX | Where it appears |
|---|---|---|
| `Δt` | `$\Delta t$` | Throughout Method/Results |
| `±` | `$\pm$` | All table cells (e.g. `1,854 ± 58`) |
| `×` | `$\times$` | Discussion (e.g. `14.5×`) |
| `≈` | `$\approx$` | Discussion (e.g. `≈ 3.6`) |
| `←` | `$\leftarrow$` | Table 6, MM/MV crossover row |
| `~` | `$\sim$` or `{\raise.17ex\hbox{$\scriptstyle\sim$}}` | "~0.5 ms" in Discussion |

Also convert inline expressions like `L/2 FLOP/byte` → `$L/2$ FLOP/byte` and `0.5 FLOP/byte` → `$0.5$ FLOP/byte`.

### Figures (all 5)

Pattern for each:

```latex
\begin{figure}
  \centering
  \includegraphics[width=\linewidth]{fig_overview_qps}
  \caption{Throughput (QPS) across schedulers for random and clustered workloads.}
  \label{fig:overview-qps}
\end{figure}
```

All 5 figure filenames (extension omitted — LaTeX finds the PNG automatically):

| Label | File |
|---|---|
| `fig:overview-qps` | `fig_overview_qps` |
| `fig:sweep` | `fig_sweep` |
| `fig:latency-random` | `fig_latency` |
| `fig:latency-clustered` | `fig_latency_clustered` |
| `fig:microbench` | `fig_microbench` |

Cross-reference with `\autoref{fig:overview-qps}` instead of hardcoding "Figure 1".

### Tables (all 6)

ACM standard: caption goes **above** the table, use `booktabs`. The Markdown has 6 tables total — note that Tables 1 and 2 each split into two sub-tables (random/clustered) in the Markdown; keep them as one `\begin{table}` each with a shared caption, using a separator row or `\midrule` between the two halves.

**Table numbering fix:** The Markdown is missing Table 3 entirely (jumps 1 → 2 → 4 → 5 → 6). Renumber to 1–6 sequentially in LaTeX.

Example (Table 1):

```latex
\begin{table}
  \caption{Throughput and Recall@10 (mean $\pm$ std, 5 runs). $\Delta t$ = 5\,ms, MaxBS = 128.}
  \label{tab:main-results}
  \begin{tabular}{lrrrrr}
    \toprule
    Scheduler & QPS & Avg Lat (ms) & P95 (ms) & P99 (ms) & Recall@10 \\
    \midrule
    \multicolumn{6}{l}{\textit{Random workload}} \\
    \midrule
    Sequential & $1{,}854 \pm 58$ & 0.54 & 0.74 & 0.81 & $0.957 \pm 0.000$ \\
    Batch(MV)  & $\mathbf{2{,}202 \pm 23}$ & 74.34 & 106.79 & 114.67 & $0.957 \pm 0.000$ \\
    Batch(MM)  & $1{,}791 \pm 87$ & 104.69 & 134.60 & 152.89 & $0.957 \pm 0.000$ \\
    \midrule
    \multicolumn{6}{l}{\textit{Clustered workload}} \\
    \midrule
    Sequential & $1{,}650 \pm 33$ & 0.61 & 0.78 & 0.88 & $0.981 \pm 0.000$ \\
    Batch(MV)  & $1{,}933 \pm 39$ & 96.19 & 127.30 & 139.06 & $0.981 \pm 0.000$ \\
    Batch(MM)  & $\mathbf{2{,}303 \pm 111}$ & 58.09 & 87.15 & 106.98 & $0.981 \pm 0.000$ \\
    \bottomrule
  \end{tabular}
\end{table}
```

Note `$1{,}854$` (braces prevent LaTeX from treating the comma as a math separator).

### Bullet lists / RQs

```
- **RQ1.** Does batching...  →  \begin{itemize}
                                  \item \textbf{RQ1.} Does batching...
                                \end{itemize}
```

---

## 5. References (BibTeX)

Create `documentation/latex/references.bib`. Minimum entries needed by the paper:

```bibtex
@article{faiss2021,
  author  = {Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  title   = {Billion-Scale Similarity Search with {GPU}s},
  journal = {{IEEE} Transactions on Big Data},
  year    = {2021},
}

@article{hnsw2020,
  author  = {Malkov, Yuri A. and Yashunin, Dmitry A.},
  title   = {Efficient and Robust Approximate Nearest Neighbor Search
             Using Hierarchical Navigable Small World Graphs},
  journal = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2020},
}
```

Cite with `\cite{faiss2021}`. Add further entries as needed from your inline citations.

---

## 6. Build

```bash
cd documentation/latex
pdflatex final_report_acm.tex
bibtex   final_report_acm
pdflatex final_report_acm.tex
pdflatex final_report_acm.tex   # third pass resolves all cross-references
```

Or VS Code + LaTeX Workshop: `Cmd+Shift+P` → **LaTeX Workshop: Build LaTeX project**.

---

## 7. Checklist

- [ ] `mkdir documentation/latex` and create `final_report_acm.tex` with preamble
- [ ] Create `documentation/latex/references.bib`
- [ ] Run Pandoc bootstrap, paste body into template
- [ ] Delete: title author line, all `---` separators, the duplicate title `#` heading
- [ ] Fix section numbering: §3.4 → §3.3
- [ ] Fix table numbering: Tables 4/5/6 → Tables 3/4/5 (or keep 6 and insert a missing Table 3 placeholder)
- [ ] Convert all 5 figures to `\begin{figure}` environments with `\label`
- [ ] Convert all 6 tables to `booktabs` format with caption above
- [ ] Merge split random/clustered sub-tables (Tables 1 and 2) into single `table` environments
- [ ] Replace all Unicode math: `Δt`, `±`, `×`, `≈`, `←`
- [ ] Replace inline math expressions: `L/2 FLOP/byte`, `0.5 FLOP/byte`, `~0.5 ms`
- [ ] Wrap bold numbers in tables with `\mathbf{}` inside math mode
- [ ] Build and verify PDF — check figure placement and table alignment
