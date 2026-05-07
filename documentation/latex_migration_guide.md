# LaTeX Migration Guide — ACM sigconf

## Current Status

The `.tex` and `.bib` files are already created at:

```
documentation/latex/final_report_acm.tex   ← complete ACM-formatted source
documentation/latex/references.bib         ← BibTeX entries
figures/                                    ← all 5 figures (referenced via \graphicspath)
```

---

## 1. Install MacTeX (full)

Download `MacTeX.pkg` (~4 GB) from https://tug.org/mactex/ and install.
No PATH configuration needed — the installer handles it.

> **Note:** BasicTeX has already been removed (`/usr/local/texlive/2026basic` and `/Library/TeX` deleted, fish PATH cleaned up).

---

## 2. Compile

```bash
cd documentation/latex
pdflatex final_report_acm.tex
bibtex   final_report_acm
pdflatex final_report_acm.tex
pdflatex final_report_acm.tex
```

Or use LaTeX Workshop in Cursor/VS Code: `Cmd+Shift+P` → **LaTeX Workshop: Build LaTeX project**.

The PDF opens automatically in the preview pane, and re-builds on save.

---

## 3. Review and Fix

Open `final_report_acm.pdf` alongside `final_report_acm.tex` and check:

- [ ] All 6 tables render correctly (column widths, alignment)
- [ ] All 5 figures appear and are legible at column width
- [ ] Math symbols render: $\Delta t$, $\pm$, $\times$, $\approx$, $\leftarrow$
- [ ] Bold QPS numbers in Tables 2 and 3 display correctly
- [ ] Cross-references (`\autoref`) resolve (requires the full 3-pass build above)
- [ ] Add `\cite{}` calls where needed and populate `references.bib`
- [ ] Build one final time and verify page count is reasonable for sigconf

---

## Reference: File Layout

```
documentation/
  latex/
    final_report_acm.tex
    references.bib
    *.aux  *.log  *.pdf  (build artifacts, gitignore these)
  latex_migration_guide.md
figures/
  fig_overview_qps.png
  fig_sweep.png
  fig_latency.png
  fig_latency_clustered.png
  fig_microbench.png
```
