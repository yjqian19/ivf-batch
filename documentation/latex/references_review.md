# References Review

Papers marked **[IN BIB]** are already in `references.bib`.
Papers marked **[ADD]** are recommended additions.
Papers marked **[SKIP]** are not directly cited in the final report.

---

## Group A — 直接相关：Batch Query Execution in Vector Search

这组论文和你的工作最直接相关——它们都研究把多个查询攒在一起执行、利用共享数据来提升吞吐量。

---

**[ADD] HQI: High-Throughput Vector Similarity Search in Knowledge Graphs**
Mohoney et al. — ACM SIGMOD, 2023 · https://arxiv.org/abs/2304.01926

这篇论文做的事和你的 Batch(MM) 几乎完全一致：把并发查询按它们需要扫描的 cluster 分组，对于每个 cluster，把所有需要它的查询堆成一个矩阵，然后用一次矩阵乘法批量计算距离，从而将扫描该 cluster 的开销均摊到整个 batch 上。这是目前最直接的 related work，必须引用，且应该在 §1 或 §2 中明确说明你们的区别（你们研究的是 GEMV vs GEMM 的 crossover 以及 workload 对最优 scan mode 的影响，HQI 没有做这个分析）。

```bibtex
@inproceedings{hqi2023,
  author    = {Mohoney, Jason and Waleffe, Roger and Xu, Henry and Rekatsinas,
               Theodoros and Venkataraman, Shivaram},
  title     = {High-Throughput Vector Similarity Search in Knowledge Graphs},
  booktitle = {Proceedings of the ACM SIGMOD International Conference on
               Management of Data},
  year      = {2023},
}
```

---

**[ADD] Quake: Adaptive Indexing for Vector Search**
Mohoney et al. — OSDI, 2025 · https://arxiv.org/abs/2506.03437

这篇论文系统地测量了 batch size 对 IVF 向量搜索吞吐量的影响，发现 batch size 从 1 增长到 10,000 时，FaissIVF 的吞吐量提升可达 6.7×。这和你论文的核心发现一致：batching 能显著提升 QPS。区别在于 Quake 关注的是 adaptive index 结构（在线分裂和合并 cluster），而你关注的是固定 index 下调度策略（MV vs MM）的选择以及 workload 对最优策略的影响。可在 §1 引用，用来支持"batching improves throughput"的论点。

```bibtex
@inproceedings{quake2025,
  author    = {Mohoney, Jason and Waleffe, Roger and Venkataraman, Shivaram
               and Rekatsinas, Theodoros},
  title     = {Quake: Adaptive Indexing for Vector Search},
  booktitle = {Proceedings of the 19th USENIX Symposium on Operating Systems
               Design and Implementation},
  year      = {2025},
}
```

---

**[ADD] Milvus: A Purpose-Built Vector Data Management System**
Wang et al. — ACM SIGMOD, 2021 · https://www.cs.purdue.edu/homes/csjgwang/pubs/SIGMOD21_Milvus.pdf

Milvus 把并发查询划分为若干 "query block"，每个 block 的向量总量恰好能放入 cache，从而在 block 内共享数据加载、减少 cache miss。这是工业界对同一问题（共享 inverted list 扫描）的解法，但 Milvus 依赖多线程和 cache-fitting 分块，而你的工作在单线程下研究 GEMV vs GEMM 的 scan mode 选择。可在 §1 引用作为 production system 的 related work。

```bibtex
@inproceedings{milvus2021,
  author    = {Wang, Jianguo and Yi, Xiaomeng and Guo, Rentong and Jin, Hai
               and Xu, Peng and Li, Shengjun and Wang, Xiangyu and Guo,
               Xiangzhou and Li, Chengming and Xu, Xiaohai and Yu, Kun and
               Yuan, Yuxing and Zou, Yinghao and Long, Jiquan and Cai, Yihua
               and Li, Zhenxiang and Zhang, Zhifan and Mo, Yihua and Gu,
               Jingyu and Jiang, Ruiyi and Wei, Yi},
  title     = {{Milvus}: A Purpose-Built Vector Data Management System},
  booktitle = {Proceedings of the ACM SIGMOD International Conference on
               Management of Data},
  year      = {2021},
}
```

---

**[ADD] GPU-Native ANN Search with IVF-RaBitQ**
NTU / NVIDIA cuVS — arXiv, 2025 · https://arxiv.org/abs/2602.23999

这篇论文明确指出：把多个 IVF 查询攒在一起执行，能将每个 inverted list 上的距离计算从 GEMV 变成 GEMM，暴露出规整的矩阵计算模式，并使 GPU 的 coalesced memory access 成为可能。这直接验证了你论文的核心假设：L 的增大使 GEMM 优于 GEMV。区别在于它在 GPU 上实现，而你在单线程 CPU 上做了更细粒度的 GEMV vs GEMM crossover 分析（找到 L=8 的 crossover point）。可在 §2.1.3 引用，支持 GEMM 的 arithmetic intensity 分析。

```bibtex
@misc{ivfrabitq2025,
  author        = {{NTU/NVIDIA cuVS Team}},
  title         = {{GPU}-Native Approximate Nearest Neighbor Search with
                   {IVF-RaBitQ}},
  year          = {2025},
  eprint        = {2602.23999},
  archivePrefix = {arXiv},
}
```

---

## Group B — IVF Index & Vector Search Foundations

这组论文是论文 §1 Introduction 里提到的技术的原始来源，需要对应的 \cite{} 调用。

---

**[IN BIB] Billion-Scale Similarity Search with GPUs (Faiss)**
Johnson, Douze, Jégou — IEEE Transactions on Big Data, 2019 · `faiss2021`
→ §2.1.1 "Our engine follows the IVF design from Faiss" 处引用。注意 bib 里年份写的是 2021，实际发表年是 2019，建议修正。

---

**[IN BIB] Efficient and Robust ANN Search Using HNSW**
Malkov, Yashunin — IEEE TPAMI, 2018 · `hnsw2020`
→ §1 "Hierarchical Navigable Small World graphs (HNSW)" 处引用。

---

**[ADD] Product Quantization for Nearest Neighbor Search**
Jégou, Douze, Schmid — IEEE TPAMI, 2011

这篇论文提出了 Product Quantization（乘积量化），将向量压缩为短码再做近似距离计算，是 Faiss 等系统的核心组件之一。同时，SIFT1M 数据集也是在这篇论文中作为标准 ANN benchmark 正式引入的。§1 "product quantization" 和 §2.2.1 "SIFT1M benchmark" 两处都应引用这篇。

```bibtex
@article{pq2011,
  author  = {J{\'e}gou, Herv{\'e} and Douze, Matthijs and Schmid, Cordelia},
  title   = {Product Quantization for Nearest Neighbor Search},
  journal = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {33},
  number  = {1},
  pages   = {117--128},
  year    = {2011},
}
```

---

## Group C — Classical DB Batch Scan (背景支撑)

---

**[ADD] Cooperative Scans: Dynamic Bandwidth Sharing in a DBMS**
Zukowski, Héman, Nes, Boncz — VLDB, 2007

经典数据库论文，提出让多个并发查询共享同一次顺序扫描，每页数据只从磁盘/内存读一次、供所有需要它的查询使用，即"load once, serve many"。这和你的 per-list batch scan 是完全相同的原理——每个 inverted list 只加载一次，供 batch 内所有需要它的查询使用。可在 §1 或 §2.1.2 用一句话引用，将你的方法与 DB 领域已有原理联系起来。

```bibtex
@inproceedings{coopscans2007,
  author    = {Zukowski, Marcin and H{\'e}man, S{\'a}ndor and Nes, Niels
               and Boncz, Peter},
  title     = {Cooperative Scans: Dynamic Bandwidth Sharing in a {DBMS}},
  booktitle = {Proceedings of the 33rd International Conference on Very
               Large Data Bases},
  pages     = {723--734},
  year      = {2007},
}
```

---

## Group D — 可跳过

以下论文在 proposal 的 reference.md 里有，但最终 report 正文没有直接涉及其内容，可以不引用：

- **Manu** (PVLDB 2022) — 云原生架构，与本文无直接关联
- **PQ Fast Scan** (PVLDB 2015) — SIMD 优化，论文没有讨论 SIMD
- **ScaNN** (ICML 2020) — 量化+SIMD，同上
- **Vearch/Gamma** (arXiv 2019) — 生产系统，关联较弱

---

## 汇总：建议加入 references.bib 的条目

| Key | 论文 | 引用位置 |
|---|---|---|
| `hqi2023` | HQI SIGMOD 2023 | §1 related work, §2.1.3 |
| `quake2025` | Quake OSDI 2025 | §1 ("batching improves throughput") |
| `milvus2021` | Milvus SIGMOD 2021 | §1 related work |
| `ivfrabitq2025` | IVF-RaBitQ arXiv 2025 | §2.1.3 GEMM discussion |
| `pq2011` | Product Quantization TPAMI 2011 | §1, §2.2.1 dataset |
| `coopscans2007` | Cooperative Scans VLDB 2007 | §1 or §2.1.2 |
