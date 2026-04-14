# New Engine Summary

Commit: `1dd56cc` — "new engine" (Apr 13 2026)

Files changed: `engine/custom_index.py` (new), `engine/schedulers.py`, `run_experiments.py`, `validate_custom.py` (new)

---

## 解决的核心问题

之前所有调度器直接调用 Faiss 的 `index.search()`，这是一个黑箱：调度器无法控制内部的扫描顺序。无论 batch 如何排列，Faiss 对每个 query 独立处理，**同一个 inverted list 可能被多个 query 反复从内存加载**，产生大量重复 cache miss。

这个 commit 用 `CustomIVFIndex` 替换了 Faiss 黑箱，实现完全自主的 IVF 搜索引擎，使调度器能够控制 inverted list 的扫描顺序，从而演示 cache-locality 带来的真实收益。

---

## `engine/custom_index.py` 的作用

### `build_custom_index(base_vectors, n_clusters)`
用 Faiss Kmeans 训练聚类中心，然后把所有向量分配到最近的 centroid，构建 Python 原生的倒排列表结构（`inverted_lists`、`vector_ids`、预计算的 L2 norm）。返回一个 `CustomIVFIndex` 实例。

### `quantizer_search(queries, nprobe) → (n, nprobe) int64`
独立暴露 centroid 查找步骤。之前这一步藏在 Faiss 内部，现在调度器可以在执行搜索之前拿到每个 query 的 probe 列表，是 Scheduler 3（cluster-based grouping）的前提。

### `search_one(query, k, nprobe)` — 顺序基线
```
query → 找最近 nprobe 个 centroid → 逐个扫描 inverted list → topk
```
行为等价于原 Faiss `index.search(q[i:i+1])`，用于 Scheduler 1（Sequential）。

### `search_batch_per_list(queries, centroid_ids, k)` — 核心创新
```
n queries → 建倒排映射 {list_id → [需要该 list 的所有 query]}
           → 每个 inverted list 只从内存加载一次
           → 所有需要它的 query 趁热批量计算 L2 距离
           → 各 query 独立取 topk
```

**关键思路**：把"以 query 为主轴"改成"以 inverted list 为主轴"。一个 list 加载进 CPU cache 后，所有关心它的 query 立刻处理完，避免重复 cache miss。batch 越大、query 间 centroid 重叠越多，cache 利用率越高。这正是 Scheduler 2 和 3 的 batch 带来收益的根本机制。

---

## Schedulers 的对应变化

| 调度器 | 之前 | 之后 |
|---|---|---|
| Sequential | `faiss.index.search(q[i:i+1])` | `index.search_one(q[i], k, nprobe)` |
| Time-window | `index.search(batch_q)` | `quantizer_search` + `search_batch_per_list` |
| Cluster-batch | 同上 + `faiss.downcast_index(index.quantizer)` | 同上，grouping 在 quantizer 之后，无需 faiss 内部 API |

移除了 `index.nprobe = nprobe` 的全局设置，改为每次调用时显式传参，更安全。

---

## `validate_custom.py` 的作用

新增的验证脚本，用于确认 `search_one` 与 `search_batch_per_list` 在相同 query 上返回一致结果（result agreement），防止两种路径因实现差异产生不同输出，是正确性保障。

---

## `run_experiments.py` 结构

脚本顺序执行 4 个实验段：

**段 1 — Sequential baseline**
跑全量 query，输出 Recall@10、QPS、wall time、avg/P95/P99 延迟，作为对比基准。

**段 2 — Time-window 参数扫描**
模拟 2000 QPS 泊松到达流，对 `Δt ∈ {0.5, 1, 2, 5, 10, 20, 50} ms × max_batch_size ∈ {32, 64, 128, 256}` 做网格搜索（28 组），每组记录 avg batch size、QPS、Recall、avg 延迟、P95 延迟、avg 排队延迟。

**段 3 — Cluster-batch 参数扫描**
相同参数网格，额外输出两列：`Groups`（batch 内形成的 centroid-group 数量）和 `AvgGrp`（平均 group 大小），用于衡量 cluster 聚合有效性。当前只运行 `grouping="primary"`（以第一个 centroid 分组）。

**段 4 — Random vs Clustered workload 对比**
固定参数（`Δt=5ms, max_batch=128`），在随机 query 和聚类 query（`generate_clustered_queries`，10 个区域）两种 workload 下，对比三个调度器的 QPS 和 wall time。聚类 workload 是 Scheduler 3 的最佳案例，随机 query 是最差案例，直接验证核心假说。
