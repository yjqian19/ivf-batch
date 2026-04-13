# Experiment Results Analysis

## Setup

| Parameter | Value |
|-----------|-------|
| Dataset | SIFT1M (1M base vectors, 128-d) |
| Queries | 10,000 |
| Index | `IndexIVFFlat`, 256 clusters |
| nprobe | 8 |
| k | 10 |
| Recall@10 | 0.955 (all schedulers) |

All three scheduler 的 Recall@10 均为 0.955，满足 ≥ 0.90 的目标，说明调度策略**不影响搜索结果的正确性**。

---

## 1. Sequential Baseline

| Metric | Value |
|--------|-------|
| QPS | 3,709 |
| Wall time | 2.696s |
| Avg latency | 0.269ms |
| P95 latency | 0.347ms |
| P99 latency | 0.377ms |

单条查询延迟极低（~0.27ms），延迟分布集中（P99 仅比 avg 高 40%），作为所有后续比较的基准。

---

## 2. Time-Window Batching

### 2.1 关键发现：吞吐量没有显著提升

所有 28 种 (Δt, max\_batch\_size) 配置的 QPS 均在 **3,650 ~ 3,790** 之间，与 Sequential 基线（3,709）基本持平。批处理在单线程 CPU 上**没有带来吞吐量增益**。

原因分析：`faiss.IndexIVFFlat.search(n, k)` 在内部对 n 条 query 是逐条处理的（单线程下无法利用 SIMD 批量优势），因此把 query 打包成一个大 batch 调用并不比逐条调用更快。

### 2.2 延迟随 batch size 线性增长

| max\_batch\_size | Avg Latency | Avg Queue Delay |
|-----------------|-------------|-----------------|
| 32 | ~12-14ms | ~4-6ms |
| 64 | ~22-25ms | ~5-12ms |
| 128 | ~43-55ms | ~9-21ms |
| 256 | ~84-120ms | ~15-51ms |

延迟由两部分组成：
- **Queue delay**：query 在窗口中等待 flush 的时间
- **Exec time**：batch 执行时间（≈ batch\_size × 单条查询时间）

两者都随 batch size 增大而增大，导致 **batch 越大延迟越差**。

### 2.3 Δt 的影响较小

在 arrival QPS = 10,000 的负载下，batch 几乎总是被 **size trigger** 触发（因为查询到达非常密集）。例如 max\_batch\_size=32 时，平均 batch size ≈ 31.9，接近上限，说明 32 条 query 在 Δt 到期前就已经攒满了。因此 Δt 参数对结果影响不大。

### 2.4 结论

Time-window batching 在当前设置下**弊大于利**：吞吐量无提升，延迟显著恶化。小 batch（32）是延迟损失最小的选择，但仍不如 sequential。

---

## 3. Cluster-Based Batching

### 3.1 三种分组策略对比

| Grouping | QPS | Groups | Avg Group Size | Grouping Overhead |
|----------|-----|--------|----------------|-------------------|
| Primary centroid | 3,793 | 256 | 39.1 | 0.9ms |
| Jaccard ≥ 0.25 | 3,587 | 369 | 27.1 | 134.7ms |
| Jaccard ≥ 0.50 | 2,413 | 3,207 | 3.1 | 1,437.5ms |

### 3.2 Primary centroid 是最优策略

- QPS 3,793，略优于 sequential（+2.3%），是所有策略中最高的
- Quantizer + grouping 开销极低（共 8.7ms），几乎可以忽略
- 256 个分组（等于聚类数），每组平均 39 条 query，分布合理（std=14.4）

### 3.3 Jaccard 分组的开销不可接受

- **Jaccard ≥ 0.25**：grouping 耗时 134.7ms（占总时间 4.8%），QPS 反而低于 sequential
- **Jaccard ≥ 0.50**：grouping 耗时 **1,437.5ms**（占总时间 34.7%），QPS 暴跌至 2,413（-35%）。阈值过高导致大多数 query 无法合并，产生 3,207 个平均仅 3.1 条的小组，分组开销远超收益

Jaccard 分组的 O(n²) 级别开销在 10K query 规模下已经成为瓶颈。

---

## 4. Random vs Clustered 负载对比

| Scheduler | Random QPS | Clustered QPS |
|-----------|-----------|---------------|
| Sequential | 3,603 | 3,525 |
| Time-window (Δt=5ms, bs=128) | 3,706 | 3,602 |
| Cluster-batch (primary) | 3,761 | 3,624 |

### 4.1 Clustered 负载下分组效果显著

Clustered workload 下 cluster-batch 仅产生 **68 个组**（vs random 的 256 组），平均组大小 **147.1**（vs 39.1）。这说明聚类查询确实高度集中在少数倒排列表上，分组策略如预期生效。

### 4.2 但 QPS 提升依然有限

即使在 best-case 的 clustered 负载下，cluster-batch 相比 sequential 仅提升 **2.8%**。这进一步证实：在单线程 CPU 上，调度策略能利用的缓存局部性收益非常有限。

---

## 5. 总体结论

1. **Recall 不受影响**：三种策略的 Recall@10 完全一致（0.955），验证了 scheduler 仅改变查询执行顺序/批次，不改变搜索结果。

2. **单线程 CPU 上批处理收益极小**：Faiss 的 `IndexIVFFlat` 在单线程下对 batch search 的优化有限，吞吐量基本不随 batch size 变化。

3. **Primary centroid 分组是最实用的策略**：开销极低（<10ms），在所有配置中 QPS 最高，且在 clustered 负载下能有效减少分组数。

4. **Time-window batching 引入了不必要的延迟**：在没有吞吐量收益的情况下，排队等待只会增加每条 query 的响应时间。

5. **Jaccard 分组计算成本过高**：当前 O(n²) 实现在 10K query 下已不实用，需要更高效的算法或更小的查询规模才有意义。

### 后续方向

- 在**多线程 / GPU** 环境下重新测试，batch search 可能获得显著加速
- 对 cluster-batch 加入 **缓存命中率** 分析（`perf stat`），量化缓存局部性的实际改善
- 优化 Jaccard 分组算法，或尝试其他更高效的 overlap metric
- 测试更大的 nprobe 值（如 16, 32），此时倒排列表访问更多，缓存局部性收益可能更明显