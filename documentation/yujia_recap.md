# Yujia Recap

---

## 1. Engine 实现

### 1.1 整体架构

引擎分两层：

**`engine/index.py` — Faiss 封装（仅用于验证）**
对 `faiss.IndexIVFFlat` 的最小封装，Faiss 内部处理 quantizer、list scan 和距离计算，是一个黑箱。`faiss.omp_set_num_threads(1)` 在模块 import 时设置，禁用 Faiss 内部并行，确保调度器实验不受多线程干扰。仅用于初始正确性验证。

**`engine/custom_index.py` — 自定义 IVF 引擎（实验全程使用）**
用 Python 原生实现替换 Faiss 黑箱，显式存储 centroid、inverted list 和预计算的 L2 norm，并将 quantizer search 和 list scan 暴露为独立接口，供调度器自由组合。这是实现 per-list batch scan 的前提。

两层共享相同的输出约定：给定 queries，返回 `(distances, ids)`，shape 均为 `(n, k)`。

### 1.2 CustomIVFIndex 数据结构

构建时，`build_custom_index` 调用 Faiss k-means 训练 centroid，再将所有 base vector 分配到最近 centroid，构建如下结构：

| 字段 | Shape | 说明 |
|---|---|---|
| `centroids` | (n_clusters, d) | k-means 训练得到的聚类中心 |
| `inverted_lists` | list of (n_c, d) | 每个 cluster 中存储的向量 |
| `vector_ids` | list of (n_c,) | 每个 cluster 中向量的原始数据库 ID |
| `centroid_norms` | (n_clusters,) | 预计算的 `‖centroid‖²`，加速 L2 计算 |
| `list_norms` | list of (n_c,) | 预计算的每个 list 向量的 `‖v‖²` |

L2 距离展开为 `‖q−v‖² = ‖q‖² + ‖v‖² − 2(q·v)`。‖v‖² 在构建时预计算，search 时只需做点积，节省重复计算。

### 1.3 以一个 Query 为例：函数调用流程

假设现在来了一个 query 向量 `q`，我们想找它在数据库中最近的 10 个邻居（k=10），探测 8 个 cluster（nprobe=8）。

**第一步：`quantizer_search(q, nprobe=8)`**

```
q → 计算 q 到全部 256 个 centroid 的 L2 距离
  → 用 argpartition 找最近的 8 个 centroid
  → 返回这 8 个 centroid 的 ID，例如 [42, 17, 203, 5, ...]
```

可以理解为：先在地图上找到 q 所在的 8 个"区域"，后续只在这些区域里找人。

**第二步：扫描 inverted list**

对每个选出的 centroid ID（比如 42），取出 `inverted_lists[42]`——这是所有被分配到 cluster 42 的向量，大约 4000 条。计算 q 到每条向量的 L2 距离：

```
d = ‖q‖² + list_norms[42] - 2 × (inverted_lists[42] @ q)
```

8 个 cluster 扫完后，合并所有候选距离。

**第三步：`_topk(dists, ids, k=10)`**

从所有候选（约 8 × 4000 = 32000 条）中选出距离最小的 10 个，用 `argpartition` 做 O(n) 选取再排序，返回 `(distances[10], ids[10])`。

以上三步合在一起就是 **`search_one(q, k=10, nprobe=8)`**，是 Scheduler 1 的基本单元。

### 1.4 Batch 场景：`search_batch_per_list` 的改进

当同时来了 100 个 query 时，`search_one` 的做法是把上面三步重复 100 次——cluster 42 的 inverted list 会被加载 100 次（每次 query 用完就从 cache 里挤走）。

`search_batch_per_list` 的思路是把顺序倒过来，**以 list 为主轴**：

```
1. 先对所有 100 个 query 做 quantizer_search，得到每个 query 要探测哪些 cluster
2. 构建倒排映射：cluster 42 → [query 3, query 17, query 58, ...]
3. 对每个 cluster（只加载一次）：
       把需要它的所有 m 个 query 组成矩阵 Q (m, d)，
       做一次 GEMM：vecs (n_c, d) @ Q.T (d, m) → dots (n_c, m)，
       再按列取出每个 query 的距离向量  ← list 数据在 L1/L2 cache 里被所有 m 个 query 复用
4. 对每个 query 汇总候选 → top-k
```

cluster 42 的 4000 条向量只从内存读一次，但同时为多个 query 服务。

**GEMV vs GEMM 的区别**

- **GEMV**（GEneral Matrix-Vector multiply，矩阵 × 向量）：形如 `(n_c, d) @ (d,) → (n_c,)`，对单个 query 计算距离。`vecs` 的每一行从内存读出来，乘一次就用完，arithmetic intensity 是 O(1)——计算量和内存访问量几乎等比，完全受内存带宽瓶颈，CPU 的 SIMD 算力大量闲置。
- **GEMM**（GEneral Matrix-Matrix multiply，矩阵 × 矩阵）：形如 `(n_c, d) @ (d, m) → (n_c, m)`，对同一个 cluster 的 m 个 query 一次性计算。BLAS 的 GEMM 实现会做 register blocking 和 cache blocking：把 `vecs` 的一个小 tile 装进 L1 cache，对所有 m 个 query 都用它算一遍再换下一个 tile，arithmetic intensity 是 O(m)。m 越大，每次内存访问摊薄的计算越多，AVX/AVX-512 越能跑满。

query 之间 centroid 重叠越多（m 越大），GEMM 收益越大。random workload 下 m ≈ 1–3，优势有限；clustered workload 下 m 可达几十甚至上百，性能差距显著。

---

## 2. Scheduler 实现逻辑

### Scheduler 1 — Sequential

按到达顺序逐条处理，每个 query 独立调用 `search_one()`。无 batch，无排队，无跨 query 共享。作为正确性和延迟的基准。

```
for each query i:
    search_one(query[i], k, nprobe) → (dists, ids)
```

### Scheduler 2 — Time-Window Batching

用虚拟时钟（`sim_time`）积累 query，双触发 flush：

- **时间触发**：距窗口开启已过 Δt
- **大小触发**：batch 已达 MaxBS

flush 后，整个 batch 先做 `quantizer_search()` 获取每个 query 的 centroid IDs，再调用 `search_batch_per_list()`，每个 inverted list 只加载一次，所有需要它的 query 批量计算距离。

```
while queries remain:
    open window; deadline = sim_time + Δt
    collect queries until deadline OR batch full
    flush → quantizer_search → search_batch_per_list
    record queue_delay + exec_time per query
```

### Scheduler 3 — Cluster-Based Batching

在 Scheduler 2 的 batch 收集逻辑基础上，在 quantizer search 和 list scan 之间插入一个分组步骤：

1. **Quantizer search**：对整个 batch 获取每个 query 的 centroid IDs
2. **分组**：按 centroid 重叠划分 sub-group
   - *Primary*：共享同一最近 centroid 的 query 归为一组，O(n)
   - *Jaccard*：贪心分组，probe set Jaccard 相似度 ≥ threshold 的 query 归为一组
3. **Search**：每个 sub-group 独立调用 `search_batch_per_list()`

```
while queries remain:
    collect batch（同 Scheduler 2 的双触发逻辑）
    quantizer_search(batch)            → centroid_ids
    group queries by centroid overlap  → sub-groups
    for each sub-group:
        search_batch_per_list(sub-group)
```

---

## 3. 实验参数说明

### 索引核心参数

| 参数 | 含义 |
|---|---|
| `n_clusters` | 向量空间被 k-means 划分成的 Voronoi 单元数量。clusters 越多，每个 inverted list 越短，扫描越快，但量化精度越粗。本实验固定为 256。 |
| `nprobe` | 每个 query 在搜索时探测的最近 centroid 数量。nprobe 越高，召回率越高，搜索越慢。本实验固定为 8。 |
| `k` | 每个 query 返回的最近邻数量，即 top-k。评估指标为 Recall@k，本实验 k=10。 |

### 调度器参数（Scheduler 2 & 3 的 sweep 参数）

| 列名 | 全称 | 含义 |
|---|---|---|
| `Δt` | Time window | 时间窗口大小（ms）。调度器最多等待 Δt 时间再 flush 一个 batch。Δt 越大，积累的 query 越多，batch 越大，但排队等待时间也越长。 |
| `MaxBS` | Max batch size | batch 大小的硬上限。当 batch 中的 query 数量达到 MaxBS 时，立即 flush，不等 Δt。两个 trigger 哪个先触发就执行哪个。 |

### Cluster-Batch 专属参数

| 参数 | 含义 |
|---|---|
| `grouping` | batch 内部的分组策略：`primary`（以最近 centroid 分组）或 `jaccard`（以 probe set 的 Jaccard 相似度分组）。 |
| `jaccard_threshold` | Jaccard 分组的阈值。两个 query 的 probe set Jaccard 相似度 ≥ 该阈值才被放入同一组。 |

### 到达模型参数

| 参数 | 含义 |
|---|---|
| `arrival_rate` (QPS) | 目标 query 到达速率，本实验为 2000 QPS。使用泊松过程模拟，inter-arrival time 服从 Exp(1/λ)。 |

### 结果指标

**吞吐与正确性**

| 指标 | 全称 | 含义 |
|---|---|---|
| `QPS` | Queries Per Second | 系统每秒处理的 query 数量 = 总 query 数 / wall time。吞吐量的主要衡量指标。 |
| `Recall@k` | Recall at k | 在返回的 top-k 结果中，真实最近邻出现的比例。本实验三个调度器均为 0.956。 |
| `Wall time` | Wall clock time | 完成全部 query 所用的实际时间（秒）。 |

**延迟指标**

| 指标 | 全称 | 含义 |
|---|---|---|
| `AvgLat` | Average latency | 每个 query 从到达到返回结果的平均端到端时间 = AvgQD + exec_time。 |
| `P95Lat` | P95 latency | 第 95 百分位延迟。95% 的 query 在此时间内完成。比均值更能反映尾延迟。 |
| `P99Lat` | P99 latency | 第 99 百分位延迟。仅 Sequential 报告，衡量极端尾延迟。 |
| `AvgQD` | Average queue delay | query 等待 batch flush 的平均时间 = flush_time − arrival_time。AvgLat − AvgQD ≈ 实际执行时间。 |

**Cluster-Batch 专属指标**

| 指标 | 全称 | 含义 |
|---|---|---|
| `AvgBS` | Average batch size | 每次 flush 的 batch 平均大小。AvgBS 越接近 MaxBS，说明 size trigger 是主导。 |
| `Groups` | Number of groups | 整个实验中所有 batch 产生的 sub-group 总数。Groups 越接近总 query 数，说明分组退化为单例，聚合无效。 |
| `AvgGrp` | Average group size | 每个 sub-group 的平均 query 数量。越大说明 centroid 重叠越多，cache 复用收益越高。Random workload 下约 1.3，clustered workload 下约 63.7。 |

**Scheduler 3 延迟分解**

```
latency = queue_delay + quantizer_time + grouping_time + exec_time
```

| 分量 | 含义 |
|---|---|
| `queue_delay` | query 等待 batch 收集完成的时间 |
| `quantizer_time` | 对整个 batch 做 centroid 查找的时间 |
| `grouping_time` | 对 batch 内 query 做分组计算的时间 |
| `exec_time` | 对各 sub-group 执行 `search_batch_per_list` 的时间 |

---

## 4. `run_experiments.py` 执行流程

### 两种 Query 的来源

**Random query**
直接使用 SIFT1M 数据集自带的 `sift_query.fvecs`，共 10,000 条，由数据集发布方预先采集，均匀分布在向量空间中。不同 query 之间探测的 centroid 几乎没有重叠，是 cluster-based batching 的最差情形。

**Clustered query**
由 `generate_clustered_queries()` 从 index 内部生成：随机选出 10 个 cluster，从每个 cluster 的 inverted list 里均匀采样 1000 条向量作为 query，共 10,000 条。这些 query 天然来自同一空间区域，会探测高度重叠的 centroid，是 cluster-based batching 的最佳情形。

> 注意：clustered query 实际上是从 base vector 里采出来的，和 random query 来自不同的源，也不存在预计算的 ground truth，所以 Section 4 的 workload 比较只报告 QPS，不报告 Recall。

### Query 如何"发送"

Sequential 不模拟到达时间，直接逐条处理。

Time-Window 和 Cluster-Batch 需要模拟真实服务器场景——query 不是一次性全部到来，而是随时间陆续到达。我们用两个机制来模拟这个过程：

**泊松到达序列（Poisson arrivals）**

现实中，用户请求的到达往往是随机的、互相独立的，泊松过程是描述这类现象的标准模型。具体做法是：先随机生成每两条 query 之间的间隔时间（inter-arrival time），间隔服从指数分布 Exp(1/λ)，均值为 1/λ 秒。λ=2000 时平均每 0.5ms 来一条。再对所有间隔做累加，得到每条 query 的绝对到达时刻：

```
inter_arrivals ~ Exp(1/2000)          ← 每条 query 距上一条的随机间隔
arrival_times  = cumsum(inter_arrivals) ← 每条 query 的绝对到达时刻（秒）
```

这样 10,000 条 query 就有了各自的"几点几秒到达"的时间戳。

**虚拟时钟（`sim_time`）**

如果真的按时间戳等待，跑完整个实验需要 10000/2000 = 5 秒，还得真实地 sleep。虚拟时钟的思路是：不真实等待，而是用一个变量 `sim_time` 记录"当前模拟时间推进到了哪里"，按 arrival_times 的逻辑决定哪些 query 应该落入同一个时间窗口，执行完一个 batch 后把执行耗时加到 `sim_time` 上继续推进。这样既保留了真实到达顺序和时间间隔的统计特性，又不需要真实等待，整个实验几秒内就能跑完。

latency 的计算依赖虚拟时钟：`queue_delay = flush_time − arrival_time[i]`，反映的是模拟时间轴上的等待，而非 wall clock。

### 四段实验

**段 1 — Sequential 基线**
用 random query，逐条调用 `search_one()`，记录每条 query 的执行时间、总 QPS、wall time、avg/P95/P99 延迟。不使用 arrival_times。

**段 2 — Time-Window 参数扫描**
用 random query + Poisson 到达（2000 QPS）。
对 Δt ∈ {0.5, 1, 2, 5, 10, 20, 50} ms × MaxBS ∈ {32, 64, 128, 256} 共 28 组做网格搜索，每组记录 AvgBS、QPS、Recall、AvgLat、P95Lat、AvgQD。

**段 3 — Cluster-Batch 参数扫描**
用 random query + 同一组 Poisson 到达时间。
相同 28 组参数，额外记录 Groups（sub-group 总数）和 AvgGrp（平均 group 大小）。当前只测 `grouping="primary"`。

**段 4 — Workload 对比**
固定参数（Δt=5ms, MaxBS=128），对 random query 和 clustered query 分别跑三个调度器，比较 QPS 和 wall time。这一段的目的是验证核心假说：clustered workload 下 cluster-based batching 是否真的优于 time-window batching。
