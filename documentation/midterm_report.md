# Mid-Term Report: Query Scheduling Optimization in IVF-based Vector Search

Yujia Qian — yjqian19@mit.edu\
Xiangyu Guan — xiang949@mit.edu

---

## 1. Project Status

**Tasks**

| Task | Complete | Pending |
|---|---|---|
| Core engine | custom engine based on FAISS package, single-threaded  | — |
| Schedulers | Sequential, Time-Window, Cluster-Based | Optimize Cluster-Based Method |
| Evaluation | scheduler parameter sweep, workload comparison (random vs. clustered) | Complete cluster query testset and sweep, multi-run, add metrics |

**Deliverables**

| Deliverable | Complete | Pending |
|---|---|---|
| A working system | all | — |
| Measured performance differences | First-round results on SIFT1M | Mentioned above in the evaluation part |
| Final report and video | — | all |

---

## 2. Experiment Setup

- **Dataset:** SIFT1M (1M × 128-d, L2 distance)
- **Device:** MacBook Pro, Apple M3 Pro, 18 GB RAM
- **Search parameters:** n_clusters=256, nprobe=8, k=10
- **Scheduler parameters:** Δt=5ms, MaxBS=128 fixed for comparison; separately swept as Δt ∈ {0.5, 1, 2, 5, 10, 20, 50} ms × MaxBS ∈ {32, 64, 128, 256} under random queries (full results recorded in a separate report)
- **Queries:** 10K queries from SIFT1M, arrival rate 2000 QPS (Poisson), tested under two workloads:
  - *Random* — queries sampled uniformly, probing largely disjoint inverted lists
  - *Clustered* — queries drawn from 10 spatial regions, with high centroid overlap across queries

  We test both because the benefit of cluster-based batching depends entirely on centroid overlap between concurrent queries — random is the worst case, clustered is the best case.

## 3. Results

### Random Queries

| Scheduler | Recall@10 | QPS | Avg Lat | P95 Lat |
|---|---|---|---|---|
| Sequential | 0.956 | 1836 | 0.5 ms | 0.7 ms |
| Time-Window | 0.956 | 2257 (+23%) | 63.7 ms (×117) | 93.0 ms (×126) |
| Cluster-Batch | 0.956 | 1954 (+6%) | 94.8 ms (×174) | 126.5 ms (×171) |

All three schedulers achieve identical Recall@10=0.956. Time-Window gains +23% QPS over Sequential at a 117× latency cost. Cluster-Batch gains only +6% QPS with higher latency overhead, due to near-singleton groups on random queries.

### Clustered Queries

| Scheduler | QPS | Avg Lat | P95 Lat |
|---|---|---|---|
| Sequential | 1943 | 0.5 ms | 0.7 ms |
| Time-Window | 2351 (+21%) | 52.4 ms (×102) | 93.5 ms (×133) |
| Cluster-Batch | 2337 (+20%) | 55.4 ms (×108) | 104.5 ms (×149) |


Both schedulers gain +20–21% QPS, versus only +4% for Cluster-Batch on random queries — confirming that spatial locality significantly amplifies batching benefit. Recall is not reported: clustered queries are sampled from base vectors, so self-retrieval makes it undefined.

### Cluster-Batch: Random vs. Clustered Workload

| Workload | QPS | Groups | Avg Group Size |
|---|---|---|---|
| Random | 1954 | 7452 | 1.3 |
| Clustered | 2337 (+20%) | 158 (−98%) | 63.3 (+49×) |


On clustered queries, group count drops by 98% and average group size grows 49×, which explains the +20% QPS gain. On random queries, near-singleton groups mean the grouping step adds overhead with no sharing benefit, leaving QPS nearly unchanged from Sequential.


---

## 4. Potential Problems

1. **Clustered query workload construction is incomplete.**
The current clustered queries are sampled directly from the base vectors, meaning each query already exists in the index. As a result, the index will always retrieve the query itself as the nearest neighbor (distance = 0), which does not reflect a realistic search scenario and makes recall undefined without excluding self-matches. The current results show promising trends but are based on a preliminary workload construction. A more realistic clustered workload — where queries are near but distinct from base vectors — remains to be built.

2. **Cluster-based scheduler has room for improvement.**
The primary-centroid grouping degenerates on random workloads (AvgGrp 1.1–1.7). We are exploring Jaccard-similarity grouping as an alternative, which is implemented but not yet fully tested.

3. **Results are based on a single run.**
All numbers come from one run. Multi-run experiments (mean ± std) would improve statistical credibility.

4. **Parameter sweep was only conducted under random queries.**
The clustered workload comparison uses a fixed config. Running the same sweep under clustered conditions may reveal different optimal parameters.

5. **Additional metrics could strengthen the argument.**
Current metrics (QPS, recall, latency) do not directly demonstrate the cache-locality effect that motivates this project. Two additions would be most meaningful: (1) *Cache miss rate* — measures how often the CPU has to fetch data from main memory rather than cache; a lower rate for larger batches would directly validate the per-list reuse argument. (2) *Latency decomposition for Scheduler 3* — breaking total latency into queue delay, quantizer time, grouping time, and scan time would show exactly where the overhead goes on random vs. clustered workloads, and whether grouping or fragmented execution is the bottleneck.

---

## 5. Updated Timeline

The project is on schedule. Phase 1 work (Mar 27 → Apr 14) is complete as planned.

| Phase | Dates | Tasks |
|---|---|---|
| Phase 1 | Mar 27 → Apr 14 ✓ | Core engine, schedulers, dataset, first round evaluation |
| Phase 2 | Apr 14 → May 3 | Optimize scheduling strategy, refine evaluation setup |
| Phase 3 | May 3 → May 7 | Compose report and prepare video |

---

## 6. Division of Work

| Member | So Far | Next |
|---|---|---|
| Xiangyu | schedulers, core engine improvement, evaluation | scheduler improvement, evaluation |
| Yujia | core engine base, evaluation, report | evaluation, report |

---

## 7. Questions
