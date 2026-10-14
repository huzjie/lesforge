# scheduler — LES 前瞻调度（核心）

- `les.py`：`LEScheduler`（编排）+ `LookaheadPredictor`（前瞻预测）。
- `tier.py`：`ResidencyTable`（RAM 常驻表 + LRU 访问序）。
- `offload.py`：`OffloadStore`（模拟 SSD，固定读延迟）。
- `prefetch.py`：`PrefetchPipeline`（在途预取队列，有界深度）。
- `budget.py`：`MemoryBudget`（MiB 记账）。
- `policy.py`：驱逐策略选择器（LRU 默认，LFU 回退 LRU）。

调度循环：`warmup(层0)` → 每层 `step`（补载缺失专家 + 预测下一层并预取 + 排空流水线 +
按预算驱逐）。命中率只统计 layer ≥ 1（冷启动不算调度失误）。
