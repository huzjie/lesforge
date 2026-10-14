# bench — 四项基准

`run_all(cfg)` / `run(name, cfg)` / `summarize(results)`：

- `memory` — 声明规模的内存足迹（35B / 3B / 65.2 GiB）。
- `prefetch` — 预取命中率（核心指标）。
- `speedup` — 串行 vs LES 重叠的加速比。
- `loadbalance` — 专家路由的辅助损失。

每个基准用 `@BENCHMARKS.register(name)` 注册。
