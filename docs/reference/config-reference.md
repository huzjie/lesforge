# 配置参考（完整字段）

见 [CONFIG.md](../CONFIG.md) 的参数表。这里是每个字段的取值域：

- `vision.img_size` / `patch_size`：`img_size % patch_size == 0`。
- `moe.n_layers`：≥ 2（否则无前瞻空间）。
- `moe.top_k`：1 ~ n_experts。
- `schedule.prefetch_depth`：≥ top_k 推荐。
- `schedule.quant_bits`：0 / 8 / 4。
- `schedule.eviction`：lru / lfu（lfu 当前回退 lru）。
