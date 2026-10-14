# 规格：基准

- 全部确定性、可复现（随机走 utils.hash）。
- `prefetch`：命中率 + 随机基线 + 增益。
- `speedup`：串行 vs 重叠，附 stall 次数。
- `memory`：声明规模的内存足迹。
- `loadbalance`：专家路由辅助损失。
