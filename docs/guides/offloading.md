# 专家卸载指南

卸载 = 把大部分专家权重从 RAM 挪到 SSD/NAND，按需读入：

1. `schedule.resident_experts` 决定 RAM 常驻专家数；
2. `OffloadStore` 模拟 SSD，`read` 返回固定延迟（默认 12ms/专家）；
3. `LEScheduler` 保证「本层所需专家」常驻，其余按需读入；
4. `PrefetchPipeline` 提前预取下一层，把 I/O 藏进计算。

调参：常驻越多越快但吃内存；预取深度越大隐藏越多 I/O 但也吃内存。
