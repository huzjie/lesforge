# 性能剖析（Performance）

## 复杂度

- **路由**：每 token O(n_experts · dim)。
- **专家 FFN**：O(dim · hidden)，仅 top_k 个专家被激活。
- **LES 调度**：每层 O(top_k + prefetch_depth)，与专家总数无关——这是它能跑在手机上的
  原因：调度开销可忽略，I/O 又藏在计算里。

## 加速比上界

设单层计算 `C`、单专家 SSD 读 `I`、激活 `k` 个专家：

- 串行：`C + k·I`
- LES 理想（零 stall）：`C`（层 0 冷启动除外）
- 加速比上界：`1 + k·I/C`

`I/C` 越大（SSD 越慢、专家越大），LES 收益越显著。
