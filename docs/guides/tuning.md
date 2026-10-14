# 调参指南

| 目标 | 手段 |
|---|---|
| 提高命中率 | 调大 `lookahead` / `prefetch_depth`，或调小 `n_experts` |
| 省内存 | 调小 `resident_experts` / 加大 `quant_bits` 量化强度 |
| 提吞吐 | `prefetch_depth ≥ top_k`，保证预测全对也装得下预取 |
| 验证 | `python -m lesforge.cli train --epochs 20` 看命中率曲线是否单调 |

**黄金规则**：`prefetch_depth` 必须 ≥ `top_k`，否则即使预测全对，流水线也装不下全部
预取，命中率被瓶颈卡住。
