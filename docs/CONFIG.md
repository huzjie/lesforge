# 配置说明

## 关键参数

| 分组 | 键 | 默认 | 含义 |
|---|---|---|---|
| vision | patch_size | 16 | 图像 patch 边长 |
| vision | img_size | 224 | 输入图像边长 |
| vision | dim | 128 | 视觉隐藏维度 |
| moe | n_experts | 32 | 每层专家数 |
| moe | top_k | 2 | 每 token 激活专家数 |
| moe | hidden | 512 | 专家 FFN 隐藏维度 |
| moe | total_params_b | 35.0 | 建模的总参数量（B） |
| moe | active_params_b | 3.0 | 建模的激活参数量（B） |
| schedule | lookahead | 1 | 预取前瞻层数 |
| schedule | resident_experts | 8 | RAM 常驻专家上限 |
| schedule | prefetch_depth | 2 | 在途预取流水线深度 |
| schedule | quant_bits | 4 | 卸载前量化位宽（0/8/4） |
| schedule | memory_budget_mb | 3000 | 内存预算（MiB） |
| schedule | eviction | lru | 驱逐策略（lru/lfu） |
| model | skill | 0.5 | 初始能力（预测器准确率） |

## 预设配置

- `configs/tiny.yaml` — 极小冒烟（秒级）。
- `configs/mobile.yaml` — 手机级预算（~3GB、INT4）。
- `configs/server.yaml` — 服务器级（全常驻、无卸载）。
- `configs/default.yaml` — 默认。

## 技巧

- **想让命中率曲线更陡**：调小 `n_experts`、调大 `top_k`。
- **想验证无量化开销**：`quant_bits: 0`。
- **prefetch_depth 必须 ≥ top_k**：否则即使预测全对，流水线也装不下全部预取。
