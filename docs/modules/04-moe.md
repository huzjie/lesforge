# moe — 路由 / 专家 / MoE 层

- `router.py`：`TopKRouter.forward(hidden) -> (indices, probs)`，确定性 top-K 门控。
- `expert.py`：`SwiGLUExpert`（W2 @ (SiLU(W1 x) * W3 x)），`quantize_expert` 伪量化。
- `layer.py`：`MoELayer`，路由 -> 分发 -> 加权求和；`selected_experts` 暴露路由决策。
- `load_balance.py`：Switch 式辅助损失 `N * sum(f_i * P_i)`。

**坑**：SwiGLU 的 W1/W3 是 (hidden × dim)、W2 是 (dim × hidden)，别写反，否则
matmul 静默错配（单测 `test_expert_shape` 会兜住）。
