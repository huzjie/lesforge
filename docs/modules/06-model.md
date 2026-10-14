# model — 投影 / 解码器 / 引擎

- `projector.py`：视觉 -> 解码器维度的线性对齐。
- `decoder.py`：`MoEDecoder`，token 嵌入 + N 层 MoE + LM head。
- `engine.py`：`VLMEngine`，端到端：ViT -> 投影 -> MoE 解码，全程受 LES 调度。

`VLMEngine.generate` 返回 `{text, stats, n_layers}`；`stats` 含命中率、stall 次数、
compute/io 时间、常驻专家数。
