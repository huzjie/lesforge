# vision — ViT 视觉塔

- `patchify.py`：`make_image`（合成灰度图）+ `PatchEmbed`（切 patch + 线性投影）。
- `attention.py`：`MultiHeadAttention`（max 减法 softmax 防溢出）。
- `vit.py`：`ViTEncoder`，L 层「pre-norm 注意力 + MLP」残差块，最后全局平均池化。

输出一个 `dim` 维向量，交给 `Projector` 对齐到解码器维度。
