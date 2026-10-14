# 架构（Architecture）

lesforge 是一套**端侧 MoE 视觉语言模型**的零依赖参考实现，复刻 MiniCPM-V 4.7
把 35B 参数 MoE VLM 压进约 3GB 手机内存的核心机制。整条推理链路：

```
图片 ──► ViT 编码器 ──► 投影层 ──► MoE 解码器（N 层）──► 文本
                                     │
                                     └── LEScheduler（前瞻专家调度）
                                          ├─ ResidencyTable   RAM 常驻表
                                          ├─ OffloadStore     SSD/NAND 卸载库
                                          ├─ PrefetchPipeline 预取流水线
                                          └─ LookaheadPredictor 下一层专家预测
```

## 目录结构

```
lesforge/
├── core/          # 注册表（backends / benchmarks）
├── utils/         # 确定性伪随机、向量、文本、yamlish
├── vision/        # ViT：patch 嵌入 + 多头注意力 + 编码器
├── moe/           # 路由 + SwiGLU 专家 + MoE 层 + 负载均衡损失
├── scheduler/     # ★ LES：分层常驻 + 前瞻预取 + LRU 驱逐 + 内存预算
├── model/         # 投影 + MoE 解码器 + VLM 引擎
├── backends.py    # mock / cpu / openai / vllm / transformers 五后端
├── train/         # 能力（skill）训练循环
├── bench/         # 内存 / 加速比 / 预取命中率 / 负载均衡
├── serving/       # 标准库 OpenAI 兼容 HTTP 服务
└── cli.py         # doctor / generate / train / bench / serve
```

## 数据流（一次前向）

1. **视觉编码**：`ViTEncoder` 把图片切成 patch、逐块投影、过 L 层注意力块，
   全局平均池化得到一个 `dim` 维向量。
2. **投影**：`Projector` 把视觉向量对齐到解码器维度。
3. **MoE 解码**：`MoEDecoder` 每层用 `TopKRouter` 给 token 打分选 top-K 专家，
   每个专家是独立的 SwiGLU FFN，按门控权重加权求和。
4. **LES 调度**：每层真正的专家选择被喂给 `LEScheduler`；调度器保证本层所需
   专家常驻 RAM，同时**提前预取下一层**所需专家，把 SSD 读延迟藏在计算里。
5. **生成**：贪心解码出文本。

## 设计原则

- **零依赖**：不装 numpy/pytorch，纯 Python 列表做张量，`random.Random` + md5
  做确定性伪随机，内置 `yamlish` 读配置。
- **确定性**：同一个 seed 下每次运行结果完全一致，方便基准复现。
- **可训练 mock**：能力用标量 `skill` 表达，训练信号单调趋 1（见 train/）。
- **单位即专家**：一个专家的权重矩阵是调度/卸载/量化的最小传输单位。
