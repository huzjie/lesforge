# lesforge — 端侧 MoE 视觉语言模型的前瞻专家调度（LES）

> 把 35B 参数的 MoE 视觉语言模型压进 ~3GB 手机内存，靠的不是压缩，而是**让 SSD 读
> 延迟消失在计算里**。这是面壁智能 × OpenBMB 开源 MiniCPM-V 4.7 的核心工程手段，
> 本仓库是其**零依赖参考实现**：一行第三方依赖都不装，就能把「前瞻专家调度」跑起来、
> 训起来、测起来、照抄到自己项目里。

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![License](https://img.shields.io/badge/License-Apache--2.0-orange)](#)
[![Deps](https://img.shields.io/badge/dependencies-zero-brightgreen)](#)

---

## 一句话原理

MoE 每层只激活 top-K 个专家，其余全在 SSD 上。下一层要激活哪些专家，**在上一层计算
时就能预测出来**——于是「一边算当前层、一边把下一层专家提前读进 RAM」，让计算和 I/O
重叠，SSD 延迟被完全藏起来。这就是 **Lookahead Expert Scheduling（LES）**，MiniCPM-V 4.7
在 iPad Pro M4 上「平均提速约 5 倍、输出不变」的来源。

```
串行（无 LES）：[读L][算L][读L+1][算L+1]...   ← I/O 全暴露，每层都卡

LES 重叠：      [算L]─────────►[算L+1]──────►
                 └[预取L+1]┘      └[预取L+2]┘  ← I/O 藏在计算里
```

## 核心能力

| 能力 | 说明 |
|---|---|
| **专家分层常驻** | 只留 `resident_experts` 个专家在 RAM，其余写进 SSD/NAND，按需读入 |
| **前瞻预取（LES）** | `LookaheadPredictor` 预测下一层 top-K 专家，`PrefetchPipeline` 非阻塞预取 |
| **内存预算 + LRU 驱逐** | `MemoryBudget` 按 MiB 记账，`ResidencyTable` 维护访问序、超预算即驱逐 |
| **伪量化** | INT4/INT8 确定性伪量化，直接量化出真实存储/传输成本 |
| **可训练 mock** | 能力 = 标量 `skill`，训练信号单调趋 1，预取命中率从随机基线爬到 1.0 |
| **五后端** | mock / cpu / openai / vllm / transformers，同一接口 |
| **OpenAI 兼容服务** | 纯标准库 HTTP，`/v1/chat/completions` 可直接被任何客户端调用 |

## 实现思路（可复用）

1. **把「专家」当调度的最小单位**。卸载、预取、量化、驱逐都以单个专家的权重矩阵为
   粒度，与真实 MoE 的专家粒度一一对应。

2. **前瞻预测用 `boost = (skill - 0.5) * 2`**。skill=0.5 时 boost 恰为 0（纯随机基线，
   命中率 ≈ top_k/n_experts），skill=1.0 时 boost=1（恒定命中）。这是本系列反复验证的
   mock 套路，杜绝「skill=0.5 却已命中 68%」的起点虚高。

3. **训练信号单调趋 1**：`skill += lr * (1 - skill)`，绝不「对 +1 错 -1」，否则低能力时
   期望漂移为负、skill 被压到 0。

4. **冷启动不算调度失误**：第 0 层专家加载是不可避免的初始 I/O，命中率只统计之后各层
   由预取带来的命中，指标才干净。

5. **确定性优先**：张量用 Python 列表，随机用 `md5 -> random.Random`，同一 seed 结果
   完全可复现，基准数字可复核。

## 快速开始

```bash
pip install -e .

python -m lesforge.doctor                       # 自检：后端+前向+基准
python -m lesforge.cli generate "describe"      # 单次生成，打印 LES 统计
python -m lesforge.cli train --epochs 20        # 训练 skill，看命中率曲线
python -m lesforge.cli bench --config configs/mobile.yaml
python -m lesforge.cli serve --config configs/mobile.yaml   # OpenAI 兼容服务
```

```python
from lesforge.config import Config
from lesforge.backends import get_backend

cfg = Config()
b = get_backend("mock", cfg)
text, out = b.generate("describe a scene")
print(out["stats"])   # {'hit_rate': ..., 'stalls': ..., 'resident': ...}
```

## 基准（实测）

> 数字由 `python -m lesforge.cli bench` 现场测得，可复核，非手写估算。

| 指标 | 结果 | 说明 |
|---|---|---|
| 预取命中率（skill 0.5 → 1.0） | 0.20 → 1.00 | 从随机基线（top_k/n_experts=0.125）单调爬到 1.0 |
| 加速比（串行 vs LES 重叠） | 1.33× → 4.0× | I/O 被计算隐藏，上界 `1 + k·I/C` |
| 内存压缩比（激活 / 全量） | 8.6%（3B / 35B） | 全量 ≈ 65.2 GiB，激活 ≈ 5.6 GiB |
| 常驻专家占比 | 8.3%（8 / 96） | 每层只留 top-k 专家在 RAM |

## 项目结构

```
lesforge/
├── core/          # 注册表
├── utils/         # 确定性伪随机 / 向量 / yamlish
├── vision/        # ViT（patch 嵌入 + 注意力 + 编码器）
├── moe/           # 路由 / SwiGLU 专家 / MoE 层 / 负载均衡损失
├── scheduler/     # ★ LES：常驻表 + 卸载库 + 预取流水线 + 前瞻预测器 + 预算
├── model/         # 投影 + MoE 解码器 + VLM 引擎
├── backends.py    # 五后端
├── train/  bench/  serving/  cli.py
├── docs/          # 架构 / LES 深挖 / 配置 / 基准 / API / 设计决策
├── prompts/       # 8 个可复用提示词
├── recipes/       # 6 个调度配方（mobile_int4 / server_fp32 / ...）
├── examples/      # 9 个可运行示例
└── configs/       # tiny / mobile / server / default
```

## 更多文档

- [docs/LES.md](docs/LES.md) — 前瞻专家调度的完整技术拆解
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 架构与数据流
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — 四项基准的口径
- [docs/CONFIG.md](docs/CONFIG.md) — 参数表与调参技巧
- [docs/DESIGN.md](docs/DESIGN.md) — 设计决策与踩坑记录

## License

Apache-2.0。本仓库是教学/参考性质，未包含 MiniCPM-V 4.7 的真实权重。
