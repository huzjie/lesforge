# 前瞻专家调度（Lookahead Expert Scheduling, LES）

这是 lesforge 的**灵魂组件**，对应 MiniCPM-V 4.7 在 iPad Pro M4 上「平均提速约
5 倍、输出不变」的工程手段。

## 问题

35B 参数的 MoE VLM，完整权重约 65.6 GiB，手机内存只有 ~3GB。就算只激活 3B
（35B-A3B），专家权重也不可能全部常驻。于是：

- 大部分专家放在 **SSD/NAND**，按需读入；
- 但 SSD 读有毫秒级延迟，若每层都「现用现读」，推理会被 I/O 卡死。

## 关键观察（这是可复用的思路）

MoE 每层只激活 top-K 个专家。**下一层要激活哪些专家，在上一层计算时就已经能
预测出来**（路由对相邻层的输入是近似稳定的，或用轻量预测头）。既然如此，与其
等算到下一层才发现「专家不在内存里」，不如**一边算当前层、一边把下一层的专家
提前从 SSD 读进 RAM**——让「计算」和「I/O」重叠，SSD 延迟就被完全藏进计算时间。

```
时间轴（串行，无 LES）：
[读专家L] [算层L] [读专家L+1] [算层L+1] ...   ← I/O 全部暴露

时间轴（LES，前瞻预取）：
[算层L] ────────────────► [算层L+1] ────────►
        └─[预取专家L+1]─┘          └─[预取专家L+2]─┘   ← I/O 藏在计算里
```

## 四个组件

1. **ResidencyTable（常驻表）**：记录当前 RAM 里有哪些专家，按 LRU 维护访问序。
2. **OffloadStore（卸载库）**：模拟 SSD，`read` 返回固定延迟；是速度基准的来源。
3. **PrefetchPipeline（预取流水线）**：有界深度的「在途加载」队列，非阻塞地
   发起下一层专家读取。
4. **LookaheadPredictor（前瞻预测器）**：用当前隐藏态预测下一层 top-K 专家。

## 预测器的能力建模（可复用的 mock 套路）

预测器的准确率由一个能力标量 `skill ∈ [0.5, 1.0]` 控制，采用已被验证单调的
**`boost = (skill - 0.5) * 2`** 配方：

- `skill = 0.5` → `boost = 0` → 纯随机猜（命中率 ≈ top_k / n_experts）；
- `skill = 1.0` → `boost = 1` → 恒定命中真实路由（命中率 = 1.0）。

```python
base = boost * (1.0 if e in gold_set else 0.0) + 0.05 * affinity + 0.6 * noise
```

训练时 `skill` 单调爬向 1（`skill += lr * (1 - skill)`），预取命中率随之单调上升。

## 调度循环

```
warmup(层0专家)                      # 冷启动，不算调度失误
for l in 0..n_layers-1:
    _load_missing(l, 本层所需专家)    # 缺失 -> SSD stall（计数）
    predict 下一层专家，入预取流水线   # 与计算重叠
    drain 流水线 -> 常驻
    _evict_to_budget()               # LRU 驱逐，控制内存
```

## 可观测指标

- **prefetch hit rate**：已提前驻留的所需专家占比（核心指标）。
- **stalls**：因预测失误触发的 SSD 阻塞次数。
- **speedup**：串行 `compute + top_k*io` 与重叠 `compute + stalls*io` 的比值。
