# 基准（Benchmarks）

四个基准都注册在 `lesforge.bench`，`run_all(cfg)` 一次跑完。

## 1. memory（内存足迹）

单个专家权重按 `dim * hidden * 3 * (bits/8)` 字节估算（3 个矩阵），对比常驻专家
与全部专家的内存占比。演示 35B 模型只留一小撮专家在 RAM 的压缩比。

## 2. prefetch（预取命中率）—— 头号指标

```python
from lesforge.bench import run
run("prefetch", cfg, skill=0.5)   # hit_rate ≈ top_k / n_experts
run("prefetch", cfg, skill=1.0)   # hit_rate = 1.0
```

命中率从随机基线（top_k/n_experts）随 skill 单调爬向 1.0，直接量化 LES 的价值。

## 3. speedup（加速比）

对比「串行（每层 compute + top_k*io）」与「LES 重叠（compute + stalls*io）」的
墙钟。预测命中越多、stall 越少，加速比越高，逼近 `1 + top_k*io/compute`。

## 4. loadbalance（负载均衡）

统计各层专家被路由次数的 Switch 辅助损失，验证路由没有坍缩到少数专家。

## 运行

```bash
python -m lesforge.cli bench --config configs/mobile.yaml
```
