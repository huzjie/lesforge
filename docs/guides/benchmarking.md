# 基准方法学

四个基准都**确定性、可复现**：

- 随机一律走 `utils.hash`（md5 种子）；
- 加速比用「串行 = compute + k*I/O，重叠 = compute + stalls*I/O」闭式对比；
- 命中率排除冷启动层，只统计前瞻预取生效的层；
- 内存用声明规模（35B/3B），不随实例化 tensor 缩小而失真。

跑：`python -m lesforge.cli bench --config configs/mobile.yaml`。
