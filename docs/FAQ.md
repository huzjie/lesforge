# FAQ

**Q: 为什么零依赖？**
A: 目标是「读代码学习 + 秒级跑通」，不因装 numpy/pytorch 卡住。张量用 Python 列表。

**Q: skill 到底是什么？**
A: 一个能力标量，控制前瞻预测器能否猜中下一层要激活的专家。0.5=随机，1.0=全中。

**Q: 命中率为什么从约 top_k/n_experts 起步？**
A: skill=0.5 时 boost=0，预测器纯随机猜 top_k 个专家，命中下一层真实专家的概率就是
top_k/n_experts，这是正确的随机基线。

**Q: 能接真实模型权重吗？**
A: 调度逻辑（分层常驻 + 前瞻预取 + 驱逐）与具体权重无关。把 mock 专家换成真实权重
矩阵即可复用同一套 `LEScheduler`。

**Q: prefetch_depth 怎么调？**
A: 至少 ≥ top_k，否则预测全对也装不下预取；再大则吃内存，需与 resident_experts 平衡。
