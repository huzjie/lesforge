# 术语表

- **MoE**：Mixture-of-Experts，每 token 只激活 top-K 专家。
- **LES**：Lookahead Expert Scheduling，前瞻专家调度。
- **top_k**：每 token 激活的专家数。
- **resident**：常驻 RAM 的专家。
- **offload**：卸载到 SSD/NAND、按需读入。
- **prefetch**：预取（提前把下一层专家读进 RAM）。
- **stall**：因专家未常驻导致的 SSD 阻塞。
- **hit_rate**：已提前驻留的所需专家占比。
- **skill**：能力标量，控制前瞻预测准确率。
