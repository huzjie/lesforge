# 量化指南

`lesforge.scheduler.quant.quantize(expert, bits)` 对专家权重做确定性伪量化：

- `bits=0`：fp32，存储比 1.0；
- `bits=8`：int8，存储比 0.25；
- `bits=4`：int4，存储比 0.125。

真实 INT4/INT8 需要校准；这里模型化的是**量化带来的存储/传输成本下降**，用于计算
卸载与内存预算。手机端 `configs/mobile.yaml` 用 `quant_bits: 4`。
