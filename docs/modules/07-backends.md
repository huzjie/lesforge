# backends — 五后端

统一接口 `generate(prompt, image=None) -> (text, out)`：

| 后端 | 说明 |
|---|---|
| mock | 确定性、可训练、零依赖（默认，基准/训练用） |
| cpu | 同一套 MoE 数学的 CPU 参考循环 |
| openai | 委托 OpenAI 兼容 `/chat/completions` |
| vllm | 委托 vLLM OpenAI 兼容服务 |
| transformers | 委托本地 HF `text-generation` pipeline |

`get_backend(name, cfg)` 取实例；mock/cpu 暴露 `skill` 能力标量。
