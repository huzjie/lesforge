# API 参考

## 顶层

- `lesforge.Config` — 配置数据类（vision / moe / schedule / model / serving）。
- `lesforge.load_config(path=None)` — 从 YAML 加载（无 pyyaml 时走内置 yamlish）。
- `lesforge.get_backend(name, cfg)` — 取后端实例。
- `lesforge.list_backends()` — 列出注册的后端。
- `lesforge.LEScheduler(cfg, dim, hidden)` — 前瞻专家调度器。
- `lesforge.VLMEngine(cfg)` — 端到端推理引擎。

## VLMEngine

```python
eng = VLMEngine(cfg)
out = eng.generate(image=None, prompt="describe", skill=None, max_tokens=None)
# out = {"text": str, "stats": dict, "n_layers": int}
```

`stats` 字段：`hit_rate`（预取命中率）、`stalls`（SSD 阻塞次数）、`compute_ms`、
`io_ms`、`resident`（常驻专家数）、`budget_ratio`。

## LEScheduler

```python
sched = LEScheduler(cfg, dim=128, hidden=512)
sched.warmup(needed_experts)
sched.step(layer_id, needed_experts, next_gold, next_hidden, skill, compute_ms=8.0)
sched.hit_rate()   # -> float
sched.stats()      # -> dict
```

## MoE

- `TopKRouter(dim, n_experts, top_k, layer_id)` — `forward(hidden) -> (indices, probs)`。
- `SwiGLUExpert(dim, hidden, expert_id)` — `forward(x) -> out`。
- `MoELayer(dim, hidden, n_experts, top_k, layer_id)` — `forward(seq) -> seq`。
- `load_balance_loss(counts, n_experts, n_tokens)` — Switch 式辅助损失。

## 量化

- `lesforge.scheduler.quant.quantize(expert, bits)` — 伪量化到 `bits` 位。
- `lesforge.scheduler.quant.storage_ratio(bits)` — 相对 fp32 的存储倍数。

## 集成

- `lesforge.integrations.LesForgeLLM(backend)` — LangChain 包装。
- `lesforge.integrations.LesForgeMCPTool(backend)` — MCP 工具包装。
