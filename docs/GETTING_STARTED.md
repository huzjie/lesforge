# 快速开始

## 环境要求

- Python ≥ 3.9，**零第三方依赖**。

## 安装与冒烟

```bash
pip install -e .

# 自检（后端 + 前向 + 基准）
python -m lesforge.doctor

# 单次生成（打印 LES 统计）
python -m lesforge.cli generate "describe the scene"

# 训练能力信号（打印 skill / hit_rate 曲线）
python -m lesforge.cli train --epochs 20

# 全量基准
python -m lesforge.cli bench --config configs/mobile.yaml

# 启动 OpenAI 兼容服务
python -m lesforge.cli serve --config configs/mobile.yaml
```

## HTTP 调用

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"describe"}]}'
```

## Docker

```bash
docker compose up --build
```

## 最小可运行示例

```python
from lesforge.config import Config
from lesforge.backends import get_backend

cfg = Config()
b = get_backend("mock", cfg)
text, out = b.generate("describe a scene")
print(text)
print(out["stats"])   # {'hit_rate': ..., 'stalls': ..., ...}
```
