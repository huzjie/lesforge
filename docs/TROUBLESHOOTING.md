# 故障排查

## `ModuleNotFoundError: No module named 'lesforge'`

在项目根目录执行 `pip install -e .`，或运行示例前先 `pip install -e .`。

## `KeyError: unknown backend`

后端名写错，或 `backends.py` 里的 `@register_backend` 未生效。可用
`python -c "from lesforge.backends import list_backends; print(list_backends())"` 查看。

## 命中率恒为 1.0

说明 `n_experts` 太小或 `top_k` 太大，随机基线本身已接近 1。调大 n_experts 再看。

## 服务启动后 curl 无响应

默认绑定 `127.0.0.1:8000`，确认端口未被占用；用 `/health` 先探活。
