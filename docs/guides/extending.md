# 扩展指南（加一个后端 / 基准）

**加后端**：
1. 在 `backends.py` 写 `@register_backend("myback")` 类，实现 `generate(prompt, image=None)`；
2. 确认 `list_backends()` 能看到它。

**加基准**：
1. 在 `bench/` 新建 `mybench.py`，写 `@BENCHMARKS.register("mybench")` 的函数；
2. 在 `bench/__init__.py` 加 `from . import mybench` 触发注册；
3. `run_all` 会自动带上它（若需特殊参数，在 `run_all` 里分发）。
