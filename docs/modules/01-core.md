# core — 注册表

`core/registry.py` 提供两个注册表实例：

- `BACKENDS` — 后端注册表，`@register_backend("mock")` 注册类。
- `BENCHMARKS` — 基准注册表，`@BENCHMARKS.register("memory")` 注册函数。

关键点：注册装饰器只在**被 import 时**执行。因此 `backends.py` 必须显式
`from . import mock, cpu, ...`（或集中在一个文件里），`bench/__init__.py` 必须
`from . import memory, speedup, ...`，否则 `list_backends()` / `list_benchmarks()`
会是空的。
