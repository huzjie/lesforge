# 贡献指南

## 环境

零第三方依赖，Python ≥ 3.9 即可。建议用 `pip install -e .` 开发。

## 提交流程

1. fork 并 clone；
2. 新建分支 `feat/xxx`；
3. 改动后运行：

```bash
python -m compileall -q lesforge
python -m unittest discover -s tests -t .
python -m lesforge.doctor
```

4. 提交 PR。

## 代码约定

- 保持零依赖：不要引入 numpy/pytorch/requests 等第三方库。
- 确定性优先：随机一律走 `lesforge.utils.hash`。
- 新后端用 `@register_backend`，新基准用 `@register("name")`，并在对应 `__init__.py`
  显式 `from . import ...` 触发注册。
