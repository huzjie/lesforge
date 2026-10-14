# utils — 确定性工具

- `hash.py`：`md5 -> random.Random` 的确定性伪随机。`stable_float/int/ints/bool/vector`
  是 mock 后端和基准的随机来源，保证同 seed 完全可复现。
- `vectors.py`：`dot` / `l2` / `normalize` / `cosine`，纯 Python。
- `text.py`：极简 tokenizer（零依赖，长度真实）。
- `yamlish.py`：极简 YAML 子集解析器，处理嵌套 map、块列表（`- item`）、行内注释。

**坑**：块列表 `key:\n  - a` 解析时要用栈在父容器里把占位 dict 换成 list，否则第二个
`- item` 会掉进孤儿 list。见 `yamlish.parse` 的实现注释。
