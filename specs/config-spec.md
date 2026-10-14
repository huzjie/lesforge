# 规格：配置

- 零依赖 YAML 读取（无 pyyaml 时用内置 yamlish）。
- `Config.from_dict` 容错：缺字段回退 dataclass 默认值。
- 四套预设：default / tiny / mobile / server。
