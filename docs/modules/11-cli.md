# cli — 命令行

`python -m lesforge.cli <command>`：

- `doctor` — 自检（后端 + 前向 + 基准）。
- `generate [prompt]` — 单次生成并打印 LES 统计。
- `train --epochs N` — 训练 skill，打印命中率曲线。
- `bench --config PATH` — 跑全部基准。
- `serve --config PATH` — 启动 HTTP 服务。

所有子命令共享 `--config`（默认读 `configs/default.yaml`，未传则用 dataclass 默认）。
