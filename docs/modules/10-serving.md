# serving — OpenAI 兼容 HTTP 服务

`server.py` 用纯标准库 `http.server` 提供 `/health`、`/v1/models`、
`/v1/chat/completions`。默认绑定 `127.0.0.1:8000`。

`python -m lesforge.cli serve --config configs/mobile.yaml` 启动。
