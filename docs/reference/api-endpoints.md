# HTTP 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 探活，返回 `{"status":"ok"}` |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | OpenAI 兼容对话补全 |

请求体示例见 [GETTING_STARTED](../GETTING_STARTED.md)。
