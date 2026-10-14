# 规格：HTTP API

## GET /health
返回 `{"status": "ok"}`。

## GET /v1/models
返回 `{"object":"list","data":[...]}`。

## POST /v1/chat/completions
请求 `{"messages":[{"role":"user","content":"..."}]}`；
返回 `{"choices":[{"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}]}`。
