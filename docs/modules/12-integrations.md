# integrations — 第三方集成

- `langchain.py`：`LesForgeLLM`，把后端包装成 LangChain LLM（`_generate` / `__call__`）。
- `mcp.py`：`LesForgeMCPTool`，把后端包装成 MCP 工具（`tool_schema` / `call`）。

两者都是可选依赖：未装 langchain/mcp 也能独立使用。
