"""Third-party integrations (LangChain, MCP) -- optional dependencies."""
from .langchain import LesForgeLLM
from .mcp import LesForgeMCPTool

__all__ = ["LesForgeLLM", "LesForgeMCPTool"]
