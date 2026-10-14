"""Expose the backend as an MCP-style tool."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.backends import get_backend
from lesforge.integrations.mcp import LesForgeMCPTool

cfg = Config()
b = get_backend("mock", cfg)
tool = LesForgeMCPTool(b)
print("schema:", tool.tool_schema["name"])
print("call  :", tool.call({"prompt": "describe"}))
