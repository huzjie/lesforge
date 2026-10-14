import unittest

from lesforge.config import Config
from lesforge.backends import get_backend
from lesforge.integrations.mcp import LesForgeMCPTool
from lesforge.integrations.langchain import LesForgeLLM


class TestIntegrations(unittest.TestCase):
    def test_mcp_tool(self):
        cfg = Config()
        tool = LesForgeMCPTool(get_backend("mock", cfg))
        self.assertEqual(tool.tool_schema["name"], "lesforge_generate")
        out = tool.call({"prompt": "describe"})
        self.assertIn("content", out)

    def test_langchain_wrapper(self):
        cfg = Config()
        llm = LesForgeLLM(get_backend("mock", cfg))
        self.assertEqual(llm._llm_type, "lesforge")
        text = llm("describe")
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
