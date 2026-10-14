"""Model Context Protocol (MCP) tool wrapper (optional).

Exposes the lesforge generator as an MCP tool with a JSON-RPC ``tools/call``
shape so an agent can invoke on-device MoE VLM inference. No hard MCP dependency:
if the ``mcp`` package is absent, the class still works standalone.
"""


class LesForgeMCPTool:
    """A callable MCP-style tool that wraps a lesforge backend."""

    def __init__(self, backend, name="lesforge_generate"):
        self.backend = backend
        self.name = name

    @property
    def tool_schema(self):
        return {
            "name": self.name,
            "description": "Generate a response from an on-device MoE vision-language model",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                },
                "required": ["prompt"],
            },
        }

    def call(self, arguments):
        prompt = arguments.get("prompt", "")
        text, out = self.backend.generate(prompt)
        return {"content": [{"type": "text", "text": text}],
                "stats": out.get("stats", {})}
