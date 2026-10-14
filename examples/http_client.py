"""Minimal client for the OpenAI-compatible server."""
import json
import urllib.request

body = json.dumps({"messages": [{"role": "user", "content": "describe"}]}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=10) as r:
    print(json.loads(r.read())["choices"][0]["message"]["content"])
