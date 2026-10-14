"""A dependency-free, OpenAI-compatible HTTP server.

Exposes ``/health``, ``/v1/models`` and ``/v1/chat/completions`` using only the
standard library (``http.server``). This mirrors the shape a production
OpenAI-compatible gateway would expose, so the mock backend can be dropped into
any existing client with zero changes.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..config import Config, load_config
from ..backends import get_backend


class _Handler(BaseHTTPRequestHandler):
    backend = None
    cfg = None

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "ok"})
        if self.path == "/v1/models":
            return self._json({"object": "list", "data": [
                {"id": "lesforge-mock", "object": "model", "owned_by": "lesforge"}]})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            messages = body.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""
            text, _ = self.backend.generate(prompt)
            return self._json({
                "id": "chatcmpl-lesforge",
                "object": "chat.completion",
                "model": "lesforge-mock",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
            })
        return self._json({"error": "not found"}, 404)


def create_server(cfg):
    class Handler(_Handler):
        pass

    Handler.cfg = cfg
    Handler.backend = get_backend(cfg.model.name, cfg)
    return Handler


def run_server(cfg):
    handler = create_server(cfg)
    host, port = cfg.serving.host, cfg.serving.port
    httpd = HTTPServer((host, port), handler)
    print("[lesforge] serving on http://%s:%s" % (host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
