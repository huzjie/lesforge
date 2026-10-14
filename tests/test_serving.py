import json
import threading
import unittest
import urllib.request
from http.server import HTTPServer

from lesforge.config import Config
from lesforge.serving.server import create_server


class TestServing(unittest.TestCase):
    def test_health_and_chat(self):
        cfg = Config()
        handler = create_server(cfg)
        httpd = HTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=10) as r:
                self.assertEqual(json.loads(r.read())["status"], "ok")
            body = json.dumps({"messages": [{"role": "user", "content": "describe"}]}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:%d/v1/chat/completions" % port, data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
                self.assertIn("choices", data)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
