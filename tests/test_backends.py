import unittest

from lesforge.config import Config
from lesforge.backends import get_backend, list_backends


class TestBackends(unittest.TestCase):
    def test_all_backends_instantiable(self):
        cfg = Config()
        names = list_backends()
        self.assertIn("mock", names)
        for name in names:
            b = get_backend(name, cfg)
            self.assertIsNotNone(b)

    def test_mock_generates(self):
        cfg = Config()
        b = get_backend("mock", cfg)
        text, out = b.generate("describe")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
