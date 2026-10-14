import unittest

from lesforge.config import Config
from lesforge.model.engine import VLMEngine


class TestEngine(unittest.TestCase):
    def test_generate(self):
        cfg = Config()
        eng = VLMEngine(cfg)
        out = eng.generate(prompt="describe")
        self.assertIn("text", out)
        self.assertIn("stats", out)
        self.assertGreater(out["n_layers"], 0)
        self.assertIn("hit_rate", out["stats"])

    def test_n_layers(self):
        cfg = Config()
        eng = VLMEngine(cfg)
        self.assertEqual(eng.decoder.n_layers, cfg.moe.n_layers)


if __name__ == "__main__":
    unittest.main()
