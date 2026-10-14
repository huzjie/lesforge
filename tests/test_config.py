import unittest

from lesforge.config import Config, load_config


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Config()
        self.assertEqual(c.model.skill, 0.5)
        self.assertEqual(c.moe.top_k, 2)

    def test_from_dict(self):
        c = Config.from_dict({
            "moe": {"n_experts": 16, "top_k": 4},
            "model": {"name": "mock", "skill": 0.7},
        })
        self.assertEqual(c.moe.n_experts, 16)
        self.assertEqual(c.moe.top_k, 4)
        self.assertEqual(c.model.skill, 0.7)

    def test_load_missing(self):
        c = load_config("does_not_exist.yaml")
        self.assertIsInstance(c, Config)


if __name__ == "__main__":
    unittest.main()
