import unittest

from lesforge.config import Config
from lesforge.backends import get_backend
from lesforge.train import Trainer


class TestTrain(unittest.TestCase):
    def test_skill_monotonic(self):
        cfg = Config()
        b = get_backend("mock", cfg)
        start = b.skill
        tr = Trainer(b, lr=0.05)
        hist = tr.train(epochs=5)
        self.assertGreater(b.skill, start)
        # hit rate at the end exceeds hit rate at the start
        self.assertGreater(hist[-1][2], hist[0][2])


if __name__ == "__main__":
    unittest.main()
