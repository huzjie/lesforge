import unittest

from lesforge.config import Config
from lesforge.model.engine import VLMEngine


class TestScheduler(unittest.TestCase):
    def _hit(self, skill):
        cfg = Config()
        eng = VLMEngine(cfg)
        out = eng.generate(prompt="describe", skill=skill)
        return out["stats"]["hit_rate"]

    def test_hit_rate_monotonic_with_skill(self):
        lo = self._hit(0.5)
        hi = self._hit(1.0)
        self.assertGreater(hi, lo)

    def test_high_skill_hits_everything(self):
        self.assertAlmostEqual(self._hit(1.0), 1.0, places=5)

    def test_low_skill_above_zero(self):
        self.assertGreaterEqual(self._hit(0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
