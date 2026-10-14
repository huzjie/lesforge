import unittest

from lesforge.scheduler.quant import storage_ratio
from lesforge.moe.expert import SwiGLUExpert, quantize_expert


class TestQuant(unittest.TestCase):
    def test_storage_ratio(self):
        self.assertEqual(storage_ratio(0), 1.0)
        self.assertEqual(storage_ratio(8), 0.25)
        self.assertEqual(storage_ratio(4), 0.125)

    def test_quantize_expert(self):
        e = SwiGLUExpert(dim=32, hidden=64, expert_id=0)
        q = quantize_expert(e, 4)
        self.assertEqual(len(q.w1), 64)
        self.assertEqual(len(q.w2), 32)


if __name__ == "__main__":
    unittest.main()
