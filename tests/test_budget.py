import unittest

from lesforge.scheduler.budget import MemoryBudget


class TestBudget(unittest.TestCase):
    def test_bytes_per_expert(self):
        b = MemoryBudget(3000)
        mb = b.bytes_per_expert(32, 64, 4)
        self.assertGreater(mb, 0.0)

    def test_add_release(self):
        b = MemoryBudget(100)
        b.add(40)
        self.assertEqual(b.used_mb, 40.0)
        b.release(10)
        self.assertEqual(b.used_mb, 30.0)

    def test_usage_ratio(self):
        b = MemoryBudget(100)
        b.add(50)
        self.assertAlmostEqual(b.usage_ratio(), 0.5)


if __name__ == "__main__":
    unittest.main()
