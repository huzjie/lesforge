import unittest

from lesforge.scheduler.policy import select_policy


class TestPolicy(unittest.TestCase):
    def test_lru(self):
        self.assertEqual(select_policy("lru"), "lru")

    def test_lfu_falls_back(self):
        self.assertEqual(select_policy("lfu"), "lru")


if __name__ == "__main__":
    unittest.main()
