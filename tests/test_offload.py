import unittest

from lesforge.scheduler.offload import OffloadStore, expert_key


class TestOffload(unittest.TestCase):
    def test_write_read(self):
        s = OffloadStore()
        s.write(expert_key(0, 1), b"blob")
        blob, lat = s.read(expert_key(0, 1))
        self.assertEqual(blob, b"blob")
        self.assertGreater(lat, 0.0)

    def test_key(self):
        self.assertEqual(expert_key(2, 5), "layer2:expert5")


if __name__ == "__main__":
    unittest.main()
