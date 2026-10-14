import unittest

from lesforge.vision.attention import MultiHeadAttention
from lesforge.utils.hash import stable_vector


class TestAttention(unittest.TestCase):
    def test_forward_shape(self):
        mha = MultiHeadAttention(dim=32, heads=4)
        seq = [stable_vector("t%d" % i, 32) for i in range(4)]
        out = mha.forward(seq)
        self.assertEqual(len(out), 4)
        self.assertEqual(len(out[0]), 32)


if __name__ == "__main__":
    unittest.main()
