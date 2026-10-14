import unittest

from lesforge.utils.hash import stable_float, stable_ints, stable_vector, stable_bool
from lesforge.utils.vectors import dot, cosine, normalize, l2


class TestUtils(unittest.TestCase):
    def test_stable_float_reproducible(self):
        self.assertEqual(stable_float("a"), stable_float("a"))

    def test_stable_ints_len(self):
        self.assertEqual(len(stable_ints("k", 8)), 8)

    def test_stable_vector_len(self):
        self.assertEqual(len(stable_vector("v", 32)), 32)

    def test_stable_bool(self):
        self.assertIn(stable_bool("b"), (True, False))

    def test_vector_ops(self):
        a, b = [1.0, 0.0], [0.0, 1.0]
        self.assertAlmostEqual(dot(a, b), 0.0)
        self.assertAlmostEqual(cosine(a, b), 0.0)
        self.assertAlmostEqual(l2(normalize([3.0, 4.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
