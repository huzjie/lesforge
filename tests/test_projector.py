import unittest

from lesforge.model.projector import Projector
from lesforge.utils.hash import stable_vector


class TestProjector(unittest.TestCase):
    def test_shape(self):
        p = Projector(32, 64)
        out = p.forward(stable_vector("v", 32))
        self.assertEqual(len(out), 64)


if __name__ == "__main__":
    unittest.main()
