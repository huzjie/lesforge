import unittest

from lesforge.moe.expert import SwiGLUExpert
from lesforge.utils.hash import stable_vector


class TestExpert(unittest.TestCase):
    def test_forward_shape(self):
        e = SwiGLUExpert(dim=32, hidden=64, expert_id=0)
        out = e.forward(stable_vector("z", 32))
        self.assertEqual(len(out), 32)

    def test_n_params(self):
        e = SwiGLUExpert(dim=32, hidden=64, expert_id=0)
        self.assertEqual(e.n_params(), 32 * 64 * 3)


if __name__ == "__main__":
    unittest.main()
