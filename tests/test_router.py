import unittest

from lesforge.moe.router import TopKRouter
from lesforge.utils.hash import stable_vector


class TestRouter(unittest.TestCase):
    def test_logits_len(self):
        r = TopKRouter(dim=32, n_experts=16, top_k=2)
        self.assertEqual(len(r.logits(stable_vector("x", 32))), 16)

    def test_topk_unique(self):
        r = TopKRouter(dim=32, n_experts=16, top_k=3)
        idxs, _ = r.forward(stable_vector("y", 32))
        self.assertEqual(len(set(idxs)), 3)


if __name__ == "__main__":
    unittest.main()
