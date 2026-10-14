import unittest

from lesforge.moe.router import TopKRouter
from lesforge.moe.expert import SwiGLUExpert
from lesforge.moe.layer import MoELayer
from lesforge.moe.load_balance import load_balance_loss
from lesforge.utils.hash import stable_vector


class TestMoE(unittest.TestCase):
    def test_router_topk(self):
        r = TopKRouter(dim=64, n_experts=8, top_k=2)
        idxs, probs = r.forward(stable_vector("x", 64))
        self.assertEqual(len(idxs), 2)
        self.assertEqual(len(set(idxs)), 2)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_expert_shape(self):
        e = SwiGLUExpert(dim=64, hidden=128, expert_id=0)
        out = e.forward(stable_vector("y", 64))
        self.assertEqual(len(out), 64)

    def test_layer_shape(self):
        layer = MoELayer(dim=64, hidden=128, n_experts=8, top_k=2)
        seq = [stable_vector("z%d" % i, 64) for i in range(3)]
        out = layer.forward(seq)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]), 64)

    def test_load_balance(self):
        loss = load_balance_loss([4, 4, 0, 0], 4, 8)
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
