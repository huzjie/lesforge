import unittest

from lesforge.model.decoder import MoEDecoder


class TestDecoder(unittest.TestCase):
    def test_embed(self):
        d = MoEDecoder(dim=32, hidden=64, n_experts=8, top_k=2, n_layers=3, vocab_size=64)
        emb = d.embed([1, 2, 3])
        self.assertEqual(len(emb), 3)
        self.assertEqual(len(emb[0]), 32)

    def test_layer_routing(self):
        d = MoEDecoder(dim=32, hidden=64, n_experts=8, top_k=2, n_layers=3, vocab_size=64)
        seq = d.embed([1, 2])
        sel = d.layer_routing(0, seq)
        self.assertEqual(len(sel), 2)
        self.assertEqual(len(sel[0]), 2)


if __name__ == "__main__":
    unittest.main()
