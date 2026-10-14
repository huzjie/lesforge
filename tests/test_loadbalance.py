import unittest

from lesforge.config import Config
from lesforge.model.engine import VLMEngine
from lesforge.moe.load_balance import routing_counts, load_balance_loss


class TestLoadBalance(unittest.TestCase):
    def test_counts_sum(self):
        counts = routing_counts([[0, 1], [1, 2], [0, 2]], 8)
        self.assertEqual(sum(counts), 6)

    def test_engine_balance(self):
        cfg = Config()
        eng = VLMEngine(cfg)
        routing = eng.route_all_layers(eng.decoder.embed(eng._token_ids("describe")))
        for l in range(eng.decoder.n_layers):
            counts = routing_counts(routing[l], cfg.moe.n_experts)
            loss = load_balance_loss(counts, cfg.moe.n_experts, max(1, sum(counts)))
            self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
