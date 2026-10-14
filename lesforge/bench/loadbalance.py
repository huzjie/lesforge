"""Load-balance benchmark: expert usage entropy across layers."""
import math

from ..core.registry import BENCHMARKS
from ..model.engine import VLMEngine
from ..moe.load_balance import routing_counts, load_balance_loss


@BENCHMARKS.register("loadbalance")
def run(cfg, **kw):
    engine = VLMEngine(cfg)
    out = engine.generate(prompt="describe")
    n_layers = out["n_layers"]
    # recompute routing counts on a fresh engine to report per-layer balance
    image = None
    routing = engine.route_all_layers(engine.decoder.embed(engine._token_ids("describe")))
    losses = []
    for l in range(n_layers):
        counts = routing_counts(routing[l], cfg.moe.n_experts)
        n_tok = max(1, sum(counts))
        losses.append(load_balance_loss(counts, cfg.moe.n_experts, n_tok))
    return {
        "avg_aux_loss": sum(losses) / max(1, len(losses)),
        "best_layer_loss": min(losses) if losses else 0.0,
        "worst_layer_loss": max(losses) if losses else 0.0,
    }
