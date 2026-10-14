"""Prefetch hit-rate benchmark (the headline LES metric)."""
from ..core.registry import BENCHMARKS
from ..model.engine import VLMEngine


@BENCHMARKS.register("prefetch")
def run(cfg, skill=None, **kw):
    skill = cfg.model.skill if skill is None else skill
    engine = VLMEngine(cfg)
    out = engine.generate(prompt="describe", skill=skill)
    stats = out["stats"]
    baseline = cfg.moe.top_k / cfg.moe.n_experts
    return {
        "hit_rate": stats["hit_rate"],
        "random_baseline": baseline,
        "gain": stats["hit_rate"] - baseline,
        "stalls": stats["stalls"],
        "resident": stats["resident"],
    }
