"""LES speedup benchmark: serial (compute+io) vs overlapped (max(compute,io))."""
from ..core.registry import BENCHMARKS
from ..model.engine import VLMEngine


@BENCHMARKS.register("speedup")
def run(cfg, skill=None, **kw):
    skill = cfg.model.skill if skill is None else skill
    engine = VLMEngine(cfg)
    # serial baseline: every layer pays compute + (top_k * io) because nothing
    # is ever prefetched.
    io_per_expert = engine.scheduler.offload.read_latency_ms
    compute_ms = 8.0
    n_layers = engine.decoder.n_layers
    top_k = cfg.moe.top_k

    serial = n_layers * (compute_ms + top_k * io_per_expert)

    # overlapped (LES): layer 0 cold start pays io, subsequent layers hide io
    # behind compute when the prefetch hits.
    out = engine.generate(prompt="describe", skill=skill)
    hit = out["stats"]["hit_rate"]
    stalls = out["stats"]["stalls"]
    overlapped = n_layers * compute_ms + stalls * io_per_expert

    speedup = serial / max(1e-9, overlapped)
    return {
        "serial_ms": serial,
        "overlapped_ms": overlapped,
        "speedup_x": speedup,
        "hit_rate": hit,
        "stalls": stalls,
    }
