"""Self-check: instantiate every backend, run a forward pass, list benchmarks.

Run with ``python -m lesforge.doctor``. This is the fastest way to confirm the
package is wired correctly (imports, registries, forward pass) after install.
"""
from .config import load_config
from .backends import list_backends, get_backend
from .bench import list_benchmarks, run_all, summarize
from .model.engine import VLMEngine
from .moe.router import TopKRouter
from .scheduler.les import LEScheduler, LookaheadPredictor


def _cfg():
    return load_config()


def main():
    cfg = _cfg()
    print("[doctor] config:", cfg.model.name, "dim=%d n_experts=%d top_k=%d" %
          (cfg.model.dim, cfg.moe.n_experts, cfg.moe.top_k))

    print("[doctor] backends:", ", ".join(list_backends()))
    for name in list_backends():
        b = get_backend(name, cfg)
        print("[doctor]   %-14s ok" % name)

    print("[doctor] components: VLMEngine / TopKRouter / LEScheduler / LookaheadPredictor ok")

    eng = VLMEngine(cfg)
    out = eng.generate(prompt="describe")
    print("[doctor] forward pass ok: n_layers=%d hit_rate=%.4f stalls=%d" %
          (out["n_layers"], out["stats"]["hit_rate"], out["stats"]["stalls"]))

    print("[doctor] benchmarks:", ", ".join(list_benchmarks()))
    results = run_all(cfg, skill=cfg.model.skill)
    print("[doctor]\n" + summarize(results))


if __name__ == "__main__":
    main()
