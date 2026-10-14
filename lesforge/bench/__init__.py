"""Benchmarks: memory, speedup, prefetch hit rate, load balance."""
from ..core.registry import BENCHMARKS


def register(key):
    return BENCHMARKS.register(key)


# import submodules so their @register decorations run
from . import memory, speedup, prefetch, loadbalance


def list_benchmarks():
    return BENCHMARKS.keys()


def run(name, cfg, **kw):
    fn = BENCHMARKS.get(name)
    if fn is None:
        raise KeyError("unknown benchmark %r (available: %s)" % (name, BENCHMARKS.keys()))
    return fn(cfg, **kw)


def run_all(cfg, **kw):
    """Run every registered benchmark; return {name: result}."""
    out = {}
    for name in BENCHMARKS.keys():
        if name == "speedup":
            out[name] = run(name, cfg, skill=kw.get("skill"))
        elif name == "prefetch":
            out[name] = run(name, cfg, skill=kw.get("skill"))
        else:
            out[name] = run(name, cfg)
    return out


def summarize(results):
    lines = []
    for k, v in results.items():
        if isinstance(v, dict):
            lines.append("%-12s %s" % (k, " ".join("%s=%.4f" % (a, b) for a, b in v.items())))
        else:
            lines.append("%-12s %s" % (k, v))
    return "\n".join(lines)
