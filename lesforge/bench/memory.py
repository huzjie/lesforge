"""Memory footprint benchmark (declared model scale, not instantiated tensors).

The instantiated tensors are a small fast proxy; the *numbers that matter* are the
declared model scale (35B total / 3B active, matching MiniCPM-V 4.7). We report
the full-weight footprint (~2 bytes/param), the active footprint, and the fraction
of experts that stay resident under the memory budget.
"""
from ..core.registry import BENCHMARKS


@BENCHMARKS.register("memory")
def run(cfg, **kw):
    m = cfg.moe
    s = cfg.schedule
    bytes_per_param = 2.0  # bf16
    full_gib = m.total_params_b * 1e9 * bytes_per_param / (1024.0 ** 3)
    active_gib = m.active_params_b * 1e9 * bytes_per_param / (1024.0 ** 3)
    total_experts = m.n_experts * m.n_layers
    resident_ratio = s.resident_experts / max(1, total_experts)
    return {
        "params_b": m.total_params_b,
        "active_b": m.active_params_b,
        "full_gib": full_gib,
        "active_gib": active_gib,
        "compression": m.active_params_b / m.total_params_b,
        "resident_ratio": resident_ratio,
        "budget_mb": s.memory_budget_mb,
    }
