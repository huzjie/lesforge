"""Thin quantization facade (see moe.expert.quantize_expert).

Kept as a standalone module so the scheduler/CLI can refer to a stable
``quantize`` entry point without importing MoE internals everywhere.
"""
from ..moe.expert import quantize_expert


def quantize(expert, bits):
    return quantize_expert(expert, bits)


def storage_ratio(bits):
    """Storage multiplier vs fp32: 0=none, 8=int8, 4=int4."""
    if bits == 0:
        return 1.0
    if bits == 8:
        return 0.25
    return bits / 32.0
