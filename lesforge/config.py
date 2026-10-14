"""Configuration dataclass + zero-dependency YAML loader.

When PyYAML is not installed (the managed Python has no third-party deps), we
fall back to :mod:`lesforge.utils.yamlish`, a minimal YAML-subset parser that
handles nested maps, block lists (``- item``) and inline comments.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .utils.yamlish import parse

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class VisionConfig:
    patch_size: int = 8
    img_size: int = 32
    dim: int = 32
    depth: int = 2
    heads: int = 4


@dataclass
class MoEConfig:
    n_experts: int = 16
    top_k: int = 2
    hidden: int = 64
    dim: int = 32
    n_layers: int = 6
    activation: str = "swiglu"
    # declared model scale (billions of params) this config is modelling;
    # the instantiated tensors are a small, fast proxy -- the *numbers* here are
    # what the memory benchmark reports (35B total / 3B active = MiniCPM-V 4.7)
    total_params_b: float = 35.0
    active_params_b: float = 3.0


@dataclass
class ScheduleConfig:
    # LES: how many layers ahead to prefetch (lookahead depth)
    lookahead: int = 1
    # resident (RAM) tier capacity in number of experts
    resident_experts: int = 8
    # SSD offload store path
    offload_path: str = ".lesforge_offload"
    # memory budget in MiB
    memory_budget_mb: int = 3000
    # prefetch pipeline depth (in-flight expert loads)
    prefetch_depth: int = 2
    # quantize experts before offloading (0=none, 8=int8, 4=int4)
    quant_bits: int = 4
    # eviction policy: lru | lfu
    eviction: str = "lru"


@dataclass
class ModelConfig:
    name: str = "mock"
    dim: int = 32
    skill: float = 0.5
    temperature: float = 0.7
    max_tokens: int = 16
    vocab_size: int = 256


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    seed: int = 42

    @classmethod
    def from_dict(cls, d):
        def as_int(v, default):
            return int(v) if v is not None else default

        def as_float(v, default):
            return float(v) if v is not None else default

        vision = d.get("vision", {}) or {}
        moe = d.get("moe", {}) or {}
        schedule = d.get("schedule", {}) or {}
        model = d.get("model", {}) or {}
        serving = d.get("serving", {}) or {}
        return cls(
            vision=VisionConfig(
                patch_size=as_int(vision.get("patch_size"), 8),
                img_size=as_int(vision.get("img_size"), 32),
                dim=as_int(vision.get("dim"), 32),
                depth=as_int(vision.get("depth"), 2),
                heads=as_int(vision.get("heads"), 4),
            ),
            moe=MoEConfig(
                n_experts=as_int(moe.get("n_experts"), 16),
                top_k=as_int(moe.get("top_k"), 2),
                hidden=as_int(moe.get("hidden"), 64),
                dim=as_int(moe.get("dim"), 32),
                n_layers=as_int(moe.get("n_layers"), 6),
                activation=moe.get("activation", "swiglu"),
                total_params_b=as_float(moe.get("total_params_b"), 35.0),
                active_params_b=as_float(moe.get("active_params_b"), 3.0),
            ),
            schedule=ScheduleConfig(
                lookahead=as_int(schedule.get("lookahead"), 1),
                resident_experts=as_int(schedule.get("resident_experts"), 8),
                offload_path=schedule.get("offload_path", ".lesforge_offload"),
                memory_budget_mb=as_int(schedule.get("memory_budget_mb"), 3000),
                prefetch_depth=as_int(schedule.get("prefetch_depth"), 2),
                quant_bits=as_int(schedule.get("quant_bits"), 4),
                eviction=schedule.get("eviction", "lru"),
            ),
            model=ModelConfig(
                name=model.get("name", "mock"),
                dim=as_int(model.get("dim"), 32),
                skill=as_float(model.get("skill"), 0.5),
                temperature=as_float(model.get("temperature"), 0.7),
                max_tokens=as_int(model.get("max_tokens"), 16),
                vocab_size=as_int(model.get("vocab_size"), 256),
            ),
            serving=ServingConfig(
                host=serving.get("host", "127.0.0.1"),
                port=as_int(serving.get("port"), 8000),
            ),
            seed=as_int(d.get("seed"), 42),
        )


def load_config(path=None):
    """Load a Config from a YAML file, or return defaults when ``path`` is None."""
    if path is None:
        return Config()
    p = Path(path)
    if not p.exists():
        return Config()
    text = p.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = parse(text)
    return Config.from_dict(data)
