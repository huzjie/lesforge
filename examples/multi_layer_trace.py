"""Trace per-layer expert selection and prefetch behaviour."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.model.engine import VLMEngine

cfg = Config()
eng = VLMEngine(cfg)
out = eng.generate(prompt="describe", skill=1.0)
print("n_layers=%d  hit_rate=%.4f  stalls=%d" %
      (out["n_layers"], out["stats"]["hit_rate"], out["stats"]["stalls"]))
