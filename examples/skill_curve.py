"""Show prefetch hit rate as a function of skill."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.model.engine import VLMEngine

cfg = Config()
for skill in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    eng = VLMEngine(cfg)
    out = eng.generate(prompt="describe", skill=skill)
    print("skill=%.1f  hit_rate=%.4f  stalls=%d" %
          (skill, out["stats"]["hit_rate"], out["stats"]["stalls"]))
