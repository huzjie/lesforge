"""Ablation: hit rate vs lookahead depth."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.model.engine import VLMEngine

for lookahead in (1, 2):
    cfg = Config()
    cfg.schedule.lookahead = lookahead
    for skill in (0.5, 1.0):
        eng = VLMEngine(cfg)
        o = eng.generate(prompt="describe", skill=skill)
        print("lookahead=%d skill=%.1f hit_rate=%.4f stalls=%d" %
              (lookahead, skill, o["stats"]["hit_rate"], o["stats"]["stalls"]))
