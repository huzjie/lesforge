"""Run all benchmarks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.bench import run_all, summarize

cfg = Config()
results = run_all(cfg, skill=cfg.model.skill)
print(summarize(results))
