"""Show speedup upper bound vs measured."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.bench import run

cfg = Config()
for skill in (0.5, 1.0):
    r = run("speedup", cfg, skill=skill)
    print("skill=%.1f speedup=%.2fx (serial=%.0fms overlapped=%.0fms stalls=%d)" %
          (skill, r["speedup_x"], r["serial_ms"], r["overlapped_ms"], r["stalls"]))
