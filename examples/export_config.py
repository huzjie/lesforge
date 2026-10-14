"""Dump an effective config to YAML (for reproducibility)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import asdict
from lesforge.config import Config

c = Config()
for k, v in asdict(c).items():
    if hasattr(v, "__dict__"):
        print("%s:" % k)
        for kk, vv in v.__dict__.items():
            print("  %s: %s" % (kk, vv))
    else:
        print("%s: %s" % (k, v))
