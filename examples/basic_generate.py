"""Minimal generation with LES stats."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.backends import get_backend

cfg = Config()
b = get_backend("mock", cfg)
text, out = b.generate("describe a scene")
print("answer:", text)
print("stats :", out["stats"])
