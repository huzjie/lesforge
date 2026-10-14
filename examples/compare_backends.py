"""Instantiate every backend and report availability."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.backends import list_backends, get_backend

cfg = Config()
for name in list_backends():
    b = get_backend(name, cfg)
    print("%-14s -> skill=%s" % (name, getattr(b, "skill", "n/a")))
