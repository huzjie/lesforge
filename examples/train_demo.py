"""Train the skill and print the hit-rate curve."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.backends import get_backend
from lesforge.train import Trainer

cfg = Config()
b = get_backend("mock", cfg)
tr = Trainer(b, lr=0.05)
hist = tr.train(epochs=20)
print("%-6s %-8s %-10s" % ("epoch", "skill", "hit_rate"))
for e, s, h in hist:
    print("%-6d %-8.4f %-10.4f" % (e, s, h))
