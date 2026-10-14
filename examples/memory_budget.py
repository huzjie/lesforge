"""Show the declared memory footprint vs budget."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.bench import run

r = run("memory", Config())
print("full weights : %.2f GiB" % r["full_gib"])
print("active        : %.2f GiB" % r["active_gib"])
print("compression   : %.2f%%" % (r["compression"] * 100))
print("resident ratio: %.2f%%" % (r["resident_ratio"] * 100))
