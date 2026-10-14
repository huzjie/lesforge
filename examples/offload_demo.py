"""Demonstrate expert offload + resident-eviction accounting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.scheduler.les import LEScheduler
from lesforge.utils.hash import stable_vector

cfg = Config()
sched = LEScheduler(cfg, cfg.moe.dim, cfg.moe.hidden)

needed_by_layer = [[0, 1], [2, 3], [1, 5], [4, 7]]
sched.warmup(needed_by_layer[0])
for l in range(len(needed_by_layer)):
    ng = needed_by_layer[l + 1] if l + 1 < len(needed_by_layer) else None
    nh = stable_vector("h%d" % l, cfg.moe.dim) if ng else None
    r = sched.step(l, needed_by_layer[l], ng, nh, skill=0.5)
    print("layer=%d missing=%d resident=%d" % (l, r["missing"], sched.resident.total_resident()))

print("offload reads=%d  hit_rate=%.4f" % (sched.offload.reads, sched.hit_rate()))
