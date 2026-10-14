"""Direct LEScheduler usage (without the full engine)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.scheduler.les import LEScheduler
from lesforge.utils.hash import stable_vector

cfg = Config()
sched = LEScheduler(cfg, cfg.moe.dim, cfg.moe.hidden)

# 4 层，每层激活 top_k=2 个专家（用确定性向量模拟路由）
needed_by_layer = [[0, 1], [2, 3], [1, 5], [4, 7]]

sched.warmup(needed_by_layer[0])
for l in range(len(needed_by_layer)):
    next_gold = needed_by_layer[l + 1] if l + 1 < len(needed_by_layer) else None
    next_hidden = stable_vector("h%d" % l, cfg.moe.dim) if next_gold else None
    sched.step(l, needed_by_layer[l], next_gold, next_hidden, skill=1.0)

print("hit_rate=%.4f  stalls=%d  resident=%d" %
      (sched.hit_rate(), sched.stalls, sched.resident.total_resident()))
