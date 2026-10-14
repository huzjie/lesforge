"""Quantize experts and report storage savings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.moe.expert import SwiGLUExpert
from lesforge.scheduler.quant import quantize, storage_ratio

e = SwiGLUExpert(dim=128, hidden=512, expert_id=0)
print("expert params:", e.n_params())
for bits in (0, 8, 4):
    r = storage_ratio(bits)
    print("bits=%d  storage_ratio=%.3f  bytes_equiv=%.1f KiB" %
          (bits, r, e.n_params() * 4 * r / 1024))
