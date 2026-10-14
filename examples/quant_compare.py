"""Compare storage across quantization bit-widths."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.scheduler.quant import storage_ratio

for bits in (0, 8, 4):
    print("bits=%d -> storage %.3fx of fp32" % (bits, storage_ratio(bits)))
