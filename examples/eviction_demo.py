"""Demonstrate LRU eviction when resident budget is tight."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.scheduler.tier import ResidencyTable

r = ResidencyTable()
for e in range(5):
    r.admit(0, e)          # admit 5 experts
print("resident before evict:", r.total_resident())
r.evict_lru()              # evict oldest
print("resident after evict :", r.total_resident())
