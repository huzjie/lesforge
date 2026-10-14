"""Eviction policy selectors (LRU is default; LFU falls back to LRU).

Kept as a thin selector so ``cfg.schedule.eviction`` can be honoured without
scattering if/else across the scheduler.
"""


def select_policy(name):
    if name == "lfu":
        # LFU is reserved; the residency table only maintains recency, so we
        # fall back to LRU and document the limitation.
        return "lru"
    return "lru"
