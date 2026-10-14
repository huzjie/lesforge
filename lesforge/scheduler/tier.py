"""Residency table: which experts are in RAM (active tier) right now."""


class ResidencyTable:
    """Tracks per-layer resident expert sets + recency for LRU eviction."""

    def __init__(self):
        # layer_id -> set(expert_id)
        self.resident = {}
        # global access-order list for LRU (oldest first)
        self._lru = []
        self._clock = 0

    def is_resident(self, layer_id, expert_id):
        return expert_id in self.resident.get(layer_id, set())

    def touch(self, layer_id, expert_id):
        self._clock += 1
        key = (layer_id, expert_id)
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)
        return self._clock

    def admit(self, layer_id, expert_id):
        self.resident.setdefault(layer_id, set()).add(expert_id)
        self.touch(layer_id, expert_id)

    def evict_lru(self):
        """Evict the least-recently-used expert, return (layer_id, expert_id)."""
        if not self._lru:
            return None
        key = self._lru.pop(0)
        layer_id, expert_id = key
        self.resident[layer_id].discard(expert_id)
        if not self.resident[layer_id]:
            del self.resident[layer_id]
        return key

    def total_resident(self):
        return sum(len(s) for s in self.resident.values())
