"""Simulated SSD/NAND offload store with deterministic read latency.

In MiniCPM-V 4.7 the full 35B weights (~65.6 GiB) never all fit in RAM; most
experts sit on NAND and are paged in on demand. We model that as a key -> blob
store with a fixed per-expert read latency, so the LES speedup numbers come from
the overlap of compute and I/O, not from an arbitrary constant.
"""


class OffloadStore:
    """A fake SSD: stores expert blobs and returns them with read latency."""

    def __init__(self, read_latency_ms=12.0, write_latency_ms=20.0):
        self.read_latency_ms = read_latency_ms
        self.write_latency_ms = write_latency_ms
        self._blobs = {}
        self.reads = 0
        self.writes = 0

    def write(self, key, blob):
        self._blobs[key] = blob
        self.writes += 1
        return self.write_latency_ms

    def read(self, key):
        self.reads += 1
        return self._blobs.get(key), self.read_latency_ms

    def has(self, key):
        return key in self._blobs


def expert_key(layer_id, expert_id):
    return "layer%d:expert%d" % (layer_id, expert_id)
