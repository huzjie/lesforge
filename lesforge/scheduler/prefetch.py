"""In-flight lookahead prefetch pipeline.

The key insight of LES: while layer L computes, we *already* know (or can
predict) which experts layer L+1 will need, so we start loading them from SSD
now. By the time layer L finishes, layer L+1's experts are in RAM and no SSD
stall occurs. ``PrefetchPipeline`` models the queue of expert loads that are
currently in flight.
"""


class PrefetchPipeline:
    """A bounded queue of asynchronous expert loads (compute-overlapped)."""

    def __init__(self, depth=2, offload=None):
        from .offload import expert_key
        self.depth = depth
        self.offload = offload
        self._expert_key = expert_key
        self.in_flight = []  # list of expert keys being loaded

    def can_issue(self):
        return len(self.in_flight) < self.depth

    def issue(self, layer_id, expert_id):
        """Start loading an expert (non-blocking)."""
        self.in_flight.append((layer_id, expert_id))
        return self.offload.read(self._expert_key(layer_id, expert_id))[1]

    def drain(self, resident):
        """Block until all in-flight loads complete, admit them to RAM."""
        admitted = []
        for layer_id, expert_id in self.in_flight:
            resident.admit(layer_id, expert_id)
            admitted.append((layer_id, expert_id))
        self.in_flight = []
        return admitted
