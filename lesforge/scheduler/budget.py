"""Memory budget accounting (MiB)."""


class MemoryBudget:
    """Track resident-expert RAM usage against a budget."""

    def __init__(self, budget_mb=3000):
        self.budget_mb = budget_mb
        self.used_mb = 0.0

    def bytes_per_expert(self, dim, hidden, bits):
        # 3 matrices (dim*hidden each) -> bytes per weight
        if bits == 0:
            bpw = 4.0  # fp32
        elif bits == 8:
            bpw = 1.0
        else:
            bpw = bits / 8.0
        return dim * hidden * 3 * bpw / (1024.0 * 1024.0)  # MiB

    def can_fit(self, expert_mb):
        return self.used_mb + expert_mb <= self.budget_mb

    def add(self, expert_mb):
        self.used_mb += expert_mb

    def release(self, expert_mb):
        self.used_mb = max(0.0, self.used_mb - expert_mb)

    def usage_ratio(self):
        return self.used_mb / max(1.0, self.budget_mb)
