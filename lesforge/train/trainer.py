"""Capability (skill) training loop for the mock backend.

The mock backend exposes a scalar ``skill`` in [0.5, 1.0] that controls how well
the LookaheadPredictor anticipates the next layer's experts. Training follows
the project convention: the signal is **monotonic toward 1** (never reward-punish
oscillation), so the prefetch hit rate climbs cleanly from the random baseline
(~ top_k / n_experts) to 1.0.
"""


class Trainer:
    """Raises ``skill`` monotonically and reports the resulting hit rate."""

    def __init__(self, backend, lr=0.05):
        self.backend = backend
        self.lr = lr
        self.history = []

    def _evaluate(self):
        _, out = self.backend.generate("describe a scene")
        return out.get("stats", {}).get("hit_rate", 0.0)

    def train(self, epochs=20):
        """Run ``epochs`` training steps, logging (epoch, skill, hit_rate)."""
        self.history = []
        for e in range(epochs):
            self.backend.skill = min(1.0, self.backend.skill + self.lr * (1.0 - self.backend.skill))
            hit = self._evaluate()
            self.history.append((e + 1, round(self.backend.skill, 4), round(hit, 4)))
        return self.history

    def train_step(self):
        self.backend.skill = min(1.0, self.backend.skill + self.lr * (1.0 - self.backend.skill))
        return self._evaluate()
