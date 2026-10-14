"""Minimal wall-clock + throughput helpers (stdlib only)."""
import time


class Stopwatch:
    def __init__(self):
        self._t0 = None
        self.elapsed = 0.0

    def start(self):
        self._t0 = time.perf_counter()
        return self

    def stop(self):
        if self._t0 is not None:
            self.elapsed = time.perf_counter() - self._t0
        return self.elapsed

    def reset(self):
        self._t0 = None
        self.elapsed = 0.0


def tokens_per_sec(n_tokens, seconds):
    return n_tokens / max(1e-9, seconds)
