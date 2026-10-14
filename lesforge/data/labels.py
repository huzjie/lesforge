"""Deterministic label generator for synthetic VQA (kept minimal)."""
from ..utils.hash import stable_ints


def synth_labels(n, vocab=64):
    return [stable_ints("label:%d" % i, 3, 0, vocab) for i in range(n)]
