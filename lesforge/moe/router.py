"""Top-K gating router.

The router scores every expert for a given token, then selects the top-K. The
scores are computed with a deterministic affine transform of the hidden state,
so the same hidden state always routes to the same experts (reproducible).

This selection is exactly what the LES scheduler consumes: layer L's router
output is the ground truth for which experts layer L needs, and the lookahead
predictor tries to anticipate layer L+1's selection one step early.
"""
from ..utils.hash import stable_vector


def _topk_indices(scores, k):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order[:k]


def _softmax(scores):
    import math
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    s = sum(exps)
    return [e / s for e in exps]


class TopKRouter:
    """Deterministic top-K expert router for a single MoE layer."""

    def __init__(self, dim=128, n_experts=32, top_k=2, layer_id=0):
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.layer_id = layer_id
        # gate weights: (n_experts, dim)
        self.w = [
            stable_vector("router:%d:%d" % (layer_id, e), dim, -0.05, 0.05)
            for e in range(n_experts)
        ]

    def logits(self, hidden):
        return [sum(hidden[k] * self.w[e][k] for k in range(self.dim))
                for e in range(self.n_experts)]

    def forward(self, hidden):
        """Return (expert_indices, gate_weights) for one token."""
        logits = self.logits(hidden)
        idxs = _topk_indices(logits, self.top_k)
        probs = _softmax([logits[i] for i in idxs])
        return idxs, probs


def route_batch(router, hidden_seq):
    """Route a full sequence, returning per-token (indices, weights)."""
    out = []
    for h in hidden_seq:
        out.append(router.forward(h))
    return out
