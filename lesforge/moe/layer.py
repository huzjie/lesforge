"""MoE layer: route tokens -> dispatch to experts -> weighted combine."""
from .router import TopKRouter
from .expert import SwiGLUExpert


class MoELayer:
    """One decoder layer's MoE block.

    ``forward`` routes every token to its top-K experts and returns the weighted
    sum of their outputs. ``selected_experts`` exposes the routing decisions so
    the LES scheduler can compute prefetch targets and hit rates.
    """

    def __init__(self, dim=128, hidden=512, n_experts=32, top_k=2, layer_id=0):
        self.dim = dim
        self.hidden = hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.layer_id = layer_id
        self.router = TopKRouter(dim, n_experts, top_k, layer_id)
        self.experts = [SwiGLUExpert(dim, hidden, layer_id * n_experts + e)
                        for e in range(n_experts)]

    def selected_experts(self, hidden_seq):
        """Return the list of top-K expert index sets per token."""
        sel = []
        for h in hidden_seq:
            idxs, _ = self.router.forward(h)
            sel.append(idxs)
        return sel

    def forward(self, hidden_seq):
        out = []
        for h in hidden_seq:
            idxs, probs = self.router.forward(h)
            combined = [0.0] * self.dim
            for i, e in enumerate(idxs):
                o = self.experts[e].forward(h)
                p = probs[i]
                for t in range(self.dim):
                    combined[t] += p * o[t]
            out.append(combined)
        return out

    def n_params(self):
        return sum(e.n_params() for e in self.experts)
