"""SwiGLU expert feed-forward network.

Each expert is an independent FFN with SwiGLU activation. In the offload setup
these are exactly the blobs that get written to SSD and paged into RAM by the
LES scheduler -- a single expert's weight matrices are the unit of transfer.
"""
import math

from ..utils.hash import stable_vector


def _silu(x):
    return x / (1.0 + math.exp(-x))


class SwiGLUExpert:
    """A single expert: x -> W2 @ (SiLU(W1 @ x) * (W3 @ x))."""

    def __init__(self, dim=128, hidden=512, expert_id=0):
        self.dim = dim
        self.hidden = hidden
        self.expert_id = expert_id
        # W1, W3: (hidden, dim); W2: (dim, hidden)
        self.w1 = [stable_vector("exp:%d:w1:%d" % (expert_id, i), dim, -0.08, 0.08)
                   for i in range(hidden)]
        self.w3 = [stable_vector("exp:%d:w3:%d" % (expert_id, i), dim, -0.08, 0.08)
                   for i in range(hidden)]
        self.w2 = [stable_vector("exp:%d:w2:%d" % (expert_id, i), hidden, -0.08, 0.08)
                   for i in range(dim)]

    def forward(self, x):
        gate = [_silu(sum(x[k] * self.w1[j][k] for k in range(self.dim)))
                for j in range(self.hidden)]
        up = [sum(x[k] * self.w3[j][k] for k in range(self.dim))
              for j in range(self.hidden)]
        h = [gate[j] * up[j] for j in range(self.hidden)]
        return [sum(h[k] * self.w2[j][k] for k in range(self.hidden))
                for j in range(self.dim)]

    def n_params(self):
        return self.dim * self.hidden * 3


def quantize_expert(expert, bits):
    """Deterministic fake-quantization of an expert's weights to ``bits`` bits.

    Real INT4/INT8 quantization needs calibration; here we model the *effect*
    (per-weight rounding to a ``bits``-bit grid) to compute realistic storage
    and transfer-cost numbers in the scheduler.
    """
    if bits == 0:
        return expert
    levels = 2 ** (bits - 1) - 1

    def q(vec):
        return [round(v * levels) / levels for v in vec]

    expert.w1 = [q(r) for r in expert.w1]
    expert.w2 = [q(r) for r in expert.w2]
    expert.w3 = [q(r) for r in expert.w3]
    return expert
