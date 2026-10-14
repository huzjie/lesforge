"""Multi-head self-attention (pure Python).

Softmax is computed with the max-subtraction trick for numerical stability.
"""
import math

from ..utils.hash import stable_vector


def _softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    s = sum(exps)
    return [e / s for e in exps]


class MultiHeadAttention:
    """Single multi-head attention layer over a token sequence."""

    def __init__(self, dim=128, heads=4):
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.hd = dim // heads
        self.wq = [stable_vector("attn:q:%d" % i, dim, -0.05, 0.05) for i in range(dim)]
        self.wk = [stable_vector("attn:k:%d" % i, dim, -0.05, 0.05) for i in range(dim)]
        self.wv = [stable_vector("attn:v:%d" % i, dim, -0.05, 0.05) for i in range(dim)]
        self.wo = [stable_vector("attn:o:%d" % i, dim, -0.05, 0.05) for i in range(dim)]

    def _proj(self, x, w):
        return [sum(x[k] * w[k][j] for k in range(self.dim)) for j in range(self.dim)]

    def forward(self, x):
        # x: (seq, dim)
        q = [self._proj(tok, self.wq) for tok in x]
        k = [self._proj(tok, self.wk) for tok in x]
        v = [self._proj(tok, self.wv) for tok in x]
        seq = len(x)
        out = []
        for i in range(seq):
            # gather this head's slice
            qh = q[i]
            scores = []
            for j in range(seq):
                kh = k[j]
                scores.append(sum(qh[t] * kh[t] for t in range(self.dim)) / math.sqrt(self.dim))
            attn = _softmax(scores)
            ctx = [0.0] * self.dim
            for j in range(seq):
                a = attn[j]
                vj = v[j]
                for t in range(self.dim):
                    ctx[t] += a * vj[t]
            out.append(self._proj(ctx, self.wo))
        return out
