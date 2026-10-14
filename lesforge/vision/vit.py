"""A minimal ViT encoder (patch embed + stacked attention blocks)."""
from ..utils.hash import stable_vector
from .attention import MultiHeadAttention
from .patchify import PatchEmbed


def _layernorm(x, eps=1e-5):
    n = len(x)
    mu = sum(x) / n
    var = sum((xi - mu) ** 2 for xi in x) / n
    inv = 1.0 / (var + eps) ** 0.5
    return [(xi - mu) * inv for xi in x]


def _gelu(x):
    import math
    return 0.5 * x * (1.0 + math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class ViTEncoder:
    """Patch embed -> L attention+MLP blocks -> pooled embedding."""

    def __init__(self, img_size=224, patch_size=16, dim=128, depth=4, heads=4):
        self.patch_embed = PatchEmbed(img_size, patch_size, dim)
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.attns = [MultiHeadAttention(dim, heads) for _ in range(depth)]
        self.mlp_w1 = [stable_vector("vit:mlp1:%d:%d" % (l, i), dim, -0.05, 0.05)
                       for l in range(depth) for i in range(dim)]
        self.mlp_w2 = [stable_vector("vit:mlp2:%d:%d" % (l, i), dim, -0.05, 0.05)
                       for l in range(depth) for i in range(dim)]

    def _block(self, l, x):
        # pre-norm attention + residual
        normed = [_layernorm(tok) for tok in x]
        attn_out = self.attns[l].forward(normed)
        x = [[x[i][t] + attn_out[i][t] for t in range(self.dim)] for i in range(len(x))]
        # pre-norm MLP + residual
        normed2 = [_layernorm(tok) for tok in x]
        mlp_out = []
        for tok in normed2:
            h = [_gelu(sum(tok[k] * self.mlp_w1[l * self.dim + j][k] for k in range(self.dim)))
                 for j in range(self.dim)]
            o = [sum(h[k] * self.mlp_w2[l * self.dim + j][k] for k in range(self.dim))
                 for j in range(self.dim)]
            mlp_out.append(o)
        return [[x[i][t] + mlp_out[i][t] for t in range(self.dim)] for i in range(len(x))]

    def forward(self, image):
        x = self.patch_embed.forward(image)
        for l in range(self.depth):
            x = self._block(l, x)
        # global average pool -> (dim,)
        n = len(x)
        return [sum(x[i][t] for i in range(n)) / n for t in range(self.dim)]
