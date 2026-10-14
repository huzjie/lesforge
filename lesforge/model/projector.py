"""Vision -> language projector (aligns the two embedding spaces)."""
from ..utils.hash import stable_vector


class Projector:
    """Linear projection from vision dim to decoder dim."""

    def __init__(self, vision_dim=128, decoder_dim=128):
        self.vision_dim = vision_dim
        self.decoder_dim = decoder_dim
        self.w = [stable_vector("proj:%d" % i, decoder_dim, -0.05, 0.05)
                  for i in range(vision_dim)]

    def forward(self, vision_embedding):
        return [sum(vision_embedding[k] * self.w[k][j]
                    for k in range(self.vision_dim))
                for j in range(self.decoder_dim)]
