"""Image -> patch embeddings.

A tiny image is represented as a list of rows, each row a list of pixel floats.
We tile it into (img_size/patch_size)^2 patches, flatten each patch, and project
to ``dim`` with a deterministic weight matrix.
"""
from ..utils.hash import stable_vector


def _make_image(img_size, seed_key="img"):
    """Deterministic synthetic grayscale image (list of rows)."""
    rows = []
    for y in range(img_size):
        row = []
        for x in range(img_size):
            # low-frequency + high-frequency mix for a non-trivial signal
            v = stable_vector("%s:%d:%d" % (seed_key, x, y), 1, -1.0, 1.0)[0]
            row.append(v)
        rows.append(row)
    return rows


class PatchEmbed:
    """Tiles an image into patches and projects each to ``dim``."""

    def __init__(self, img_size=224, patch_size=16, dim=128):
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.n_patches = (img_size // patch_size) ** 2
        self.w = self._build_w()

    def _build_w(self):
        n = self.patch_size * self.patch_size
        return [
            stable_vector("patch:w:%d" % i, self.dim, -0.05, 0.05)
            for i in range(n)
        ]

    def _patches(self, image):
        ps = self.patch_size
        n = self.img_size // ps
        patches = []
        for py in range(n):
            for px in range(n):
                flat = []
                for dy in range(ps):
                    for dx in range(ps):
                        flat.append(image[py * ps + dy][px * ps + dx])
                patches.append(flat)
        return patches

    def forward(self, image):
        patches = self._patches(image)
        out = []
        for p in patches:
            out.append([sum(p[k] * self.w[k][j] for k in range(len(p)))
                        for j in range(self.dim)])
        return out  # (n_patches, dim)


def make_image(img_size=224, seed_key="img"):
    return _make_image(img_size, seed_key)
