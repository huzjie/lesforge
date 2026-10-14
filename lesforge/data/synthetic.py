"""Deterministic synthetic VLM dataset (images + prompts + labels)."""
from ..vision.patchify import make_image
from ..utils.hash import stable_ints, stable_float


class SynthDataset:
    """A small synthetic image-caption/VQA dataset for benchmarks and demos."""

    def __init__(self, n=16, img_size=32):
        self.n = n
        self.img_size = img_size

    def images(self):
        return [make_image(self.img_size, seed_key="ds:%d" % i) for i in range(self.n)]

    def prompts(self):
        from .prompts import PROMPTS
        return [PROMPTS[i % len(PROMPTS)] for i in range(self.n)]

    def labels(self):
        return ["label-%d" % i for i in range(self.n)]

    def iter_batches(self, batch_size=4):
        imgs, prs = self.images(), self.prompts()
        for i in range(0, self.n, batch_size):
            yield imgs[i:i + batch_size], prs[i:i + batch_size]


def make_image_batch(n, img_size=32):
    return [make_image(img_size, seed_key="batch:%d" % i) for i in range(n)]
