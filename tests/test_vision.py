import unittest

from lesforge.vision.vit import ViTEncoder
from lesforge.vision.patchify import make_image, PatchEmbed


class TestVision(unittest.TestCase):
    def test_patch_embed(self):
        img = make_image(32)
        pe = PatchEmbed(32, 16, 64)
        out = pe.forward(img)
        self.assertEqual(len(out), 4)  # (32/16)^2 patches
        self.assertEqual(len(out[0]), 64)

    def test_vit(self):
        img = make_image(32)
        vit = ViTEncoder(32, 16, 64, depth=2, heads=4)
        emb = vit.forward(img)
        self.assertEqual(len(emb), 64)


if __name__ == "__main__":
    unittest.main()
