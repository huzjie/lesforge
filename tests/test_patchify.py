import unittest

from lesforge.vision.patchify import make_image, PatchEmbed


class TestPatchify(unittest.TestCase):
    def test_make_image_shape(self):
        img = make_image(16)
        self.assertEqual(len(img), 16)
        self.assertEqual(len(img[0]), 16)

    def test_patch_embed_count(self):
        pe = PatchEmbed(16, 8, 32)
        self.assertEqual(pe.n_patches, 4)


if __name__ == "__main__":
    unittest.main()
