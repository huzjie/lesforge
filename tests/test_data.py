import unittest

from lesforge.data.synthetic import SynthDataset, make_image_batch
from lesforge.data.prompts import PROMPTS


class TestData(unittest.TestCase):
    def test_dataset(self):
        ds = SynthDataset(n=8, img_size=16)
        self.assertEqual(len(ds.images()), 8)
        self.assertEqual(len(ds.prompts()), 8)

    def test_batches(self):
        ds = SynthDataset(n=8, img_size=16)
        batches = list(ds.iter_batches(batch_size=4))
        self.assertEqual(len(batches), 2)

    def test_prompts(self):
        self.assertGreater(len(PROMPTS), 0)

    def test_batch(self):
        self.assertEqual(len(make_image_batch(3, img_size=16)), 3)


if __name__ == "__main__":
    unittest.main()
