"""Iterate the synthetic dataset through the mock backend."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.backends import get_backend
from lesforge.data.synthetic import SynthDataset

cfg = Config()
b = get_backend("mock", cfg)
ds = SynthDataset(n=4, img_size=16)
for img, prompt in zip(ds.images(), ds.prompts()):
    text, _ = b.generate(prompt, image=img)
    print("%-28s -> %s" % (prompt, text))
