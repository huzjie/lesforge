"""Synthetic data: images, prompts, labels (deterministic)."""
from .synthetic import SynthDataset, make_image_batch
from .prompts import PROMPTS

__all__ = ["SynthDataset", "make_image_batch", "PROMPTS"]
