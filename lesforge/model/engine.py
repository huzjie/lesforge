"""VLMEngine: vision -> projector -> MoE decoder, scheduled by LES.

The engine is the concrete thing a backend drives. It:

1. encodes the image with the ViT tower and projects it into the decoder space;
2. builds the token sequence (vision prefix + text prompt);
3. steps the MoE decoder layer by layer, feeding each layer's *actual* expert
   selection to the LEScheduler and, one step ahead, the *predicted* next-layer
   selection for prefetch;
4. collects scheduler metrics (hit rate, stalls, compute/io time).

A real deployment would swap the mock experts for real weights; the scheduling
logic (tiering + lookahead prefetch) is identical.
"""
from ..vision.vit import ViTEncoder
from ..vision.patchify import make_image
from ..model.projector import Projector
from ..model.decoder import MoEDecoder
from ..scheduler.les import LEScheduler
from ..utils.hash import stable_ints


class VLMEngine:
    """End-to-end on-device MoE VLM with lookahead expert scheduling."""

    def __init__(self, cfg):
        self.cfg = cfg
        v = cfg.vision
        m = cfg.moe
        self.vision = ViTEncoder(v.img_size, v.patch_size, v.dim, v.depth, v.heads)
        self.projector = Projector(v.dim, m.dim)
        self.decoder = MoEDecoder(
            m.dim, m.hidden, m.n_experts, m.top_k,
            n_layers=m.n_layers, vocab_size=cfg.model.vocab_size,
        )
        self.scheduler = LEScheduler(cfg, m.dim, m.hidden)

    def _token_ids(self, prompt):
        return stable_ints("tok:" + prompt, len(prompt) // 2 + 1, 0, self.cfg.model.vocab_size)

    @staticmethod
    def _mean_pool(hidden_seq):
        n = len(hidden_seq)
        dim = len(hidden_seq[0])
        return [sum(hidden_seq[i][t] for i in range(n)) / n for t in range(dim)]

    def route_all_layers(self, hidden_seq):
        """Return per-layer routing decisions (list of list of expert-id lists)."""
        routing = []
        for l in range(self.decoder.n_layers):
            routing.append(self.decoder.layer_routing(l, hidden_seq))
        return routing

    def generate(self, image=None, prompt="describe", skill=None, max_tokens=None):
        """Run a full forward pass under LES scheduling.

        Returns a dict with the generated text and scheduler stats.
        """
        skill = self.cfg.model.skill if skill is None else skill
        max_tokens = self.cfg.model.max_tokens if max_tokens is None else max_tokens
        if image is None:
            image = make_image(self.cfg.vision.img_size)

        vision_emb = self.vision.forward(image)
        prefix = self.projector.forward(vision_emb)
        text_ids = self._token_ids(prompt)
        # hidden sequence: vision prefix (1 token) + prompt tokens
        hidden_seq = [prefix] + self.decoder.embed(text_ids)

        # Precompute each layer's "needed" experts: route the mean-pooled hidden
        # state (the deterministic mock makes this the true future routing).
        needed_by_layer = []
        probe = hidden_seq
        for l in range(self.decoder.n_layers):
            mean = self._mean_pool(probe)
            idxs, _ = self.decoder.layers[l].router.forward(mean)
            needed_by_layer.append(idxs)
            probe = self.decoder.forward_layer(l, probe)

        # Warm up layer 0 (cold start; not a scheduling miss).
        self.scheduler.warmup(needed_by_layer[0])

        for l in range(self.decoder.n_layers):
            needed = needed_by_layer[l]
            next_gold = needed_by_layer[l + 1] if l + 1 < self.decoder.n_layers else None
            next_hidden = self._mean_pool(hidden_seq) if next_gold is not None else None
            self.scheduler.step(l, needed, next_gold, next_hidden, skill)
            hidden_seq = self.decoder.forward_layer(l, hidden_seq)

        # greedy decode
        generated = []
        state = hidden_seq[-1]
        for _ in range(max_tokens):
            logits = self.decoder.logits(state)
            t = max(range(len(logits)), key=lambda i: logits[i])
            generated.append(t)
            state = self.decoder.emb[t % self.decoder.vocab_size]
        text = self._decode(generated)

        return {
            "text": text,
            "stats": self.scheduler.stats(),
            "n_layers": self.decoder.n_layers,
        }

    def _decode(self, token_ids):
        # deterministic mock decode: map token ids to readable words
        words = ["the", "a", "image", "object", "scene", "shows", "red", "blue",
                 "green", "person", "car", "tree", "building", "sky", "water",
                 "and", "with", "in", "of", "on"]
        return " ".join(words[t % len(words)] for t in token_ids[:8])
