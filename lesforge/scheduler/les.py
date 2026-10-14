"""LEScheduler + LookaheadPredictor -- the heart of lesforge.

Model (deterministic, per the project's mock-backend conventions):

* A decoder has ``n_layers`` MoE layers; each token routes to ``top_k`` experts
  per layer. Experts live in an OffloadStore (SSD) and only ``resident_experts``
  of them fit in RAM at once (ResidencyTable + MemoryBudget).

* ``LookaheadPredictor`` predicts the *next* layer's expert set from the current
  hidden state. Its accuracy is governed by a skill ``s`` in [0.5, 1.0] using
  the proven monotonic boost ``boost = (s - 0.5) * 2``:
      - s = 0.5  -> boost 0  -> random guesses (hit ~ top_k / n_experts)
      - s = 1.0  -> boost 1  -> always predicts the true next-layer experts

* ``LEScheduler.step``:
      1. actual routing for layer L is the ground truth (needed experts);
      2. any needed expert missing from RAM causes an SSD stall (miss);
      3. after compute, predict layer L+lookahead's experts and prefetch them
         (overlapping with the next layer's compute);
      4. evict LRU experts to keep RAM within budget.

Measured outcomes: prefetch hit rate, SSD stalls, and end-to-end speedup
(compute+io serial vs compute-overlapped).
"""
from ..utils.hash import stable_float, stable_vector
from .tier import ResidencyTable
from .offload import OffloadStore, expert_key
from .prefetch import PrefetchPipeline
from .budget import MemoryBudget


class LookaheadPredictor:
    """Predicts layer L+1's top-k experts from layer L's hidden state."""

    def __init__(self, n_experts, top_k, dim=128):
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.w = [stable_vector("predictor:%d" % i, dim, -0.05, 0.05)
                  for i in range(n_experts)]

    def predict(self, hidden, gold_set, skill):
        """Return predicted expert ids for one token, given its gold set.

        ``gold_set`` is the *actual* next-layer routing (available because the
        mock model is deterministic). The skill controls how often we pick the
        gold experts vs. noise -- the classic ``boost=(skill-0.5)*2`` recipe.
        """
        boost = (skill - 0.5) * 2.0
        # affinity score per expert: boost on gold, plus deterministic noise
        scores = []
        for e in range(self.n_experts):
            aff = sum(hidden[k] * self.w[e][k] for k in range(self.dim))
            aff = max(-1.0, min(1.0, aff / (self.dim * 0.05 + 1e-9)))
            base = boost * (1.0 if e in gold_set else 0.0) + 0.05 * aff
            noise = stable_float("pred:%d:%s" % (e, ".".join(map(str, gold_set))), 0.0, 1.0)
            scores.append((base + 0.6 * noise, e))
        scores.sort(reverse=True)
        return [e for _, e in scores[:self.top_k]]


class LEScheduler:
    """Orchestrates expert residency + lookahead prefetch across decoder layers."""

    def __init__(self, cfg, dim=128, hidden=512):
        sched = cfg.schedule
        self.lookahead = sched.lookahead
        self.resident_experts = sched.resident_experts
        self.prefetch_depth = sched.prefetch_depth
        self.eviction = sched.eviction
        self.dim = dim
        self.hidden = hidden
        self.n_experts = cfg.moe.n_experts
        self.top_k = cfg.moe.top_k
        self.budget = MemoryBudget(sched.memory_budget_mb)
        self.expert_mb = self.budget.bytes_per_expert(dim, hidden, sched.quant_bits)
        self.offload = OffloadStore()
        self.resident = ResidencyTable()
        self.pipeline = PrefetchPipeline(sched.prefetch_depth, self.offload)
        self.predictor = LookaheadPredictor(self.n_experts, self.top_k, dim)

        self.hits = 0
        self.needed = 0
        self.stalls = 0
        self.compute_ms = 0.0
        self.io_ms = 0.0

    def _load_missing(self, layer_id, needed_experts):
        """Load needed experts not in RAM; returns (load_ms, missing)."""
        ms = 0.0
        missing = 0
        for e in needed_experts:
            if not self.resident.is_resident(layer_id, e):
                _, lat = self.offload.read(expert_key(layer_id, e))
                ms += lat
                missing += 1
                self.resident.admit(layer_id, e)
                self.stalls += 1
            else:
                self.resident.touch(layer_id, e)
        self.io_ms += ms
        return ms, missing

    def _evict_to_budget(self):
        while self.resident.total_resident() > self.resident_experts:
            self.resident.evict_lru()

    def warmup(self, needed_experts):
        """Preload the first layer's experts (cold start, not a scheduling miss)."""
        for e in needed_experts:
            if not self.resident.is_resident(0, e):
                self.offload.read(expert_key(0, e))
                self.resident.admit(0, e)
            else:
                self.resident.touch(0, e)

    def step(self, layer_id, needed_experts, next_gold, next_hidden, skill,
             compute_ms=8.0):
        """Process one layer: load what's needed, then prefetch the next.

        ``needed_experts`` is a list of expert ids this layer will activate;
        ``next_gold`` is the *true* next-layer routing (available because the
        mock model is deterministic), which the predictor tries to anticipate
        using its skill. Returns a dict with the layer's timing breakdown.
        """
        # 1. ensure this layer's experts are resident (cold miss => stall)
        load_ms, missing = self._load_missing(layer_id, needed_experts)
        # count hit/miss only for layers where lookahead prefetch applies
        # (layer 0 is the cold start and is handled by warmup())
        if layer_id >= 1:
            self.needed += len(needed_experts)
            self.hits += len(needed_experts) - missing

        # 2. compute overlaps with prefetch of the NEXT layer
        self.compute_ms += compute_ms
        prefetched = 0
        if next_gold is not None and next_hidden is not None:
            preds = self.predictor.predict(next_hidden, next_gold, skill)
            for e in preds:
                if self.pipeline.can_issue():
                    self.pipeline.issue(layer_id + 1, e)
                    prefetched += 1
        # 3. drain pipeline -> admitted to RAM (overlapped load finishes)
        self.pipeline.drain(self.resident)

        # 4. enforce budget
        self._evict_to_budget()

        return {
            "layer": layer_id,
            "compute_ms": compute_ms,
            "io_ms": load_ms,
            "missing": missing,
            "prefetched": prefetched,
        }

    def hit_rate(self):
        return self.hits / max(1, self.needed)

    def stats(self):
        return {
            "hit_rate": self.hit_rate(),
            "stalls": self.stalls,
            "compute_ms": self.compute_ms,
            "io_ms": self.io_ms,
            "resident": self.resident.total_resident(),
            "budget_ratio": self.budget.usage_ratio(),
        }
