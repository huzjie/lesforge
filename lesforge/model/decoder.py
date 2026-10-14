"""MoE decoder: token embeddings -> stacked MoE layers -> logits."""
from ..utils.hash import stable_vector
from ..moe.layer import MoELayer


class MoEDecoder:
    """A decoder composed of ``n_layers`` MoE layers.

    Each layer has its own router (so expert selection varies per layer, which
    is what makes lookahead scheduling non-trivial) and its own expert set.
    """

    def __init__(self, dim=128, hidden=512, n_experts=32, top_k=2, n_layers=4,
                 vocab_size=4096):
        self.dim = dim
        self.hidden = hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.emb = [stable_vector("emb:%d" % i, dim, -0.1, 0.1)
                    for i in range(vocab_size)]
        self.layers = [MoELayer(dim, hidden, n_experts, top_k, layer_id=l)
                       for l in range(n_layers)]
        self.lm_head = [stable_vector("lmhead:%d" % i, vocab_size, -0.05, 0.05)
                        for i in range(dim)]

    def embed(self, token_ids):
        return [self.emb[t % self.vocab_size] for t in token_ids]

    def layer_routing(self, layer_id, hidden_seq):
        return self.layers[layer_id].selected_experts(hidden_seq)

    def forward_layer(self, layer_id, hidden_seq):
        return self.layers[layer_id].forward(hidden_seq)

    def logits(self, hidden):
        return [sum(hidden[k] * self.lm_head[k][j] for k in range(self.dim))
                for j in range(self.vocab_size)]

    def n_params(self):
        return sum(l.n_params() for l in self.layers)
