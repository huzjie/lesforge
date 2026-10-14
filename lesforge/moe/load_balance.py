"""Auxiliary load-balancing loss for MoE routing.

Without a balancing signal, a router can collapse onto a handful of experts and
starve the rest, which wastes the offload layout and hurts throughput. The
standard Switch-style auxiliary loss penalises uneven expert usage.
"""


def load_balance_loss(routing_counts, n_experts, n_tokens):
    """Switch-transformer aux loss: N * sum(f_i * P_i).

    ``routing_counts[i]`` = number of tokens routed to expert i.
    """
    if n_tokens == 0:
        return 0.0
    f = [c / n_tokens for c in routing_counts]
    # mean routing probability across tokens ~ uniform; use count fraction
    p = [c / n_tokens for c in routing_counts]
    return n_experts * sum(f[i] * p[i] for i in range(n_experts))


def routing_counts(selected, n_experts):
    counts = [0] * n_experts
    for sel in selected:
        for e in sel:
            counts[e] += 1
    return counts
