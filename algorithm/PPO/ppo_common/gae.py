from typing import List, Sequence, Tuple


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> Tuple[List[float], List[float]]:

    advantages: List[float] = []
    gae = 0.0
    values_ext = list(values) + [last_value]

    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = (
            rewards[t]
            + gamma * values_ext[t + 1] * nonterminal
            - values_ext[t]
        )
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages.append(gae)
    advantages.reverse()

    returns = [adv + v for adv, v in zip(advantages, values)]
    return advantages, returns