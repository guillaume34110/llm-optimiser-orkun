"""Curriculum policies — pick which goal families to spend rollouts on.

LPCurriculum: epsilon-greedy on the learning-progress scores (exploit high-LP
families, explore uniformly with prob epsilon). RandomCurriculum: the ablation
baseline that ignores LP entirely. Both return up to k family names.
"""
from __future__ import annotations

import random


class LPCurriculum:
    def __init__(self, epsilon: float = 0.2, rng: random.Random | None = None):
        self.epsilon = epsilon
        self.rng = rng or random.Random()

    def pick(self, families: list[str], lp_scores: dict[str, float], k: int) -> list[str]:
        k = min(k, len(families))
        if self.rng.random() < self.epsilon:
            return self.rng.sample(families, k)
        ranked = sorted(families, key=lambda f: lp_scores.get(f, 0.0), reverse=True)
        return ranked[:k]


class RandomCurriculum:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def pick(self, families: list[str], lp_scores: dict[str, float], k: int) -> list[str]:
        k = min(k, len(families))
        return self.rng.sample(families, k)
