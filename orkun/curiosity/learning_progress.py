"""Learning-progress estimator — non-stationary per-family success-rate derivative.

LP(family) = |mean(recent half of window) - mean(older half)|. High while the
policy is improving on a family (recent successes > older), ≈0 once mastered
(both halves ≈1) or hopeless (both ≈0). This is the curiosity signal: the
curriculum spends rollouts where |LP| is highest (the zone of proximal
development). Window is a deque per family so the estimate tracks a moving policy.
"""
from __future__ import annotations

from collections import deque


class LearningProgress:
    def __init__(self, window: int = 20):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self._hist: dict[str, deque[int]] = {}

    def update(self, family: str, success: bool) -> None:
        dq = self._hist.setdefault(family, deque(maxlen=self.window))
        dq.append(1 if success else 0)

    def predict(self, families: list[str]) -> dict[str, float]:
        return {f: self._lp(f) for f in families}

    def _lp(self, family: str) -> float:
        dq = self._hist.get(family)
        if dq is None or len(dq) < 2:
            return 0.0
        seq = list(dq)
        half = len(seq) // 2
        older = seq[:half]
        recent = seq[half:]
        if not older or not recent:
            return 0.0
        return abs(sum(recent) / len(recent) - sum(older) / len(older))
