"""DifficultyTracker — per-family difficulty progression for GoalGen.

GoalGen reads the per-family difficulty from `context["difficulty"]`; without a
tracker the loop passes `context={}` and every family sits at the default level
forever — half the curriculum (the LP side picks *which* family, nothing picks
*how hard*) is inert. The tracker closes that loop: it watches the recent success
rate per family AT THE CURRENT LEVEL and promotes when the family is mastered
(rate ≥ `promote_at` over a full window), demotes when it is hopeless (rate ≤
`demote_at`). The observation window resets on every level change so the rate
always describes the level being played, not a mix — this is what prevents the
promote/demote oscillation a stateless rule (derive level from a global success
rate) would suffer.
"""
from __future__ import annotations

from collections import deque


class DifficultyTracker:
    def __init__(
        self,
        start: int = 0,
        max_level: int = 4,
        promote_at: float = 0.8,
        demote_at: float = 0.2,
        window: int = 8,
    ):
        if not 0.0 <= demote_at < promote_at <= 1.0:
            raise ValueError("need 0 <= demote_at < promote_at <= 1")
        self.start = int(start)
        self.max_level = int(max_level)
        self.promote_at = promote_at
        self.demote_at = demote_at
        self.window = int(window)
        self._level: dict[str, int] = {}
        self._hist: dict[str, deque[int]] = {}

    def level(self, family: str) -> int:
        return self._level.get(family, self.start)

    def update(self, family: str, success: bool) -> None:
        dq = self._hist.setdefault(family, deque(maxlen=self.window))
        dq.append(1 if success else 0)
        if len(dq) < self.window:
            return  # judge only on a full window at the current level
        rate = sum(dq) / len(dq)
        lvl = self.level(family)
        if rate >= self.promote_at and lvl < self.max_level:
            self._level[family] = lvl + 1
            dq.clear()
        elif rate <= self.demote_at and lvl > 0:
            self._level[family] = lvl - 1
            dq.clear()

    def context(self) -> dict:
        """GoalGen-shaped context: families never seen fall back to `start` via level()."""
        return {"difficulty": {f: self.level(f) for f in set(self._level) | set(self._hist)}}
