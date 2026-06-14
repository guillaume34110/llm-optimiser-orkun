"""EpistemicCuriosity — surprise + prediction-progress at the observation level.

surprise = -wm.score(context, real_result): how unexpected the real tool result was
under the world-model. Two EMAs per task family track surprise over time: a fast
`recent` EMA and a lagged `old` EMA. progress = max(0, old - recent) is positive
exactly while surprise is falling — i.e. the WM is learning that family (Oudeyer's
learning-progress, applied to observation prediction). curiosity combines a
clipped/normalised surprise with progress. The WM's online finetune (WMTrainer)
makes surprise fall, so progress is a real, earned signal.

Caveat (accepted): the WM shares its net with the policy, so RLVR steps also move
`wm.score` — measured surprise mixes world novelty with policy drift. The dual-EMA
progress is robust to a slow common drift (both EMAs shift together), but a sharp
policy update can transiently inflate surprise; a frozen-copy WM would isolate the
signal at twice the memory.
"""
from __future__ import annotations


class EpistemicCuriosity:
    def __init__(
        self,
        w_surprise: float = 0.5,
        w_progress: float = 0.5,
        ema_alpha: float = 0.1,
        surprise_scale: float = 10.0,
    ):
        self.w_s = w_surprise
        self.w_p = w_progress
        self.alpha = ema_alpha
        self.scale = surprise_scale
        self._recent: dict[str, float] = {}
        self._old: dict[str, float] = {}

    def surprise(self, wm, prefix: str, real_result: str) -> float:
        return -wm.score(prefix, real_result)

    def observe(self, key: str, surprise: float) -> None:
        prev_recent = self._recent.get(key, surprise)
        recent = (1 - self.alpha) * prev_recent + self.alpha * surprise
        self._recent[key] = recent
        # `old` follows the PRIOR `recent` at half the rate, so it genuinely lags a falling series
        old = self._old.get(key, surprise)
        old = (1 - self.alpha / 2) * old + (self.alpha / 2) * prev_recent
        self._old[key] = old

    def progress(self, key: str) -> float:
        if key not in self._recent:
            return 0.0
        return max(0.0, self._old[key] - self._recent[key])

    def curiosity(self, key: str, surprise: float) -> float:
        surprise_norm = max(0.0, min(1.0, surprise / self.scale))
        return self.w_s * surprise_norm + self.w_p * self.progress(key)
