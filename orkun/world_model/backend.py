"""WorldModelBackend seam — a swappable observation-space forward model.

Structurally typed like `LMBackend`, but a separate protocol: the WM can be a
different trained LLM than the policy. `score` answers "how probable is this real
tool result?" (surprise = -score); `encode` gives a pooled hidden representation
the reward-head reads. Any trained checkpoint that can answer both is a valid WM.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WorldModelBackend(Protocol):
    def score(self, prefix: str, continuation: str) -> float:
        """Mean per-token logprob of `continuation` given `prefix` (one forward,
        log-softmax, gather over the continuation span). Surprise = -score."""
        ...

    def encode(self, text: str) -> list[float]:
        """Pooled hidden representation (final-token hidden) of `text`."""
        ...
