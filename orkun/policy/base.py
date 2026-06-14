"""Policy seams — protocols that decouple the loop from any specific LLM.

`LMBackend` is the low-level text engine: given a prompt string, return generated
text. Swapping the LLM (local WaaaghNet now; an HTTP API, llama.cpp, etc. later)
means writing one new LMBackend — nothing else in Orkun changes. `Policy` is the
high-level seam the loop depends on: it turns a Task into a verified trajectory.
Both are structural typing.Protocols, so implementations match by shape, with no
base-class import coupling.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from scripts.self_play import SelfPlayResult
from scripts.verifier import Task


@runtime_checkable
class LMBackend(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_new: int,
        temperature: float,
        top_p: float,
        stop_strings: list[str],
        seed: int | None,
    ) -> str:
        """Return text generated after `prompt`, stopping at any `stop_strings` or `max_new`."""
        ...


@runtime_checkable
class Policy(Protocol):
    known_tools: object  # frozenset[str] | None — tool names the wire parser accepts

    def rollout(self, task: Task, samples: int) -> SelfPlayResult | None:
        """Attempt the task up to `samples` times; return the first verified trajectory."""
        ...
