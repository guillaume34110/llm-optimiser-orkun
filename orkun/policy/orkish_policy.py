"""OrkishPolicy — adapts any LMBackend to the Orkish self-play GenFn.

`gen_fn(task, attempt) -> str` builds a prompt wire ending in <|assistant|> and
returns the backend's continuation. `rollout` defers to Orkish `play_task` —
rejection sampling against the verifier, keeping the first trajectory whose every
check passes. `attempt` reseeds the backend so retries explore (Orkish needs
stochastic samples for rejection sampling). The policy depends on the LMBackend
protocol, so changing the LLM means passing a different backend — not editing here.
"""
from __future__ import annotations

from scripts.self_play import SelfPlayResult, play_task
from scripts.verifier import Task
from orkun.policy.backends.waaagh import WaaaghBackend
from orkun.policy.base import LMBackend
from orkun.policy.prompting import prompt_wire


class OrkishPolicy:
    def __init__(
        self,
        backend: LMBackend,
        system: str | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new: int = 256,
        known_tools=None,
        base_seed: int = 0,
    ):
        self.backend = backend
        self.system = system
        self.temperature = temperature
        self.top_p = top_p
        self.max_new = max_new
        self.known_tools = known_tools
        self.base_seed = base_seed

    def _prompt_wire(self, task: Task) -> str:
        return prompt_wire(task, self.system)

    def gen_fn(self, task: Task, attempt: int) -> str:
        return self.backend.generate(
            self._prompt_wire(task),
            max_new=self.max_new,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_strings=["<|eos|>", "<|user|>"],
            seed=self.base_seed + attempt,
        )

    def rollout(self, task: Task, samples: int = 8) -> SelfPlayResult | None:
        return play_task(task, self.gen_fn, samples, known_tools=self.known_tools)


def make_orkish_policy(net, tokenizer, **kwargs) -> OrkishPolicy:
    """Convenience constructor: the common WaaaghBackend-backed policy."""
    return OrkishPolicy(WaaaghBackend(net, tokenizer), **kwargs)
