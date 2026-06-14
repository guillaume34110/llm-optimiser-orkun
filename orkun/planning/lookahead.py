"""LookaheadPlanner — imagine k plans, rank by reward-head, execute best-first.

The policy proposes k candidate assistant turns (varying the sampling seed). For
each, the WM imagines the tool results (predict) to build an imagined wire, which
the reward-head scores. We then run the candidates through Orkish's `play_task` in
ranked order: the first one the sandbox actually verifies is kept. Imagination only
reorders which real plan we try first — the sandbox stays the sole oracle, and an
over-optimistic imagined score never enters the corpus (only verified wires do).
"""
from __future__ import annotations

from infer.monkey_wire import parse_calls
from scripts.self_play import play_task


class LookaheadPlanner:
    def __init__(self, policy, wm, reward_head, k: int = 4, horizon: int = 3):
        self.policy = policy
        self.wm = wm
        self.reward_head = reward_head
        self.k = k
        self.horizon = horizon
        self.known_tools = getattr(policy, "known_tools", None)

    def _imagined_wire(self, task, assistant_text: str) -> str:
        wire = self.policy._prompt_wire(task) + assistant_text
        calls = parse_calls(assistant_text, known_tools=self.known_tools)[: self.horizon]
        for call in calls:
            predicted = self.wm.predict(wire, max_new=48)
            wire = wire + f"<|tool|>{call.name}\n{predicted}"
        return wire

    def imagine(self, task, n: int | None = None) -> list[tuple[float, str]]:
        n = n or self.k
        scored: list[tuple[float, str]] = []
        for attempt in range(n):
            text = self.policy.gen_fn(task, attempt)
            score = self.reward_head.predict(self._imagined_wire(task, text))
            scored.append((score, text))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def rollout(self, task, samples: int = 8):
        ranked = self.imagine(task, n=samples)
        texts = [text for _, text in ranked]

        def gen_fn(t, attempt):
            return texts[attempt] if attempt < len(texts) else self.policy.gen_fn(t, attempt)

        return play_task(task, gen_fn, samples, known_tools=self.known_tools)
