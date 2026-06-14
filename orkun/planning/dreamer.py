"""Dreamer — imagine m trajectories from a goal, rank, then verify the best for real.

dream(): the policy proposes m candidate assistant turns; the reward-head ranks the
imagined wires. execute_verified(): runs the top-N candidates' calls in a fresh
sandbox; only the ones that actually pass become real verified trajectories
(SelfPlayResult), eligible for the corpus with source "orkun_dream". This is the
"imagination proposes, the sandbox disposes" gate — an over-optimistic imagined
score is discarded the moment the real world disagrees, so hallucinations never
reach the store or the trainer.
"""
from __future__ import annotations

from infer.monkey_wire import parse_calls
from scripts.self_play import SelfPlayResult, _build_wire

from orkun.world.verifier import run_fresh


class Dreamer:
    def __init__(self, policy, wm, reward_head, m: int = 4):
        self.policy = policy
        self.wm = wm
        self.reward_head = reward_head
        self.m = m
        self.known_tools = getattr(policy, "known_tools", None)

    def dream(self, task, m: int | None = None) -> list[tuple[float, str]]:
        m = m or self.m
        scored: list[tuple[float, str]] = []
        for attempt in range(m):
            text = self.policy.gen_fn(task, attempt)
            wire = self.policy._prompt_wire(task) + text
            scored.append((self.reward_head.predict(wire), text))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def execute_verified(self, task, ranked: list[tuple[float, str]], top_n: int = 2) -> list[SelfPlayResult]:
        kept: list[SelfPlayResult] = []
        for _, text in ranked[:top_n]:
            calls = parse_calls(text, known_tools=self.known_tools)
            reward = run_fresh(task, calls)
            if reward.passed:
                wire = _build_wire(task, text, reward.steps)
                kept.append(SelfPlayResult(task_id=task.id, wire=wire, n_calls=len(reward.steps)))
        return kept
