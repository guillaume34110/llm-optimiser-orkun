"""GoalGen — autotelic goal proposal with a verifiability gate.

The agent proposes its own goals; in Phase A goals are drawn from parametric
families (Task 3) at a per-family difficulty supplied via `context` (the
curriculum will set this). A goal is admitted ONLY if its oracle solves it in a
fresh sandbox — guaranteeing a well-defined verifier reward and keeping
infeasible (LP≈0) goals out of the loop. (MAGELLAN-style language-proposed goals
are a later extension; the verifiability gate is the same.)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from infer.monkey_wire import ToolCall
from scripts.verifier import Task
from orkun.goals.families import GoalFamily
from orkun.world.verifier import run_fresh


@dataclass
class ProposedGoal:
    task: Task
    family: str
    difficulty: int
    oracle_calls: list[ToolCall]


class GoalGen:
    def __init__(self, families: list[GoalFamily], seed: int = 0, default_difficulty: int = 1):
        if not families:
            raise ValueError("GoalGen needs at least one family")
        self.families = families
        self.rng = random.Random(seed)
        self.default_difficulty = default_difficulty

    def _difficulty_for(self, context: dict, family_name: str) -> int:
        return int(context.get("difficulty", {}).get(family_name, self.default_difficulty))

    def propose(self, context: dict, n: int, max_tries_per_goal: int = 5) -> list[ProposedGoal]:
        out: list[ProposedGoal] = []
        # Global cap so a regressed oracle (no family ever admits) fails loudly
        # instead of spinning forever; generous enough not to bite healthy runs.
        budget = n * max_tries_per_goal * max(1, len(self.families)) * 4
        while len(out) < n:
            if budget <= 0:
                raise RuntimeError(
                    f"GoalGen.propose: exhausted try budget with {len(out)}/{n} goals "
                    "admitted — no family is producing verifier-passing goals"
                )
            family = self.rng.choice(self.families)
            difficulty = self._difficulty_for(context, family.name)
            admitted = None
            for _ in range(max_tries_per_goal):
                budget -= 1
                task = family.sample(self.rng, difficulty)
                calls = family.oracle(task)
                if run_fresh(task, calls).passed:
                    admitted = ProposedGoal(task, family.name, difficulty, calls)
                    break
            if admitted is not None:
                out.append(admitted)
        return out
