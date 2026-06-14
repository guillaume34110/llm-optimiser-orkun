"""Ablation runner — measures whether the LP curriculum beats random.

Curriculum-agnostic: it drives `cycles` of (pick families → attempt → update LP)
against a provided `solve_fn(family) -> bool` and records the running success rate.
`solve_fn` is the seam: a deterministic stub in tests, `OrkishPolicy.rollout` in
production. This is the Phase-A done-criterion harness: LP-curriculum > random-goal.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from orkun.curiosity.curriculum import LPCurriculum, RandomCurriculum
from orkun.curiosity.learning_progress import LearningProgress

SolveFn = Callable[[str], bool]


@dataclass
class AblationArm:
    mode: str
    success_curve: list[float]                       # running success rate per cycle
    per_family_success: dict[str, list[int]] = field(default_factory=dict)
    progress_curve: list[float] = field(default_factory=list)   # curiosity progress per cycle

    def final_success_rate(self, family: str) -> float:
        hist = self.per_family_success.get(family, [])
        if not hist:
            return 0.0
        return sum(hist) / len(hist)


def run_curriculum_ablation(
    mode: str,
    families: list[str],
    solve_fn: SolveFn,
    cycles: int,
    k: int = 1,
    window: int = 8,
    seed: int = 0,
) -> AblationArm:
    rng = random.Random(seed)
    lp = LearningProgress(window=window)
    if mode == "lp":
        curriculum = LPCurriculum(epsilon=0.2, rng=rng)
    elif mode == "random":
        curriculum = RandomCurriculum(rng=rng)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    successes = 0
    attempts = 0
    curve: list[float] = []
    per_family: dict[str, list[int]] = {f: [] for f in families}

    for _ in range(cycles):
        chosen = curriculum.pick(families, lp.predict(families), k=k)
        for fam in chosen:
            ok = solve_fn(fam)
            lp.update(fam, ok)
            per_family[fam].append(1 if ok else 0)
            successes += int(ok)
            attempts += 1
        curve.append(successes / attempts if attempts else 0.0)

    return AblationArm(mode=mode, success_curve=curve, per_family_success=per_family)
