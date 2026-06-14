# orkun/eval/loop_ablation.py
"""Loop-level ablations for Done-B (ii curiosity on/off, iii WM online vs frozen).

Each arm drives `cycles` of an injected `cycle_fn(arm_cfg, rng)` seam and records the
per-cycle success and curiosity progress. `arm_cfg` is the toggle dict the cycle
interprets (e.g. {"curiosity": True} or {"wm_online": False}); in production `cycle_fn`
runs one `kunnin_cycle` with those flags and returns its outcome, in tests it is a
deterministic stub. Reuses `AblationArm` (success_curve, per_family_success,
progress_curve). `coverage` counts families solved at least once — the (ii) coverage
metric. (iii) reads `progress_curve`: WM-online should trend positive, frozen ≈ 0.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from orkun.eval.ablation import AblationArm


@dataclass
class CycleMetrics:
    success: bool
    progress: float             # curiosity.progress for this cycle's family (0.0 if none)
    family: str


CycleFn = Callable[[dict, random.Random], CycleMetrics]   # (arm_cfg, rng) -> metrics


def run_loop_ablation(arms: dict[str, dict], cycle_fn: CycleFn, cycles: int,
                      families: list[str], seed: int = 0) -> dict[str, AblationArm]:
    """Drive `cycles` of `cycle_fn` per arm; return one AblationArm per arm name."""
    out: dict[str, AblationArm] = {}
    for name, arm_cfg in arms.items():
        rng = random.Random(seed)
        successes = 0
        attempts = 0
        success_curve: list[float] = []
        progress_curve: list[float] = []
        per_family: dict[str, list[int]] = {f: [] for f in families}
        for _ in range(cycles):
            m = cycle_fn(arm_cfg, rng)
            per_family.setdefault(m.family, []).append(1 if m.success else 0)
            successes += int(m.success)
            attempts += 1
            success_curve.append(successes / attempts)
            progress_curve.append(m.progress)
        out[name] = AblationArm(mode=name, success_curve=success_curve,
                                per_family_success=per_family, progress_curve=progress_curve)
    return out


def coverage(arm: AblationArm, families: list[str]) -> int:
    """Number of families solved at least once (the (ii) task-coverage metric)."""
    return sum(1 for f in families if sum(arm.per_family_success.get(f, [])) > 0)
