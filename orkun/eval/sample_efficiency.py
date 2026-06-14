# orkun/eval/sample_efficiency.py
"""Planning-beats-reactive sample-efficiency measurement (Done-B ablation i).

For each task the harness runs two arms through an injected `attempt_fn` seam — a
"reactive" arm (Phase A blind sampling) and a "planned" arm (lookahead/dreamer execute
the WM-ranked plan for real until verified) — and records how many REAL sandbox
executions each consumed. The means are over all tasks (a failed attempt still counts
the executions it burned, penalising failure-by-exhaustion). gain_ratio = reactive_mean /
planned_mean; > 1 means planning wins. In production the seam wires OrkishPolicy.rollout
(reactive) and Dreamer.execute_verified (planned); in tests it is a deterministic stub.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class AttemptResult:
    solved: bool
    n_real_exec: int            # real sandbox executions consumed by this attempt


AttemptFn = Callable[[object, str], AttemptResult]   # (task, mode) -> result; mode in {"reactive","planned"}


@dataclass
class SampleEfficiencyReport:
    reactive_mean_exec: float
    planned_mean_exec: float
    gain_ratio: float                                # reactive_mean / planned_mean (>1 = planning wins)
    per_task: dict[str, tuple[int, int]]             # task_id -> (reactive_exec, planned_exec)


def _gain_ratio(reactive_mean: float, planned_mean: float) -> float:
    if planned_mean == 0.0:
        return 1.0 if reactive_mean == 0.0 else float("inf")
    return reactive_mean / planned_mean


def run_sample_efficiency(tasks: list, attempt_fn: AttemptFn, seed: int = 0
                          ) -> SampleEfficiencyReport:
    per_task: dict[str, tuple[int, int]] = {}
    reactive_total = 0
    planned_total = 0
    for task in tasks:
        reactive = attempt_fn(task, "reactive")
        planned = attempt_fn(task, "planned")
        per_task[task.id] = (reactive.n_real_exec, planned.n_real_exec)
        reactive_total += reactive.n_real_exec
        planned_total += planned.n_real_exec
    n = len(tasks)
    reactive_mean = reactive_total / n if n else 0.0
    planned_mean = planned_total / n if n else 0.0
    return SampleEfficiencyReport(
        reactive_mean_exec=reactive_mean,
        planned_mean_exec=planned_mean,
        gain_ratio=_gain_ratio(reactive_mean, planned_mean),
        per_task=per_task,
    )
