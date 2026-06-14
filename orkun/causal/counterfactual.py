"""Counterfactual ablation — turns correlation into causal necessity via do().

The sandbox is deterministic, so `do(remove a_i)` is exact: re-run the trajectory
without action i in a fresh seeded sandbox and measure the graded-reward drop.
necessity[i] = graded(full) - graded(\\a_i). Necessary actions break the task when
removed (necessity > 0); redundant ones don't (≈0). This is intervention, not
correlation — the verifier (the world) labels every counterfactual for free.

`pruned_calls` returns the minimal causal trajectory (drops zero-necessity actions)
used to build the cf-distill corpus.
"""
from __future__ import annotations

from dataclasses import dataclass

from infer.monkey_wire import ToolCall
from scripts.verifier import Task
from orkun.world.verifier import run_fresh

_EPS = 1e-9


@dataclass
class CounterfactualResult:
    base_passed: bool
    base_graded: float
    necessity: list[float]          # per original action, graded-reward drop on removal
    cf_variants: list[dict]         # {"action_idx", "passed", "graded"} per ablation


def ablation_necessity(task: Task, calls: list[ToolCall]) -> CounterfactualResult:
    base = run_fresh(task, calls)
    necessity: list[float] = []
    variants: list[dict] = []
    for i in range(len(calls)):
        cf_calls = calls[:i] + calls[i + 1:]      # do(remove a_i)
        r = run_fresh(task, cf_calls)
        drop = max(0.0, base.graded - r.graded)
        necessity.append(drop)
        variants.append({"action_idx": i, "passed": r.passed, "graded": r.graded})
    return CounterfactualResult(
        base_passed=base.passed,
        base_graded=base.graded,
        necessity=necessity,
        cf_variants=variants,
    )


def pruned_calls(calls: list[ToolCall], necessity: list[float]) -> list[ToolCall]:
    """Keep only actions whose ablation dropped reward (necessity > 0)."""
    return [c for c, n in zip(calls, necessity) if n > _EPS]
