"""Parallel sandbox pool — ThreadPoolExecutor wrappers for oracle verify + counterfactual.

run_fresh creates an isolated temporary sandbox per call, so oracle verification and
counterfactual ablation are embarrassingly parallel. ThreadPoolExecutor (not Process)
because: (a) Orkish execute() uses in-process sandbox I/O that releases the GIL;
(b) threads share sys.path trivially, avoiding the pickle/spawn path-setup complexity.

Two entry points:
  parallel_ablation_necessity  — drop-in for ablation_necessity, same return type
  parallel_propose             — samples candidates eagerly then verifies concurrently
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from orkun.causal.counterfactual import CounterfactualResult
from orkun.goals.generator import GoalGen, ProposedGoal
from orkun.world.verifier import run_fresh


def parallel_ablation_necessity(task, calls, max_workers: int = 8) -> CounterfactualResult:
    """Parallel drop-in for ablation_necessity — N counterfactuals run concurrently."""
    base = run_fresh(task, calls)
    if not calls:
        return CounterfactualResult(base.passed, base.graded, [], [])

    necessity = [0.0] * len(calls)
    variants: list[dict] = [{}] * len(calls)

    def _cf(i: int):
        cf_calls = calls[:i] + calls[i + 1:]
        r = run_fresh(task, cf_calls)
        drop = max(0.0, base.graded - r.graded)
        return i, drop, {"action_idx": i, "passed": r.passed, "graded": r.graded}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as pool:
        for i, drop, v in pool.map(_cf, range(len(calls))):
            necessity[i] = drop
            variants[i] = v

    return CounterfactualResult(base.passed, base.graded, necessity, variants)


def parallel_propose(
    goal_gen: GoalGen,
    context: dict,
    n: int,
    *,
    max_workers: int = 8,
    oversample: int = 4,
) -> list[ProposedGoal]:
    """Parallel oracle verification for goal proposals.

    Samples n*oversample candidates eagerly (fast, no I/O — uses goal_gen.rng),
    then verifies them concurrently. Admits the first n that pass. Falls back to
    sequential GoalGen for any shortfall (guarantees exactly n goals or raises).
    """
    rng = goal_gen.rng
    candidates: list[tuple] = []
    for _ in range(n * oversample):
        family = rng.choice(goal_gen.families)
        difficulty = goal_gen._difficulty_for(context, family.name)
        task = family.sample(rng, difficulty)
        calls = family.oracle(task)
        candidates.append((task, calls, family.name, difficulty))

    admitted: list[ProposedGoal] = []

    def _verify(item: tuple):
        task, calls, fname, diff = item
        return run_fresh(task, calls).passed, task, calls, fname, diff

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for passed, task, calls, fname, diff in pool.map(_verify, candidates):
            if passed:
                admitted.append(ProposedGoal(task, fname, diff, calls))
            if len(admitted) >= n:
                break

    if len(admitted) < n:
        admitted += goal_gen.propose(context, n - len(admitted))

    return admitted[:n]
