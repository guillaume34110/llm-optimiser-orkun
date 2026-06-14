"""Parallel sandbox pool — oracle verify + counterfactual ablation.

Three assertions:
  1. parallel_ablation_necessity returns the same CounterfactualResult as the
     sequential version (same necessity values, same variants).
  2. parallel_propose returns exactly n verified goals.
  3. kunnin_cycle with parallel_workers > 0 runs without error and returns a
     CycleReport (same shape as the sequential path).
"""
from __future__ import annotations

from orkun.causal.counterfactual import ablation_necessity
from orkun.goals.families import ALL_FAMILIES
from orkun.goals.generator import GoalGen
from orkun.parallel.pool import parallel_ablation_necessity, parallel_propose


def _get_oracle_goal(seed: int = 0):
    gen = GoalGen(ALL_FAMILIES, seed=seed)
    goals = gen.propose({}, n=1)
    return goals[0]


def test_parallel_ablation_matches_sequential():
    goal = _get_oracle_goal(seed=0)
    seq = ablation_necessity(goal.task, goal.oracle_calls)
    par = parallel_ablation_necessity(goal.task, goal.oracle_calls, max_workers=4)

    assert par.base_passed == seq.base_passed
    assert abs(par.base_graded - seq.base_graded) < 1e-6
    assert len(par.necessity) == len(seq.necessity)
    for s, p in zip(seq.necessity, par.necessity):
        assert abs(s - p) < 1e-6
    assert par.cf_variants == seq.cf_variants


def test_parallel_ablation_empty_calls():
    goal = _get_oracle_goal(seed=1)
    result = parallel_ablation_necessity(goal.task, [], max_workers=4)
    assert result.necessity == []
    assert result.cf_variants == []


def test_parallel_propose_returns_n_verified():
    gen = GoalGen(ALL_FAMILIES, seed=7)
    goals = parallel_propose(gen, context={}, n=4, max_workers=4, oversample=4)
    assert len(goals) == 4
    # each admitted goal must have passed oracle verification at propose time
    from orkun.world.verifier import run_fresh
    for g in goals:
        r = run_fresh(g.task, g.oracle_calls)
        assert r.passed, f"goal {g.task.id} from family {g.family} not actually verifiable"


def test_cycle_with_parallel_workers(tiny_net, tokenizer, known_tools):
    from orkun.curiosity.curriculum import LPCurriculum
    from orkun.curiosity.learning_progress import LearningProgress
    from orkun.data.store import Store
    from orkun.goals.families import ALL_FAMILIES
    from orkun.goals.generator import GoalGen
    from orkun.loop import kunnin_cycle
    from orkun.policy.orkish_policy import make_orkish_policy
    import random, tempfile, pathlib

    policy = make_orkish_policy(tiny_net, tokenizer, temperature=1.0, known_tools=known_tools)
    gen = GoalGen(ALL_FAMILIES, seed=0)
    lp = LearningProgress(window=4)
    families = [f.name for f in ALL_FAMILIES]
    rng = random.Random(0)
    curriculum = LPCurriculum(epsilon=1.0, rng=rng)

    with tempfile.TemporaryDirectory() as td:
        store = Store(pathlib.Path(td) / "loot.jsonl")
        rep = kunnin_cycle(
            gen, lp, curriculum, policy, store, families,
            n_goals=4, k=2, samples=2, rng=rng,
            parallel_workers=2,
        )

    assert rep.n_attempts >= 0
    assert isinstance(rep.picked_families, list)
