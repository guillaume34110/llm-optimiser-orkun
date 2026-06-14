import random

from orkun.goals.families import ALL_FAMILIES
from orkun.goals.generator import GoalGen, ProposedGoal


def test_propose_returns_n_admitted_goals():
    gen = GoalGen(ALL_FAMILIES, seed=0)
    goals = gen.propose(context={}, n=8)
    assert len(goals) == 8
    assert all(isinstance(g, ProposedGoal) for g in goals)
    assert all(g.family in {f.name for f in ALL_FAMILIES} for g in goals)


def test_all_proposed_goals_are_solvable():
    from orkun.world.verifier import run_fresh

    gen = GoalGen(ALL_FAMILIES, seed=1)
    for g in gen.propose(context={}, n=12):
        r = run_fresh(g.task, g.oracle_calls)
        assert r.passed, f"admitted an unsolvable goal in family {g.family}"


def test_context_difficulty_is_honoured():
    gen = GoalGen(ALL_FAMILIES, seed=2)
    goals = gen.propose(context={"difficulty": {"compute": 0}}, n=20)
    compute_goals = [g for g in goals if g.family == "compute"]
    assert compute_goals  # at least one drawn
    assert all(g.difficulty == 0 for g in compute_goals)
