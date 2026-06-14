import random

from orkun.eval.ablation import AblationArm, run_curriculum_ablation


def test_lp_curriculum_beats_random_on_skewed_learnability():
    # World model for the test: 4 families, only "learnable" improves with practice;
    # the other three are either trivially solved or impossible (LP stays ≈0).
    # An LP curriculum should concentrate practice on "learnable" and finish with a
    # higher mean success rate than random, which wastes rollouts on the others.
    families = ["learnable", "trivial", "hard1", "hard2"]

    def make_solve_fn():
        practice = {f: 0 for f in families}

        def solve_fn(family: str) -> bool:
            practice[family] += 1
            if family == "trivial":
                return True
            if family in ("hard1", "hard2"):
                return False
            # learnable: succeeds once it has had enough practice
            return practice[family] >= 6

        return solve_fn

    lp = run_curriculum_ablation("lp", families, make_solve_fn(), cycles=40, k=1, seed=0)
    rnd = run_curriculum_ablation("random", families, make_solve_fn(), cycles=40, k=1, seed=0)
    assert isinstance(lp, AblationArm)
    assert len(lp.success_curve) == 40
    # LP focuses practice on the only learnable family → higher final success rate
    assert lp.final_success_rate("learnable") > rnd.final_success_rate("learnable")


def test_curve_values_in_unit_interval():
    fams = ["a", "b"]
    arm = run_curriculum_ablation("lp", fams, lambda f: True, cycles=10, k=1, seed=1)
    assert all(0.0 <= v <= 1.0 for v in arm.success_curve)
