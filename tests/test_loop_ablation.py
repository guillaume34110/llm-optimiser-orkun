# tests/test_loop_ablation.py
import random

from orkun.eval.ablation import AblationArm
from orkun.eval.loop_ablation import CycleMetrics, run_loop_ablation, coverage


def test_curiosity_on_arm_covers_more_families():
    families = ["f1", "f2", "f3"]

    def cycle_fn(arm_cfg, rng):
        fam = rng.choice(families)
        if arm_cfg.get("curiosity"):
            success = True                       # curiosity helps explore → solves any family
        else:
            success = (fam == "f1")              # no curiosity → only the easy family
        return CycleMetrics(success=success, progress=0.0, family=fam)

    arms = run_loop_ablation(
        {"on": {"curiosity": True}, "off": {"curiosity": False}},
        cycle_fn, cycles=60, families=families, seed=0,
    )
    assert isinstance(arms["on"], AblationArm)
    assert len(arms["on"].success_curve) == 60
    assert coverage(arms["on"], families) > coverage(arms["off"], families)


def test_wm_online_arm_shows_positive_progress():
    families = ["f1"]

    def cycle_fn(arm_cfg, rng):
        progress = 0.3 if arm_cfg.get("wm_online") else 0.0
        return CycleMetrics(success=True, progress=progress, family="f1")

    arms = run_loop_ablation(
        {"online": {"wm_online": True}, "frozen": {"wm_online": False}},
        cycle_fn, cycles=10, families=families, seed=0,
    )
    assert len(arms["online"].progress_curve) == 10
    assert sum(arms["online"].progress_curve) > sum(arms["frozen"].progress_curve)
    assert sum(arms["frozen"].progress_curve) == 0.0


def test_coverage_counts_families_with_any_success():
    arm = AblationArm(mode="x", success_curve=[],
                      per_family_success={"a": [0, 1], "b": [0, 0], "c": [1]})
    assert coverage(arm, ["a", "b", "c"]) == 2     # a and c solved at least once
