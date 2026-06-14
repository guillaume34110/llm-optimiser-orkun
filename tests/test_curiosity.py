import random

from orkun.curiosity.curriculum import LPCurriculum, RandomCurriculum
from orkun.curiosity.learning_progress import LearningProgress


def test_lp_zero_when_no_history():
    lp = LearningProgress(window=10)
    assert lp.predict(["a", "b"]) == {"a": 0.0, "b": 0.0}


def test_lp_high_while_improving_then_decays_when_mastered():
    lp = LearningProgress(window=8)
    # ramp: fail, fail, fail, fail, then succeed, succeed, succeed, succeed
    for s in [0, 0, 0, 0, 1, 1, 1, 1]:
        lp.update("fam", bool(s))
    improving = lp.predict(["fam"])["fam"]
    assert improving > 0.3  # recent half all-success vs older half all-fail
    # now saturate at success → recent ≈ older ≈ 1 → LP decays
    for _ in range(8):
        lp.update("fam", True)
    assert lp.predict(["fam"])["fam"] < improving


def test_lp_curriculum_prefers_high_lp_family():
    lp = LearningProgress(window=8)
    for s in [0, 0, 0, 0, 1, 1, 1, 1]:
        lp.update("hi", bool(s))
    for _ in range(8):
        lp.update("lo", True)  # mastered → LP≈0
    cur = LPCurriculum(epsilon=0.0, rng=random.Random(0))
    picked = cur.pick(["hi", "lo"], lp.predict(["hi", "lo"]), k=1)
    assert picked == ["hi"]


def test_random_curriculum_ignores_lp():
    cur = RandomCurriculum(rng=random.Random(0))
    picked = cur.pick(["a", "b", "c"], {"a": 9.0, "b": 0.0, "c": 0.0}, k=2)
    assert len(picked) == 2
    assert set(picked) <= {"a", "b", "c"}
