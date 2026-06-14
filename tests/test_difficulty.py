"""DifficultyTracker — per-family promotion/demotion driving GoalGen difficulty."""
import pytest

from orkun.goals.difficulty import DifficultyTracker


def test_starts_at_start_level():
    t = DifficultyTracker(start=0)
    assert t.level("arith") == 0
    t2 = DifficultyTracker(start=2)
    assert t2.level("arith") == 2


def test_promotes_after_full_successful_window():
    t = DifficultyTracker(start=0, window=4, promote_at=0.8)
    for _ in range(3):
        t.update("arith", True)
        assert t.level("arith") == 0  # window not full yet — no judgement
    t.update("arith", True)
    assert t.level("arith") == 1      # 4/4 ≥ 0.8 → promote


def test_demotes_after_full_failing_window():
    t = DifficultyTracker(start=2, window=4, demote_at=0.2)
    for _ in range(4):
        t.update("arith", False)
    assert t.level("arith") == 1


def test_window_resets_on_level_change_no_oscillation():
    # After a promotion the history is cleared: the old successes at level 0 must
    # not count toward judging level 1.
    t = DifficultyTracker(start=0, window=4)
    for _ in range(4):
        t.update("arith", True)
    assert t.level("arith") == 1
    # 3 failures at the new level: window not full again → still level 1
    for _ in range(3):
        t.update("arith", False)
    assert t.level("arith") == 1
    t.update("arith", False)
    assert t.level("arith") == 0      # only a FULL window at level 1 demotes


def test_clamped_at_bounds():
    t = DifficultyTracker(start=0, max_level=1, window=2)
    for _ in range(20):
        t.update("arith", True)
    assert t.level("arith") == 1      # never above max_level
    t2 = DifficultyTracker(start=0, window=2)
    for _ in range(20):
        t2.update("seq", False)
    assert t2.level("seq") == 0       # never below 0


def test_families_are_independent():
    t = DifficultyTracker(start=0, window=2)
    t.update("arith", True)
    t.update("arith", True)
    t.update("seq", False)
    assert t.level("arith") == 1
    assert t.level("seq") == 0


def test_context_shape_matches_goalgen():
    t = DifficultyTracker(start=0, window=2)
    t.update("arith", True)
    t.update("arith", True)
    ctx = t.context()
    assert "difficulty" in ctx
    assert ctx["difficulty"]["arith"] == 1


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        DifficultyTracker(promote_at=0.2, demote_at=0.8)
