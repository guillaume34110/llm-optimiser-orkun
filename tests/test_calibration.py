# tests/test_calibration.py
from orkun.eval.calibration import wm_nll, reward_head_mae


class ScriptedWM:
    def __init__(self, score_value):
        self._v = score_value
    def score(self, prefix, continuation):
        return self._v


class ScriptedHead:
    def __init__(self, value):
        self._v = value
    def predict(self, wire):
        return self._v


def test_wm_nll_is_mean_negative_score():
    wm = ScriptedWM(-1.5)            # NLL/token = +1.5 for every transition
    transitions = [("p1", "r1"), ("p2", "r2"), ("p3", "r3")]
    assert abs(wm_nll(wm, transitions) - 1.5) < 1e-9


def test_wm_nll_empty_is_zero():
    assert wm_nll(ScriptedWM(-1.5), []) == 0.0


def test_reward_head_mae_over_graded_rows():
    head = ScriptedHead(0.8)
    rows = [
        {"wire": "w1", "graded": 1.0},   # |0.8 - 1.0| = 0.2
        {"wire": "w2", "graded": 0.5},   # |0.8 - 0.5| = 0.3
    ]
    assert abs(reward_head_mae(head, rows) - 0.25) < 1e-9


def test_reward_head_mae_skips_rows_without_graded():
    head = ScriptedHead(0.8)
    rows = [
        {"wire": "w1", "graded": 1.0},   # |0.8 - 1.0| = 0.2
        {"wire": "w2"},                  # no graded → skipped
    ]
    assert abs(reward_head_mae(head, rows) - 0.2) < 1e-9


def test_reward_head_mae_no_graded_rows_is_zero():
    assert reward_head_mae(ScriptedHead(0.8), [{"wire": "w"}]) == 0.0
