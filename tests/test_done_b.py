# tests/test_done_b.py
from orkun.eval.done_b import (
    CriterionResult, DoneBReport, evaluate_done_b, load_eval_config,
)

CFG = {"wm_nll_max": 5.0, "reward_head_mae_max": 0.25, "sample_gain_min": 1.1,
       "holdout_frac": 0.2, "holdout_seed": 0}


def test_all_criteria_pass():
    report = evaluate_done_b(CFG, wm_nll_value=3.0, reward_head_mae_value=0.1,
                             sample_gain_ratio=2.0)
    assert isinstance(report, DoneBReport)
    assert report.passed is True
    assert report.criteria["wm_nll"].passed is True
    assert report.criteria["sample_gain"].passed is True


def test_one_failing_criterion_fails_the_verdict():
    report = evaluate_done_b(CFG, wm_nll_value=9.0,        # 9.0 > 5.0 → fail
                             reward_head_mae_value=0.1, sample_gain_ratio=2.0)
    assert report.passed is False
    assert report.criteria["wm_nll"].passed is False
    assert report.criteria["reward_head_mae"].passed is True


def test_sample_gain_below_threshold_fails():
    report = evaluate_done_b(CFG, wm_nll_value=3.0, reward_head_mae_value=0.1,
                             sample_gain_ratio=1.0)        # 1.0 < 1.1 → fail
    assert report.criteria["sample_gain"].passed is False
    assert report.passed is False


def test_criterion_result_carries_value_and_threshold():
    report = evaluate_done_b(CFG, wm_nll_value=3.0, reward_head_mae_value=0.1,
                             sample_gain_ratio=2.0)
    c = report.criteria["wm_nll"]
    assert isinstance(c, CriterionResult)
    assert c.value == 3.0 and c.threshold == 5.0


def test_load_eval_config_reads_yaml():
    cfg = load_eval_config("configs/eval.yaml")
    assert cfg["wm_nll_max"] == 5.0
    assert cfg["sample_gain_min"] == 1.1
