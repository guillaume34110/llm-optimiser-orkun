# orkun/eval/done_b.py
"""Done-B verdict — compare measured Phase B metrics to configured thresholds.

`evaluate_done_b` is a pure verdict function: it takes the three already-computed scalar
values (WM held-out NLL from calibration.wm_nll, reward-head MAE from
calibration.reward_head_mae, and the sample-gain ratio from
sample_efficiency.run_sample_efficiency) and compares each to its threshold in the eval
config, producing a per-criterion pass/fail plus an aggregate AND. Checkpoint loading and
metric computation live with the caller (out of this module's scope). Thresholds are
sentinels in configs/eval.yaml, recalibrated at the first real-checkpoint run.
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class CriterionResult:
    name: str
    value: float
    threshold: float
    passed: bool


@dataclass
class DoneBReport:
    criteria: dict[str, CriterionResult]
    passed: bool                # AND of all criteria.passed


def load_eval_config(path: str = "configs/eval.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_done_b(cfg: dict, *, wm_nll_value: float, reward_head_mae_value: float,
                    sample_gain_ratio: float) -> DoneBReport:
    """Build the Done-B verdict. wm_nll/MAE are upper-bounded; sample_gain is lower-bounded."""
    criteria = {
        "wm_nll": CriterionResult(
            "wm_nll", wm_nll_value, cfg["wm_nll_max"],
            wm_nll_value <= cfg["wm_nll_max"]),
        "reward_head_mae": CriterionResult(
            "reward_head_mae", reward_head_mae_value, cfg["reward_head_mae_max"],
            reward_head_mae_value <= cfg["reward_head_mae_max"]),
        "sample_gain": CriterionResult(
            "sample_gain", sample_gain_ratio, cfg["sample_gain_min"],
            sample_gain_ratio >= cfg["sample_gain_min"]),
    }
    return DoneBReport(criteria=criteria, passed=all(c.passed for c in criteria.values()))
