# orkun/eval/calibration.py
"""Pure WM-calibration metrics for Done-B.

`wm_nll` is the mean held-out NLL/token of real ToolResults under the world-model
(NLL = -score, since the WM's `score` is mean per-token logprob). `reward_head_mae` is
the mean absolute error of the reward-head's predicted reward vs the real `graded` score
on held-out rows. Both are pure: they take already-split held-out data plus the (real or
stub) wm / reward_head, and read nothing else. The sandbox's `graded` stays ground truth.
"""
from __future__ import annotations


def wm_nll(wm, transitions: list[tuple[str, str]]) -> float:
    """Mean NLL/token of real results under the WM. 0.0 for an empty list."""
    if not transitions:
        return 0.0
    total = sum(-wm.score(prefix, result) for prefix, result in transitions)
    return total / len(transitions)


def reward_head_mae(reward_head, rows: list[dict]) -> float:
    """Mean |predict(wire) - graded| over rows that carry a `graded` field.

    Rows without `graded` (e.g. Phase A rows) are skipped. 0.0 if none qualify.
    """
    errors = [abs(reward_head.predict(r["wire"]) - float(r["graded"]))
              for r in rows if r.get("graded") is not None]
    if not errors:
        return 0.0
    return sum(errors) / len(errors)
