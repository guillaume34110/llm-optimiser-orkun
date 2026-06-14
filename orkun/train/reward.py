"""Reward composition for RLVR — pure, testable scalar arithmetic.

r_total fuses the extrinsic graded verifier reward with the intrinsic curiosity
(scaled by beta). baseline is the running mean of past r_total (REINFORCE variance
reduction); advantage = r_total - baseline is what RLVR weights each action by.
"""
from __future__ import annotations


def r_total(graded: float, curiosity: float, beta: float) -> float:
    return graded + beta * curiosity


def baseline(history: list[float]) -> float:
    return sum(history) / len(history) if history else 0.0


def advantage(reward: float, base: float) -> float:
    return reward - base
