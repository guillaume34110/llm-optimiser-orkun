"""Graded verifier reward, reusing Orkish's CHECKS evaluators.

The Orkish verifier answers a binary pass/fail and owns its sandbox. Orkun needs
two extra things: a *graded* score (fraction of checks satisfied) to feed the
learning-progress signal, and the ability to run in a *caller-provided* sandbox
so the counterfactual module can replay from a restored snapshot. We reuse the
exact CHECKS evaluators and `execute` — no check logic is reimplemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

from infer.executors import Sandbox, execute
from infer.monkey_wire import ToolCall
from scripts.verifier import CHECKS, StepResult, Task


@dataclass
class Reward:
    passed: bool
    graded: float                  # fraction of checks satisfied, in [0, 1]
    steps: list[StepResult]
    failed_checks: list[dict]


def _evaluate(task: Task, sb: Sandbox, steps: list[StepResult]) -> tuple[bool, list[dict]]:
    failed: list[dict] = []
    for spec in task.checks:
        fn = CHECKS.get(spec.get("type"))
        if fn is None:
            failed.append({**spec, "_reason": "unknown check type"})
            continue
        try:
            if not fn(spec, sb, steps):
                failed.append(spec)
        except Exception as e:  # a malformed spec must never crash scoring
            failed.append({**spec, "_reason": f"{type(e).__name__}: {e}"})
    return (not failed), failed


def run_in(task: Task, calls: list[ToolCall], sb: Sandbox, max_calls: int = 16) -> Reward:
    """Execute `calls` in the GIVEN sandbox, then score every task check."""
    steps: list[StepResult] = []
    for call in calls[:max_calls]:
        steps.append(StepResult(call=call, result=execute(call.name, call.args, sb)))
    passed, failed = _evaluate(task, sb, steps)
    n = len(task.checks)
    graded = (n - len(failed)) / n if n else 0.0
    return Reward(passed=passed, graded=graded, steps=steps, failed_checks=failed)


def run_fresh(task: Task, calls: list[ToolCall], max_calls: int = 16) -> Reward:
    """Create a fresh seeded sandbox (honouring task.seed/clean), score, then clean up."""
    sb = Sandbox.create(seed=not task.clean)
    try:
        for rel, content in task.seed.items():
            p = sb.resolve(rel)
            if p is None:
                return Reward(False, 0.0, [], [{"_reason": f"seed escapes sandbox: {rel!r}"}])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        return run_in(task, calls, sb, max_calls=max_calls)
    finally:
        sb.cleanup()
