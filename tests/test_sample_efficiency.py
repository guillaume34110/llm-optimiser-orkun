# tests/test_sample_efficiency.py
from orkun.eval.sample_efficiency import (
    AttemptResult, SampleEfficiencyReport, run_sample_efficiency,
)


class _Task:
    def __init__(self, tid):
        self.id = tid


def test_gain_ratio_when_planning_uses_fewer_executions():
    tasks = [_Task("t1"), _Task("t2")]

    def attempt_fn(task, mode):
        # planning solves in 2 real executions; reactive takes 6
        return AttemptResult(solved=True, n_real_exec=2 if mode == "planned" else 6)

    report = run_sample_efficiency(tasks, attempt_fn, seed=0)
    assert isinstance(report, SampleEfficiencyReport)
    assert report.reactive_mean_exec == 6.0
    assert report.planned_mean_exec == 2.0
    assert report.gain_ratio == 3.0
    assert report.per_task["t1"] == (6, 2)


def test_unsolved_attempt_counts_its_full_budget():
    tasks = [_Task("t1")]

    def attempt_fn(task, mode):
        if mode == "reactive":
            return AttemptResult(solved=False, n_real_exec=8)   # burned full budget, failed
        return AttemptResult(solved=True, n_real_exec=2)

    report = run_sample_efficiency(tasks, attempt_fn, seed=0)
    assert report.reactive_mean_exec == 8.0     # failure still counts its executions
    assert report.planned_mean_exec == 2.0
    assert report.gain_ratio == 4.0


def test_gain_ratio_divide_by_zero_guards():
    tasks = [_Task("t1")]

    def attempt_fn(task, mode):
        return AttemptResult(solved=True, n_real_exec=0)        # both arms zero

    report = run_sample_efficiency(tasks, attempt_fn, seed=0)
    assert report.gain_ratio == 1.0            # both zero → neutral
