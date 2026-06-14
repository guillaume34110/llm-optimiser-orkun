from infer.monkey_wire import ToolCall
from scripts.verifier import Task
from orkun.causal.counterfactual import CounterfactualResult, ablation_necessity, pruned_calls


def _hello_task() -> Task:
    return Task(
        id="hello",
        prompt="write WAAAGH to hello.txt",
        checks=[
            {"type": "file_equals", "path": "hello.txt", "content": "WAAAGH", "strip": True},
            {"type": "no_error"},
        ],
        clean=True,
    )


def test_single_necessary_action_scores_one():
    task = _hello_task()
    calls = [ToolCall("write_file", {"path": "hello.txt", "content": "WAAAGH\n"})]
    res = ablation_necessity(task, calls)
    assert isinstance(res, CounterfactualResult)
    assert res.base_passed is True
    assert len(res.necessity) == 1
    # ablating the only action → empty trajectory: file_equals fails but no_error
    # still passes (no steps → no error), so graded drops 1.0 → 0.5, a 0.5 drop.
    assert res.necessity[0] == 0.5


def test_redundant_action_scores_zero():
    task = _hello_task()
    calls = [
        ToolCall("py_run", {"code": "print('noise')"}),     # redundant
        ToolCall("write_file", {"path": "hello.txt", "content": "WAAAGH\n"}),  # necessary
    ]
    res = ablation_necessity(task, calls)
    assert res.base_passed is True
    assert res.necessity[0] == 0.0          # noise removable
    assert res.necessity[1] > 0.0           # write needed
    # pruning drops the redundant action but the result still passes
    pruned = pruned_calls(calls, res.necessity)
    assert pruned == [calls[1]]


def test_unsolved_trajectory_returns_base_failed():
    task = _hello_task()
    res = ablation_necessity(task, [ToolCall("py_run", {"code": "print(1)"})])
    assert res.base_passed is False
    assert res.necessity == [0.0]  # nothing was necessary to a failure
