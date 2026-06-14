from infer.executors import Sandbox
from infer.monkey_wire import ToolCall
from scripts.verifier import Task
from orkun.world.verifier import Reward, run_fresh, run_in


def _hello_task() -> Task:
    return Task(
        id="hello",
        prompt="write WAAAGH to hello.txt",
        checks=[
            {"type": "file_equals", "path": "hello.txt", "content": "WAAAGH", "strip": True},
            {"type": "no_error"},
        ],
    )


def test_run_fresh_passes_with_correct_calls():
    task = _hello_task()
    calls = [ToolCall("write_file", {"path": "hello.txt", "content": "WAAAGH\n"})]
    r = run_fresh(task, calls)
    assert isinstance(r, Reward)
    assert r.passed is True
    assert r.graded == 1.0
    assert r.failed_checks == []


def test_run_fresh_graded_partial_on_partial_failure():
    # empty trajectory: file_equals fails (no file), but no_error passes → 1/2 = 0.5
    task = _hello_task()
    r = run_fresh(task, [])
    assert r.passed is False
    assert r.graded == 0.5


def test_run_in_uses_caller_sandbox():
    task = _hello_task()
    sb = Sandbox.create(seed=True)
    try:
        calls = [ToolCall("write_file", {"path": "hello.txt", "content": "WAAAGH\n"})]
        r = run_in(task, calls, sb)
        assert r.passed is True
        # the caller's sandbox now holds the file (proves no fresh sandbox was made)
        assert (sb.root / "hello.txt").is_file()
    finally:
        sb.cleanup()
