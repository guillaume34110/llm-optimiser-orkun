from scripts.verifier import Task
from orkun.policy.backends.waaagh import WaaaghBackend
from orkun.policy.base import Policy
from orkun.policy.orkish_policy import OrkishPolicy, make_orkish_policy


def _hello_task() -> Task:
    return Task(
        id="hello",
        prompt="Create a file hello.txt containing exactly: WAAAGH",
        checks=[
            {"type": "file_equals", "path": "hello.txt", "content": "WAAAGH", "strip": True},
            {"type": "no_error"},
        ],
        clean=True,
    )


def test_policy_satisfies_protocol(tokenizer, tiny_net):
    pol = make_orkish_policy(tiny_net, tokenizer)
    assert isinstance(pol, Policy)  # structural check on the loop-facing seam


def test_gen_fn_returns_str(tokenizer, tiny_net):
    pol = OrkishPolicy(WaaaghBackend(tiny_net, tokenizer), temperature=1.0, max_new=12)
    t = _hello_task()
    assert isinstance(pol.gen_fn(t, 0), str)
    assert isinstance(pol.gen_fn(t, 1), str)


def test_rollout_with_scripted_oracle_verifies(tokenizer, tiny_net, known_tools):
    pol = OrkishPolicy(WaaaghBackend(tiny_net, tokenizer), known_tools=known_tools)
    t = _hello_task()
    # inject a scripted generator (mock allowed in a unit test) to exercise the
    # verified path deterministically, independent of the random net.
    pol.gen_fn = lambda task, attempt: 'CALL: write_file(path="hello.txt", content="WAAAGH")'
    res = pol.rollout(t, samples=2)
    assert res is not None
    assert res.task_id == "hello"
    assert res.n_calls == 1
    assert "<|assistant|>" in res.wire


def test_rollout_returns_none_when_unsolved(tokenizer, tiny_net, known_tools):
    pol = OrkishPolicy(WaaaghBackend(tiny_net, tokenizer), known_tools=known_tools)
    pol.gen_fn = lambda task, attempt: "no tool call here"
    assert pol.rollout(_hello_task(), samples=3) is None
