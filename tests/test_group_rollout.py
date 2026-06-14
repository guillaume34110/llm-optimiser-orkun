"""GroupRollout: batched generation + per-sample verification, all Orkun-side."""
import random

from orkun.goals.families import ArithFamily
from orkun.policy.group_rollout import GroupRollout


def _oracle_call_text(family, task):
    calls = family.oracle(task)
    return "\n".join(
        f"CALL: {c.name}(" + ", ".join(f"{k}={v!r}" for k, v in c.args.items()) + ")"
        for c in calls
    )


def test_group_rollout_returns_n_samples(tiny_net, tokenizer, known_tools):
    """Real batched forward over a random net: shape + best contract hold (calls rarely parse)."""
    task = ArithFamily().sample(random.Random(0), difficulty=0)
    g = GroupRollout(net=tiny_net, tok=tokenizer, known_tools=known_tools, max_new=8)
    res = g.rollout_group(task, samples=3, seed=0)
    assert len(res.samples) == 3
    assert res.best is not None
    assert all(0.0 <= s.graded <= 1.0 for s in res.samples)


def test_group_rollout_verifies_each_sample(tokenizer, known_tools, monkeypatch):
    """Each sample is verified independently: half emit the oracle call (pass), half don't."""
    import orkun.policy.group_rollout as gr_mod

    fam = ArithFamily()
    task = fam.sample(random.Random(0), difficulty=0)
    good_ids = tokenizer.encode(_oracle_call_text(fam, task), add_bos=False, add_eos=False)
    bad_ids = tokenizer.encode("no tool call here", add_bos=False, add_eos=False)

    def fake_batch(net, prompt_ids, n, **kw):
        return [good_ids if i % 2 == 0 else bad_ids for i in range(n)]

    monkeypatch.setattr(gr_mod, "sample_batch", fake_batch)
    g = GroupRollout(net=None, tok=tokenizer, known_tools=known_tools)
    res = g.rollout_group(task, samples=4)

    assert len(res.samples) == 4
    assert len(res.passed) == 2
    assert res.best.passed
    for s in res.passed:
        assert s.n_calls >= 1
        assert s.wire and "<|assistant|>" in s.wire
    for s in res.samples:
        if not s.passed:
            assert s.wire == "" and s.n_calls == 0


def test_group_rollout_seed_is_deterministic(tiny_net, tokenizer, known_tools):
    task = ArithFamily().sample(random.Random(1), difficulty=0)
    g = GroupRollout(net=tiny_net, tok=tokenizer, known_tools=known_tools, max_new=8)
    a = g.rollout_group(task, samples=3, seed=42)
    b = g.rollout_group(task, samples=3, seed=42)
    assert [s.assistant_text for s in a.samples] == [s.assistant_text for s in b.samples]
