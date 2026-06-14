from infer.monkey_wire import Conversation, UserMsg, AssistantMsg, ToolResult, serialize

from orkun.eval.holdout import transitions_from_wire, split_by_task


def _wire(user, assistant, results):
    conv = Conversation(eos=True)
    conv.turns.append(UserMsg(text=user))
    conv.turns.append(AssistantMsg(text=assistant))
    for name, payload in results:
        conv.turns.append(ToolResult(name=name, payload_str=payload))
    return serialize(conv)


def test_transitions_grow_prefix_turn_by_turn():
    wire = _wire("do X", "CALL: a()", [("a", "R1"), ("a", "R2")])
    trans = transitions_from_wire(wire)
    assert len(trans) == 2
    # first transition: prefix = user + assistant text, result = R1
    p0, r0 = trans[0]
    assert r0 == "R1"
    assert "do X" in p0 and "CALL: a()" in p0
    # second transition: prefix has grown to include R1, result = R2
    p1, r1 = trans[1]
    assert r1 == "R2"
    assert p1.endswith("R1")


def test_transitions_empty_when_no_results():
    wire = _wire("do X", "CALL: a()", [])
    assert transitions_from_wire(wire) == []


def test_split_by_task_no_leak_and_deterministic():
    rows = [{"task": f"t{i}", "wire": "w", "n_calls": 1,
             "necessity": [1.0], "source": "self_play"} for i in range(100)]
    train, hold = split_by_task(rows, frac=0.2, seed=0)
    train_ids = {r["task"] for r in train}
    hold_ids = {r["task"] for r in hold}
    # partition: every row placed once, no task on both sides
    assert train_ids.isdisjoint(hold_ids)
    assert len(train) + len(hold) == 100
    # roughly frac held out (loose bound — hashing is not exactly uniform on 100 ids)
    assert 5 <= len(hold) <= 40
    # deterministic: same seed → same split
    train2, hold2 = split_by_task(rows, frac=0.2, seed=0)
    assert [r["task"] for r in hold] == [r["task"] for r in hold2]


def test_split_all_rows_of_a_task_stay_together():
    rows = [{"task": "shared", "wire": f"w{i}", "n_calls": 1,
             "necessity": [1.0], "source": "self_play"} for i in range(5)]
    rows += [{"task": "other", "wire": "w", "n_calls": 1,
              "necessity": [1.0], "source": "self_play"}]
    train, hold = split_by_task(rows, frac=0.5, seed=3)
    # the 5 "shared" rows must all be on the same side
    shared_sides = {("train" if r in train else "hold")
                    for r in rows if r["task"] == "shared"}
    assert len(shared_sides) == 1
