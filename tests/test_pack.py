"""load_examples_boosted — repetition boost survives the stock packer's dedup."""
import json

from infer.monkey_wire import AssistantMsg, Conversation, ToolResult, UserMsg, serialize
from orkun.train.cf_distill import build_corpus
from orkun.train.pack import load_examples_boosted


def _verified_wire(prompt: str, calls_text: str, results: list[tuple[str, str]]) -> str:
    conv = Conversation(eos=True)
    conv.turns.append(UserMsg(text=prompt))
    conv.turns.append(AssistantMsg(text=calls_text))
    for name, payload in results:
        conv.turns.append(ToolResult(name=name, payload_str=payload))
    return serialize(conv)


def _row(task_id: str, necessity: list[float]) -> dict:
    wire = _verified_wire(
        f"write WAAAGH for {task_id}",
        f'CALL: write_file(path="{task_id}.txt", content="WAAAGH")',
        [("write_file", '{"bytes_written": 7}')],
    )
    return {"task": task_id, "wire": wire, "n_calls": 1,
            "necessity": necessity, "source": "self_play"}


def test_boost_copies_are_re_expanded(tmp_path):
    # necessity 1.0, boost 4 → 5 identical copies; the stock loader dedups them,
    # the boosted loader must restore all 5.
    from scripts.pack_sft import load_examples

    out = tmp_path / "cf.jsonl"
    n = build_corpus([_row("a", [1.0])], mode="cf", out_path=out, boost=4)
    assert n == 5
    assert len(load_examples([out])) == 1          # stock loader dedups (the regression)
    assert len(load_examples_boosted([out])) == 5  # boost restored


def test_unique_wires_pass_through_unchanged(tmp_path):
    out = tmp_path / "van.jsonl"
    n = build_corpus([_row("a", [1.0]), _row("b", [0.5])],
                     mode="vanilla", out_path=out, boost=4)
    assert n == 2
    exs = load_examples_boosted([out])
    assert len(exs) == 2
    assert len({ex["wire"] for ex in exs}) == 2


def test_filtered_wires_are_not_re_expanded(tmp_path):
    # A wire the stock loader REJECTS (ends on a ToolResult, no final report) must
    # stay rejected — re-expansion only compensates the dedup, never the filters.
    bad_wire = _verified_wire("x", 'CALL: write_file(path="x", content="y")',
                              [("write_file", "{}")])  # ends on ToolResult → rejected
    good = tmp_path / "good.jsonl"
    build_corpus([_row("a", [1.0])], mode="cf", out_path=good, boost=4)  # 5 copies
    with good.open("a") as f:
        for _ in range(3):
            f.write(json.dumps({"task": "bad", "wire": bad_wire}) + "\n")
    exs = load_examples_boosted([good])
    assert len(exs) == 5                          # bad ×3 filtered, never re-expanded
    assert all(ex["task"] == "a" for ex in exs)


def test_mixed_corpus_preserves_per_wire_multiplicity(tmp_path):
    # boost applies per row: nec 1.0 → 5 copies, nec 0.0 → 1 copy.
    out = tmp_path / "cf.jsonl"
    n = build_corpus([_row("hot", [1.0]), _row("cold", [0.0])],
                     mode="cf", out_path=out, boost=4)
    assert n == 6
    exs = load_examples_boosted([out])
    assert len(exs) == 6
    by_task = {}
    for ex in exs:
        by_task[ex["task"]] = by_task.get(ex["task"], 0) + 1
    assert by_task == {"hot": 5, "cold": 1}
