import json

from infer.monkey_wire import (
    AssistantMsg, Conversation, ToolResult, UserMsg, parse, serialize,
)
from orkun.train.cf_distill import build_corpus


def _verified_wire(prompt: str, calls_text: str, results: list[tuple[str, str]]) -> str:
    """Mirror scripts.self_play._build_wire: user, assistant(all CALL:s), one
    ToolResult per executed call. NOTE: a real self-play wire ends on a ToolResult
    and carries NO final assistant report — the corpus builder must add one."""
    conv = Conversation(eos=True)
    conv.turns.append(UserMsg(text=prompt))
    conv.turns.append(AssistantMsg(text=calls_text))
    for name, payload in results:
        conv.turns.append(ToolResult(name=name, payload_str=payload))
    return serialize(conv)


def _row(task_id, necessity):
    wire = _verified_wire(
        "write WAAAGH to hello.txt",
        'CALL: write_file(path="hello.txt", content="WAAAGH")',
        [("write_file", '{"bytes_written": 7}')],
    )
    return {"task": task_id, "wire": wire, "n_calls": 1,
            "necessity": necessity, "source": "self_play"}


def test_vanilla_emits_one_row_per_trajectory(tmp_path):
    out = tmp_path / "vanilla.jsonl"
    n = build_corpus([_row("a", [1.0]), _row("b", [1.0])], mode="vanilla", out_path=out, boost=4)
    assert n == 2
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["wire"].startswith("<|bos|>") for r in rows)


def test_cf_repeats_high_necessity(tmp_path):
    out = tmp_path / "cf.jsonl"
    # necessity 1.0, boost 4 → 1 + round(4) = 5 copies
    n = build_corpus([_row("a", [1.0])], mode="cf", out_path=out, boost=4)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == 5
    assert len(rows) == 5


def test_both_arms_emit_packer_valid_wires(tmp_path):
    # The stock Orkish packer rejects any wire whose last turn has a CALL:.
    # Self-play wires end on a ToolResult, so BOTH arms must repair them.
    from scripts.pack_sft import load_examples

    van = tmp_path / "vanilla.jsonl"
    cf = tmp_path / "cf.jsonl"
    build_corpus([_row("a", [1.0])], mode="vanilla", out_path=van, boost=0)
    build_corpus([_row("a", [1.0])], mode="cf", out_path=cf, boost=0)
    assert len(load_examples([van])) == 1   # raises if missing final report
    assert len(load_examples([cf])) == 1


def test_cf_prunes_redundant_call_and_its_result(tmp_path):
    # Two calls, only the first necessary → cf keeps call 0 + its result, drops call 1.
    wire = _verified_wire(
        "write then touch",
        'CALL: write_file(path="a.txt", content="A")\n'
        'CALL: write_file(path="b.txt", content="B")',
        [("write_file", '{"bytes_written": 1}'), ("write_file", '{"bytes_written": 1}')],
    )
    row = {"task": "t", "wire": wire, "n_calls": 2, "necessity": [1.0, 0.0],
           "source": "self_play"}
    out = tmp_path / "cf.jsonl"
    build_corpus([row], mode="cf", out_path=out, boost=0)
    pruned = json.loads(out.read_text().splitlines()[0])["wire"]
    conv = parse(pruned)
    # exactly one tool call and one result survive, plus the synthetic final report
    call_turns = [t for t in conv.turns if isinstance(t, AssistantMsg) and t.tool_calls]
    results = [t for t in conv.turns if isinstance(t, ToolResult)]
    assert len(call_turns) == 1 and len(call_turns[0].tool_calls) == 1
    assert call_turns[0].tool_calls[0].args["path"] == "a.txt"
    assert len(results) == 1
    last = conv.turns[-1]
    assert isinstance(last, AssistantMsg) and not last.tool_calls


def test_cf_keeps_whole_wire_when_necessity_misaligned(tmp_path):
    # Defensive: if the necessity vector doesn't match the executed calls/results,
    # pruning must NOT corrupt the wire — it falls back to the full (repaired) one.
    from scripts.pack_sft import load_examples

    row = _row("a", [1.0, 1.0])   # 2 necessity entries vs 1 call / 1 result
    out = tmp_path / "cf.jsonl"
    build_corpus([row], mode="cf", out_path=out, boost=0)
    assert len(load_examples([out])) == 1
