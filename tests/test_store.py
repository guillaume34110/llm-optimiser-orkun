import json

from orkun.data.store import Store


def test_add_and_persist(tmp_path):
    path = tmp_path / "store.jsonl"
    s = Store(path)
    s.add(task_id="t1", wire="<|bos|><|user|>x<|assistant|>done<|eos|>",
          n_calls=1, necessity=[1.0], source="self_play")
    s.add(task_id="t2", wire="<|bos|><|user|>y<|assistant|>done<|eos|>",
          n_calls=0, necessity=[], source="self_play")
    assert path.is_file()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["task"] == "t1"          # 'task' key = what pack_sft reads
    assert rows[0]["necessity"] == [1.0]
    assert rows[0]["wire"].startswith("<|bos|>")


def test_ready_for_update(tmp_path):
    s = Store(tmp_path / "s.jsonl")
    assert not s.ready_for_update(min_rows=2)
    s.add("a", "w", 0, [], "self_play")
    assert not s.ready_for_update(min_rows=2)
    s.add("b", "w", 0, [], "self_play")
    assert s.ready_for_update(min_rows=2)


def test_sample_returns_rows(tmp_path):
    s = Store(tmp_path / "s.jsonl")
    for i in range(5):
        s.add(f"t{i}", "w", 0, [], "self_play")
    got = s.sample(3)
    assert len(got) == 3
    assert all("wire" in r for r in got)


def test_reload_existing_loot_on_restart(tmp_path):
    # A restarted arm (crash, resumed Kaggle session) must keep sampling from its
    # full on-disk history, not restart from an empty in-memory list.
    path = tmp_path / "store.jsonl"
    s1 = Store(path)
    for i in range(3):
        s1.add(f"t{i}", "w", 0, [], "self_play")

    s2 = Store(path)                     # fresh process, same file
    assert len(s2) == 3
    assert {r["task"] for r in s2.sample(3)} == {"t0", "t1", "t2"}
    s2.add("t3", "w", 0, [], "self_play")
    assert len(s2) == 4
    # the file holds exactly the 4 rows — reload did not rewrite or duplicate
    assert len(path.read_text().splitlines()) == 4


def test_add_with_phase_b_fields(tmp_path):
    path = tmp_path / "store.jsonl"
    s = Store(path)
    s.add(task_id="t1", wire="<|bos|>w<|eos|>", n_calls=1, necessity=[1.0],
          source="self_play", graded=0.75, surprise=2.0, reward_hat=0.6, curiosity=0.3)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["graded"] == 0.75
    assert row["surprise"] == 2.0
    assert row["reward_hat"] == 0.6
    assert row["curiosity"] == 0.3


def test_add_without_phase_b_fields_is_phase_a_identical(tmp_path):
    # Flag-off path must produce rows with NO Phase B keys (byte-identical to Phase A).
    path = tmp_path / "store.jsonl"
    s = Store(path)
    s.add(task_id="t1", wire="<|bos|>w<|eos|>", n_calls=1, necessity=[1.0], source="self_play")
    row = json.loads(path.read_text().splitlines()[0])
    assert set(row.keys()) == {"task", "wire", "n_calls", "necessity", "source"}
