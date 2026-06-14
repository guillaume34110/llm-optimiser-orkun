"""Group path in kunnin_cycle: group-relative advantage + one batched update per cycle."""
import json
import random

from orkun.curiosity.curriculum import LPCurriculum
from orkun.curiosity.epistemic import EpistemicCuriosity
from orkun.curiosity.learning_progress import LearningProgress
from orkun.data.store import Store
from orkun.goals.families import ALL_FAMILIES
from orkun.goals.generator import GoalGen
from orkun.loop import kunnin_cycle
from orkun.policy.group_rollout import GroupResult, GroupRollout, SampleResult
from orkun.world.verifier import run_fresh


class _ScriptedGroup:
    """Returns `n_pass` oracle-solved samples + `n_fail` empty ones for each task."""
    def __init__(self, families, known_tools, n_pass=2, n_fail=2):
        self._fam = {f.name: f for f in families}
        self.known_tools = known_tools
        self.n_pass = n_pass
        self.n_fail = n_fail

    def _prompt_wire(self, task):
        return "<|bos|><|user|>" + task.prompt + "<|assistant|>"

    def rollout_group(self, task, samples=8):
        fam = self._fam[task.id.split("-")[0]]
        calls = fam.oracle(task)
        text = "\n".join(
            f"CALL: {c.name}(" + ", ".join(f"{k}={v!r}" for k, v in c.args.items()) + ")"
            for c in calls
        )
        out = []
        for _ in range(self.n_pass):
            r = run_fresh(task, calls)
            wire = GroupRollout._build_wire(task, text, r.steps)
            out.append(SampleResult(text, calls[: len(r.steps)], r.graded, r.passed, len(r.steps), wire))
        for _ in range(self.n_fail):
            out.append(SampleResult("nope", [], 0.0, False, 0, ""))
        return GroupResult(samples=out)


class _ScriptedWM:
    def score(self, prefix, continuation):
        return -2.0

    def encode(self, text):
        return [0.0]

    def predict(self, prefix, max_new=64, **kw):
        return "imagined"


class _CapWMTrainer:
    def __init__(self):
        self.batches = []

    def update(self, transitions):
        self.batches.append(list(transitions))


class _CapRLVR:
    def __init__(self):
        self.batches = []

    def update(self, samples, n_steps=1):
        self.batches.append(list(samples))


def _arith_only():
    return [f for f in ALL_FAMILIES if f.name == "arith"]


def test_group_path_batches_one_update_per_cycle(tmp_path, tokenizer, known_tools):
    rng = random.Random(0)
    gen = GoalGen(_arith_only(), seed=0)
    lp = LearningProgress(window=8)
    cur = LPCurriculum(epsilon=1.0, rng=rng)
    store = Store(tmp_path / "loot.jsonl")
    group = _ScriptedGroup(_arith_only(), known_tools, n_pass=2, n_fail=2)
    wm_trainer = _CapWMTrainer()
    rlvr = _CapRLVR()

    report = kunnin_cycle(
        goal_gen=gen, lp=lp, curriculum=cur, policy=None, store=store,
        families=["arith"], n_goals=8, k=1, samples=4, rng=rng,
        world_model_on=True, wm=_ScriptedWM(), curiosity=EpistemicCuriosity(),
        reward_head=None, wm_trainer=wm_trainer, rlvr=rlvr, tok=tokenizer,
        group=group,
    )

    assert report.n_verified == 1
    # ONE batched flush per cycle (not one per goal).
    assert len(rlvr.batches) == 1
    assert len(wm_trainer.batches) == 1
    # Both passing samples contribute a policy-gradient sample with positive credit.
    batch = rlvr.batches[0]
    assert len(batch) == 2
    for sample in batch:
        assert sample["advantage"] > 0.0       # passing beats the group mean (failures drag it down)
        assert sample["necessity"] > 0.0       # the single oracle call is causally necessary
        assert sample["prefix_ids"] and sample["action_ids"]


def test_group_path_profile_accumulates_phase_times(tmp_path, tokenizer, known_tools):
    rng = random.Random(0)
    gen = GoalGen(_arith_only(), seed=0)
    lp = LearningProgress(window=8)
    cur = LPCurriculum(epsilon=1.0, rng=rng)
    store = Store(tmp_path / "loot.jsonl")
    group = _ScriptedGroup(_arith_only(), known_tools, n_pass=2, n_fail=2)
    profile: dict = {}

    kunnin_cycle(
        goal_gen=gen, lp=lp, curriculum=cur, policy=None, store=store,
        families=["arith"], n_goals=8, k=1, samples=4, rng=rng,
        world_model_on=True, wm=_ScriptedWM(), curiosity=EpistemicCuriosity(),
        reward_head=None, wm_trainer=_CapWMTrainer(), rlvr=_CapRLVR(), tok=tokenizer,
        group=group, profile=profile,
    )

    assert set(profile) == {"propose", "rollout", "train"}
    assert all(v >= 0.0 for v in profile.values())
    assert profile["rollout"] > 0.0  # the group rollout phase ran


def test_group_path_stores_best_sample(tmp_path, tokenizer, known_tools):
    rng = random.Random(1)
    gen = GoalGen(_arith_only(), seed=1)
    lp = LearningProgress(window=8)
    cur = LPCurriculum(epsilon=1.0, rng=rng)
    store = Store(tmp_path / "loot.jsonl")
    group = _ScriptedGroup(_arith_only(), known_tools, n_pass=2, n_fail=2)

    kunnin_cycle(
        goal_gen=gen, lp=lp, curriculum=cur, policy=None, store=store,
        families=["arith"], n_goals=8, k=1, samples=4, rng=rng,
        world_model_on=True, wm=_ScriptedWM(), curiosity=EpistemicCuriosity(),
        reward_head=None, wm_trainer=_CapWMTrainer(), rlvr=_CapRLVR(), tok=tokenizer,
        group=group,
    )

    rows = [json.loads(l) for l in (tmp_path / "loot.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["graded"] == 1.0
    assert row["necessity"] and max(row["necessity"]) > 0.0
    assert "<|assistant|>" in row["wire"]
