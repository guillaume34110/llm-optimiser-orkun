import json
import random

from orkun.curiosity.curriculum import LPCurriculum
from orkun.curiosity.epistemic import EpistemicCuriosity
from orkun.curiosity.learning_progress import LearningProgress
from orkun.data.store import Store
from orkun.goals.families import ALL_FAMILIES
from orkun.goals.generator import GoalGen
from orkun.loop import kunnin_cycle


class _ScriptedPolicy:
    """Solves each family via its oracle, replayed as a CALL: string."""
    def __init__(self, families):
        self._oracle = {f.name: f for f in families}
        self.known_tools = None

    def rollout(self, task, samples=8):
        from scripts.self_play import play_task
        family = self._oracle[task.id.split("-")[0]]
        calls = family.oracle(task)
        text = "\n".join(
            f'CALL: {c.name}(' + ", ".join(f'{k}={v!r}' for k, v in c.args.items()) + ')'
            for c in calls
        )
        return play_task(task, lambda t, a: text, samples, known_tools=self.known_tools)


class _ScriptedWM:
    def score(self, prefix, continuation):
        return -2.0          # surprise = 2.0

    def encode(self, text):
        return [0.0]

    def predict(self, prefix, max_new=64, **kw):
        return "imagined"


class _ScriptedRewardHead:
    def predict(self, wire):
        return 0.5


def _phase_b_kwargs():
    return dict(
        world_model_on=True,
        wm=_ScriptedWM(),
        curiosity=EpistemicCuriosity(),
        reward_head=_ScriptedRewardHead(),
    )


def test_flag_on_stores_phase_b_fields(tmp_path, known_tools):
    rng = random.Random(0)
    gen = GoalGen(ALL_FAMILIES, seed=0)
    lp = LearningProgress(window=8)
    cur = LPCurriculum(epsilon=1.0, rng=rng)
    store = Store(tmp_path / "loot.jsonl")
    policy = _ScriptedPolicy(ALL_FAMILIES)
    policy.known_tools = known_tools

    report = kunnin_cycle(
        goal_gen=gen, lp=lp, curriculum=cur, policy=policy, store=store,
        families=[f.name for f in ALL_FAMILIES], n_goals=8, k=4, samples=4, rng=rng,
        **_phase_b_kwargs(),
    )
    assert report.n_verified >= 1
    rows = [json.loads(l) for l in (tmp_path / "loot.jsonl").read_text().splitlines()]
    for r in rows:
        assert "graded" in r and 0.0 <= r["graded"] <= 1.0
        assert "surprise" in r and abs(r["surprise"] - 2.0) < 1e-9
        assert "reward_hat" in r and r["reward_hat"] == 0.5
        assert "curiosity" in r and r["curiosity"] >= 0.0


def test_flag_on_corpus_packs_in_both_modes(tmp_path, known_tools):
    # load_examples_boosted: the cf boost is wire repetition, which the stock
    # packer dedups since Orkish 83d05ac.
    from orkun.train.cf_distill import build_corpus
    from orkun.train.pack import load_examples_boosted

    rng = random.Random(1)
    gen = GoalGen(ALL_FAMILIES, seed=1)
    lp = LearningProgress(window=8)
    cur = LPCurriculum(epsilon=1.0, rng=rng)
    store = Store(tmp_path / "loot.jsonl")
    policy = _ScriptedPolicy(ALL_FAMILIES)
    policy.known_tools = known_tools

    report = kunnin_cycle(
        goal_gen=gen, lp=lp, curriculum=cur, policy=policy, store=store,
        families=[f.name for f in ALL_FAMILIES], n_goals=8, k=4, samples=4, rng=rng,
        **_phase_b_kwargs(),
    )
    assert report.n_verified >= 1
    rows = store.sample(len(store))
    van = tmp_path / "vanilla.jsonl"
    cf = tmp_path / "cf.jsonl"
    n_van = build_corpus(rows, mode="vanilla", out_path=van, boost=4)
    n_cf = build_corpus(rows, mode="cf", out_path=cf, boost=4)
    assert len(load_examples_boosted([van])) == n_van
    assert len(load_examples_boosted([cf])) == n_cf


def test_flag_off_is_phase_a_identical(tmp_path, known_tools):
    # No Phase B keys when the flag is off, even if collaborators are not passed.
    rng = random.Random(2)
    gen = GoalGen(ALL_FAMILIES, seed=2)
    lp = LearningProgress(window=8)
    cur = LPCurriculum(epsilon=1.0, rng=rng)
    store = Store(tmp_path / "loot.jsonl")
    policy = _ScriptedPolicy(ALL_FAMILIES)
    policy.known_tools = known_tools

    report = kunnin_cycle(
        goal_gen=gen, lp=lp, curriculum=cur, policy=policy, store=store,
        families=[f.name for f in ALL_FAMILIES], n_goals=8, k=4, samples=4, rng=rng,
    )
    assert report.n_verified >= 1
    rows = [json.loads(l) for l in (tmp_path / "loot.jsonl").read_text().splitlines()]
    for r in rows:
        assert set(r.keys()) == {"task", "wire", "n_calls", "necessity", "source"}
