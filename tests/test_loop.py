import random

from infer.monkey_wire import ToolCall
from orkun.curiosity.curriculum import LPCurriculum
from orkun.curiosity.learning_progress import LearningProgress
from orkun.data.store import Store
from orkun.goals.families import ALL_FAMILIES
from orkun.goals.generator import GoalGen
from orkun.loop import CycleReport, kunnin_cycle


class _ScriptedPolicy:
    """Solves write_file/fix_token/compute/json via each family's own oracle."""

    def __init__(self, families):
        self._oracle = {f.name: f for f in families}
        self.known_tools = None

    def rollout(self, task, samples=8):
        from scripts.self_play import play_task
        # find the family by id prefix, replay its oracle as a CALL: string
        fam_name = task.id.split("-")[0]
        family = self._oracle[fam_name]
        calls = family.oracle(task)
        text = "\n".join(f'CALL: {c.name}(' + ", ".join(f'{k}={v!r}' for k, v in c.args.items()) + ')'
                         for c in calls)
        return play_task(task, lambda t, a: text, samples, known_tools=self.known_tools)


def test_one_cycle_stores_verified_trajectories(tmp_path, known_tools):
    rng = random.Random(0)
    gen = GoalGen(ALL_FAMILIES, seed=0)
    lp = LearningProgress(window=8)
    cur = LPCurriculum(epsilon=1.0, rng=rng)  # epsilon=1 → explore all families
    store = Store(tmp_path / "loot.jsonl")
    policy = _ScriptedPolicy(ALL_FAMILIES)
    policy.known_tools = known_tools

    report = kunnin_cycle(
        goal_gen=gen, lp=lp, curriculum=cur, policy=policy, store=store,
        families=[f.name for f in ALL_FAMILIES], n_goals=8, k=4, samples=4, rng=rng,
    )
    assert isinstance(report, CycleReport)
    assert report.n_attempts >= 1
    assert report.n_verified >= 1
    assert len(store) == report.n_verified
    # every stored row carries a necessity vector from the counterfactual step
    assert all("necessity" in r for r in store.sample(len(store)))


def test_lp_updates_after_cycle(tmp_path, known_tools):
    rng = random.Random(1)
    gen = GoalGen(ALL_FAMILIES, seed=1)
    lp = LearningProgress(window=8)
    cur = LPCurriculum(epsilon=1.0, rng=rng)
    store = Store(tmp_path / "loot.jsonl")
    policy = _ScriptedPolicy(ALL_FAMILIES)
    policy.known_tools = known_tools
    kunnin_cycle(goal_gen=gen, lp=lp, curriculum=cur, policy=policy, store=store,
                 families=[f.name for f in ALL_FAMILIES], n_goals=8, k=4, samples=4, rng=rng)
    # at least one family now has history (LP no longer trivially empty everywhere)
    scores = lp.predict([f.name for f in ALL_FAMILIES])
    assert any(fam in lp._hist for fam in scores)


def test_loop_output_packs_in_both_modes(tmp_path, known_tools):
    # End-to-end contract: trajectories the loop actually stores (real self-play
    # wires that end on a ToolResult) must produce packer-valid corpora in BOTH
    # modes. This is the regression that guards the loop→store→corpus→pack_sft path.
    # load_examples_boosted (not the stock loader) because the cf boost lives in
    # wire repetition, which the stock packer dedups since Orkish 83d05ac.
    from orkun.train.cf_distill import build_corpus
    from orkun.train.pack import load_examples_boosted

    rng = random.Random(2)
    gen = GoalGen(ALL_FAMILIES, seed=2)
    lp = LearningProgress(window=8)
    cur = LPCurriculum(epsilon=1.0, rng=rng)
    store = Store(tmp_path / "loot.jsonl")
    policy = _ScriptedPolicy(ALL_FAMILIES)
    policy.known_tools = known_tools

    report = kunnin_cycle(goal_gen=gen, lp=lp, curriculum=cur, policy=policy, store=store,
                          families=[f.name for f in ALL_FAMILIES], n_goals=8, k=4, samples=4, rng=rng)
    assert report.n_verified >= 1

    rows = store.sample(len(store))
    van = tmp_path / "vanilla.jsonl"
    cf = tmp_path / "cf.jsonl"
    n_van = build_corpus(rows, mode="vanilla", out_path=van, boost=4)
    n_cf = build_corpus(rows, mode="cf", out_path=cf, boost=4)
    # the boosted packer accepts every wire from both arms, copies included
    assert len(load_examples_boosted([van])) == n_van
    assert len(load_examples_boosted([cf])) == n_cf


def test_curiosity_over_results_uses_shared_transitions():
    """_curiosity_over_results must return the same transitions as the shared helper
    and the mean surprise over them — proving the DRY refactor preserved behaviour."""
    from infer.monkey_wire import (
        Conversation, UserMsg, AssistantMsg, ToolResult, serialize,
    )
    from orkun.loop import _curiosity_over_results
    from orkun.eval.holdout import transitions_from_wire

    conv = Conversation(eos=True)
    conv.turns.append(UserMsg(text="do X"))
    conv.turns.append(AssistantMsg(text="CALL: a()"))
    conv.turns.append(ToolResult(name="a", payload_str="R1"))
    conv.turns.append(ToolResult(name="a", payload_str="R2"))
    wire = serialize(conv)

    class ScriptedWM:
        def score(self, prefix, continuation):
            return -2.0          # mean logprob/token; surprise = +2.0

    class ScriptedCuriosity:
        def surprise(self, wm, prefix, result):
            return -wm.score(prefix, result)

    mean_surprise, transitions = _curiosity_over_results(wire, ScriptedWM(), ScriptedCuriosity())
    assert transitions == transitions_from_wire(wire)
    assert abs(mean_surprise - 2.0) < 1e-9
