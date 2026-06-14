from orkun.planning.dreamer import Dreamer


class _ScriptedWM:
    def score(self, prefix, continuation):
        return -1.0

    def encode(self, text):
        return [0.0]

    def predict(self, prefix, max_new=64, **kw):
        return "imagined"


class _ScriptedPolicy:
    def __init__(self, texts, known_tools=None):
        self._texts = texts
        self.known_tools = known_tools

    def _prompt_wire(self, task):
        return "<|bos|><|user|>" + task.prompt + "<|assistant|>"

    def gen_fn(self, task, attempt):
        return self._texts[attempt % len(self._texts)]


def test_verification_gate_rejects_hallucinated_plan(known_tools):
    # Candidate 0 (ranked first by the reward-head) is a hallucination whose real
    # calls FAIL; candidate 1 is the true oracle solution. The gate must keep only
    # the verified one.
    from orkun.goals.families import ALL_FAMILIES
    from orkun.goals.generator import GoalGen

    goal = GoalGen(ALL_FAMILIES, seed=0).propose(context={}, n=8)[0]
    fam = {f.name: f for f in ALL_FAMILIES}[goal.family]
    good_calls = fam.oracle(goal.task)
    good_text = "\n".join(
        f'CALL: {c.name}(' + ", ".join(f'{k}={v!r}' for k, v in c.args.items()) + ')'
        for c in good_calls
    )
    hallucinated = 'CALL: read_file(path="nonexistent-file-xyz.txt")'

    class _RankHallucinationFirst:
        def predict(self, wire):
            return 2.0 if "nonexistent-file-xyz" in wire else 1.0

    policy = _ScriptedPolicy(texts=[hallucinated, good_text], known_tools=known_tools)
    dreamer = Dreamer(policy, _ScriptedWM(), _RankHallucinationFirst(), m=2)
    ranked = dreamer.dream(goal.task, m=2)
    assert "nonexistent-file-xyz" in ranked[0][1]      # hallucination ranked first
    kept = dreamer.execute_verified(goal.task, ranked, top_n=2)
    assert len(kept) == 1                              # only the verified plan survives
    assert kept[0].wire.startswith("<|bos|>")
