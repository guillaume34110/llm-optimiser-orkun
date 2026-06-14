from orkun.planning.lookahead import LookaheadPlanner


class _ScriptedPolicy:
    """gen_fn returns a distinct text per attempt; prompt wire is trivial."""
    def __init__(self, texts, known_tools=None):
        self._texts = texts
        self.known_tools = known_tools

    def _prompt_wire(self, task):
        return "<|bos|><|user|>" + task.prompt + "<|assistant|>"

    def gen_fn(self, task, attempt):
        return self._texts[attempt % len(self._texts)]


class _ScriptedWM:
    def score(self, prefix, continuation):
        return -1.0

    def encode(self, text):
        return [0.0]

    def predict(self, prefix, max_new=64, **kw):
        return "imagined-result"


class _ScriptedRewardHead:
    """Prefers any wire containing the token 'BEST'."""
    def predict(self, wire):
        return 1.0 if "BEST" in wire else 0.0


class _Task:
    id = "write_file-1"
    prompt = "do the thing"


def test_imagine_ranks_best_candidate_first():
    policy = _ScriptedPolicy(texts=["plan A meh", "plan BEST win"])
    planner = LookaheadPlanner(policy, _ScriptedWM(), _ScriptedRewardHead(), k=2, horizon=1)
    ranked = planner.imagine(_Task(), n=2)
    assert ranked[0][1] == "plan BEST win"
    assert ranked[0][0] >= ranked[1][0]


def test_rollout_returns_verified_result(known_tools):
    # Real verifier: the solving CALL must produce a verified trajectory.
    from orkun.goals.families import ALL_FAMILIES
    from orkun.goals.generator import GoalGen

    goal = GoalGen(ALL_FAMILIES, seed=0).propose(context={}, n=8)[0]
    fam = {f.name: f for f in ALL_FAMILIES}[goal.family]
    calls = fam.oracle(goal.task)
    text = "\n".join(
        f'CALL: {c.name}(' + ", ".join(f'{k}={v!r}' for k, v in c.args.items()) + ')'
        for c in calls
    )
    policy = _ScriptedPolicy(texts=[text], known_tools=known_tools)
    planner = LookaheadPlanner(policy, _ScriptedWM(), _ScriptedRewardHead(), k=1, horizon=2)
    result = planner.rollout(goal.task, samples=1)
    assert result is not None
    assert result.wire.startswith("<|bos|>")
