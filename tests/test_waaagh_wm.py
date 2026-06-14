from orkun.world_model.backend import WorldModelBackend


class _ScriptedWM:
    def score(self, prefix: str, continuation: str) -> float:
        return -1.5

    def encode(self, text: str) -> list[float]:
        return [0.0, 1.0, 2.0]


def test_protocol_is_structural():
    wm = _ScriptedWM()
    assert isinstance(wm, WorldModelBackend)


def test_protocol_rejects_missing_method():
    class _Half:
        def score(self, prefix, continuation):
            return 0.0
    assert not isinstance(_Half(), WorldModelBackend)


def test_score_matches_manual(tiny_net, tokenizer):
    import torch
    from orkun.world_model.scoring import span_logprob
    from orkun.world_model.waaagh_wm import WaaaghWorldModel

    wm = WaaaghWorldModel(tiny_net, tokenizer)
    prefix, cont = "hello ", "world"
    got = wm.score(prefix, cont)
    p = tokenizer.encode(prefix)
    c = tokenizer.encode(cont)
    ref = float(span_logprob(tiny_net, p + c, len(c)))
    assert abs(got - ref) < 1e-4
    assert got <= 0.0  # mean logprob is non-positive


def test_encode_dim_and_hook_removed(tiny_net, tokenizer):
    from orkun.world_model.waaagh_wm import WaaaghWorldModel

    wm = WaaaghWorldModel(tiny_net, tokenizer)
    vec = wm.encode("hello world")
    assert len(vec) == tiny_net.cfg.dim
    assert all(isinstance(x, float) for x in vec)
    # the forward hook must be removed after encode (no leak across calls)
    assert len(tiny_net.blocks[-1]._forward_hooks) == 0


def test_predict_returns_text(tiny_net, tokenizer):
    from orkun.world_model.waaagh_wm import WaaaghWorldModel

    wm = WaaaghWorldModel(tiny_net, tokenizer)
    out = wm.predict("the result is ", max_new=8, seed=0)
    assert isinstance(out, str)
