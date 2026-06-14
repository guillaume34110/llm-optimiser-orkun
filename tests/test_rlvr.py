import torch

from orkun.train.rlvr import RLVRTrainer
from orkun.world_model.scoring import span_logprob


def _action_lp(net, prefix_ids, action_ids):
    with torch.no_grad():
        return float(span_logprob(net, prefix_ids + action_ids, len(action_ids)))


def test_positive_advantage_increases_action_logprob(tiny_net, tokenizer):
    prefix_ids = tokenizer.encode("the orc says ")
    action_ids = tokenizer.encode("WAAAGH")
    before = _action_lp(tiny_net, prefix_ids, action_ids)
    trainer = RLVRTrainer(tiny_net, tokenizer)
    trainer.update(
        [{"prefix_ids": prefix_ids, "action_ids": action_ids, "advantage": 1.0, "necessity": 1.0}],
        n_steps=5,
    )
    after = _action_lp(tiny_net, prefix_ids, action_ids)
    assert after > before


def test_negative_advantage_decreases_action_logprob(tiny_net, tokenizer):
    prefix_ids = tokenizer.encode("the orc says ")
    action_ids = tokenizer.encode("WAAAGH")
    before = _action_lp(tiny_net, prefix_ids, action_ids)
    trainer = RLVRTrainer(tiny_net, tokenizer)
    trainer.update(
        [{"prefix_ids": prefix_ids, "action_ids": action_ids, "advantage": -1.0, "necessity": 1.0}],
        n_steps=5,
    )
    after = _action_lp(tiny_net, prefix_ids, action_ids)
    assert after < before


def test_zero_necessity_gives_no_credit(tiny_net, tokenizer):
    prefix_ids = tokenizer.encode("the orc says ")
    action_ids = tokenizer.encode("WAAAGH")
    snapshot = [p.detach().clone() for p in tiny_net.parameters()]
    trainer = RLVRTrainer(tiny_net, tokenizer)
    trainer.update(
        [{"prefix_ids": prefix_ids, "action_ids": action_ids, "advantage": 5.0, "necessity": 0.0}],
        n_steps=5,
    )
    # necessity 0 -> no term contributes -> no optimiser step -> params untouched
    for before, after in zip(snapshot, tiny_net.parameters()):
        assert torch.equal(before, after)
