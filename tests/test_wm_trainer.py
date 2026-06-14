from orkun.world_model.trainer import WMTrainer
from orkun.world_model.waaagh_wm import WaaaghWorldModel


def test_update_lowers_surprise_on_trained_transition(tiny_net, tokenizer):
    wm = WaaaghWorldModel(tiny_net, tokenizer)
    prefix, result = "the file contains ", "WAAAGH the orcs win"
    before_surprise = -wm.score(prefix, result)
    trainer = WMTrainer(tiny_net, tokenizer)
    trainer.update([(prefix, result)], n_steps=5)
    after_surprise = -wm.score(prefix, result)
    assert after_surprise < before_surprise


def test_update_empty_transitions_is_noop(tiny_net, tokenizer):
    trainer = WMTrainer(tiny_net, tokenizer)
    loss = trainer.update([], n_steps=3)
    assert loss == 0.0
