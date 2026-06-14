"""AnchorReg — proximal (L2-SP) anti-forgetting regulariser.

Unit tests pin the parameter-space pull (no-op at the reference, linear toward the
reference by `eta`, eta=1 snaps back exactly, Fisher scales the pull). The behavioural
test drives the real WMTrainer twice from the same init — once free, once anchored —
and asserts the anchored net stays closer to the reference checkpoint, i.e. it forgets
less of where it started. (An earlier L2-SP-as-loss-penalty design FAILED this test:
Muon/AdamW normalise the gradient magnitude, so a loss penalty only tilts direction and
gets normalised away — hence the parameter-space pull.)
"""
import copy

import torch

from orkun.train.anchor import AnchorReg
from orkun.world_model.trainer import WMTrainer


def _sq_dist_to_ref(net, ref: dict) -> float:
    return sum(float(((p - ref[n]) ** 2).sum()) for n, p in net.named_parameters())


def test_apply_is_noop_at_reference(tiny_net):
    anchor = AnchorReg(tiny_net, eta=0.5)
    snapshot = [p.detach().clone() for p in tiny_net.parameters()]
    anchor.apply(tiny_net)  # weights still at ref -> (ref - p) == 0 -> no movement
    for before, after in zip(snapshot, tiny_net.parameters()):
        assert torch.equal(before, after)


def test_apply_pulls_fraction_eta_toward_reference(tiny_net):
    anchor = AnchorReg(tiny_net, eta=0.1)
    name, p = next(iter(tiny_net.named_parameters()))
    ref = anchor.ref[name].clone()
    with torch.no_grad():
        p.add_(0.5)  # deviation = +0.5
    anchor.apply(tiny_net)
    # θ ← θ + eta·(θ* − θ): new deviation = 0.5 - 0.1*0.5 = 0.45
    assert torch.allclose(p - ref, torch.full_like(p, 0.45), atol=1e-5)


def test_eta_one_restores_reference(tiny_net):
    anchor = AnchorReg(tiny_net, eta=1.0)
    name, p = next(iter(tiny_net.named_parameters()))
    ref = anchor.ref[name].clone()
    with torch.no_grad():
        p.add_(0.7)
    anchor.apply(tiny_net)
    assert torch.allclose(p, ref, atol=1e-6)


def test_fisher_scales_pull(tiny_net):
    name, p = next(iter(tiny_net.named_parameters()))
    fisher = {name: torch.full_like(p, 0.5)}
    anchor = AnchorReg(tiny_net, eta=0.2, fisher=fisher)
    ref = anchor.ref[name].clone()
    with torch.no_grad():
        p.add_(1.0)
    anchor.apply(tiny_net)
    # effective pull fraction = eta*F = 0.2*0.5 = 0.1 -> new deviation = 1.0 - 0.1 = 0.9
    assert torch.allclose(p - ref, torch.full_like(p, 0.9), atol=1e-5)


def test_anchor_keeps_weights_closer_to_reference(tiny_net, tokenizer):
    ref = {n: p.detach().clone() for n, p in tiny_net.named_parameters()}
    transitions = [("the file contains ", "WAAAGH the orcs win")]

    free_net = copy.deepcopy(tiny_net)
    anchored_net = copy.deepcopy(tiny_net)

    WMTrainer(free_net, tokenizer).update(transitions, n_steps=20)
    anchor = AnchorReg(anchored_net, eta=0.2)
    WMTrainer(anchored_net, tokenizer, anchor=anchor).update(transitions, n_steps=20)

    assert _sq_dist_to_ref(anchored_net, ref) < _sq_dist_to_ref(free_net, ref)


def test_exclude_leaves_params_free(tiny_net):
    params = dict(tiny_net.named_parameters())
    name = next(iter(params))
    anchor = AnchorReg(tiny_net, eta=1.0, exclude=(name,))
    assert name not in anchor.ref
    p = params[name]
    with torch.no_grad():
        p.add_(0.7)
    drifted = p.detach().clone()
    anchor.apply(tiny_net)
    # excluded param untouched even at eta=1 (full snap-back for the others)
    assert torch.equal(p, drifted)
