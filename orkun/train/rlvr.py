"""RLVRTrainer — real RLVR (REINFORCE + baseline) with causal credit assignment.

For each verified trajectory sample we recompute the action tokens' logprob WITH
gradient (`span_logprob` on the policy net), weight it by advantage = r_total -
baseline, and by Phase A `necessity` so causally-necessary actions get more credit
(necessity 0 -> no credit). loss = -Σ necessity·advantage·logπ(action). Optimised
with the Orkish Muon (2D Linear weights) + AdamW (everything else) primitives, via
`split_params` — no Orkish file is edited. Coexists with the cf_distill SFT path.

Each sample is a dict: {prefix_ids: list[int], action_ids: list[int],
advantage: float, necessity: float}. The loop builds these from a stored wire
(prefix = prompt up to the assistant turn; action = the assistant CALL tokens).
"""
from __future__ import annotations

import torch
from torch_impl.model.muon import Muon, split_params

from orkun.world_model.scoring import span_logprob


class RLVRTrainer:
    def __init__(self, net, tok, lr_muon: float = 0.02, lr_adam: float = 1e-3, anchor=None):
        self.net = net
        self.tok = tok
        self.anchor = anchor  # optional AnchorReg (L2-SP anti-forgetting); None -> un-anchored
        muon_params, adam_params = split_params(net)
        self.muon = Muon(muon_params, lr=lr_muon)
        self.adam = torch.optim.AdamW(adam_params, lr=lr_adam)

    def update(self, samples: list[dict], n_steps: int = 1) -> float:
        last_loss = 0.0
        for _ in range(n_steps):
            self.muon.zero_grad()
            self.adam.zero_grad()
            loss = torch.zeros((), dtype=torch.float32)
            for s in samples:
                weight = float(s["necessity"]) * float(s["advantage"])
                if weight == 0.0:
                    continue            # no causal credit -> contributes no gradient
                lp = span_logprob(self.net, s["prefix_ids"] + s["action_ids"], len(s["action_ids"]))
                loss = loss - weight * lp
            if loss.requires_grad:
                loss.backward()
                self.muon.step()
                self.adam.step()
                # anchor only on a real step (preserves the zero-necessity no-op: no causal
                # credit -> no step -> proximal pull never fires -> params untouched)
                if self.anchor is not None:
                    self.anchor.apply(self.net)  # proximal pull toward the reference checkpoint (anti-forgetting)
                self.net.normalize_weights()  # WAAAGH invariant: project weights back to unit-norm manifold after each step
            last_loss = float(loss)
        return last_loss
