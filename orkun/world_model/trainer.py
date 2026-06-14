"""WMTrainer — online finetune of the world-model to predict real tool results.

Each transition is (prefix_text, result_text). The loss is the negative mean
logprob of the result span under the WM net (next-token CE on the observed result)
— minimising it raises the WM's probability of the real result, i.e. lowers future
surprise on that family, which is exactly the curiosity *progress* signal. A few
light steps per cycle. Reuses the Orkish Muon+AdamW primitives via `split_params`;
no Orkish file is edited. Starts from an already-trained checkpoint in production.
"""
from __future__ import annotations

import torch
from torch_impl.model.muon import Muon, split_params

from orkun.world_model.scoring import span_logprob


class WMTrainer:
    def __init__(self, net, tok, lr_muon: float = 0.02, lr_adam: float = 1e-3, anchor=None):
        self.net = net
        self.tok = tok
        self.anchor = anchor  # optional AnchorReg (L2-SP anti-forgetting); None -> un-anchored
        muon_params, adam_params = split_params(net)
        self.muon = Muon(muon_params, lr=lr_muon)
        self.adam = torch.optim.AdamW(adam_params, lr=lr_adam)

    def update(self, transitions: list[tuple[str, str]], n_steps: int = 1) -> float:
        last_loss = 0.0
        for _ in range(n_steps):
            self.muon.zero_grad()
            self.adam.zero_grad()
            loss = torch.zeros((), dtype=torch.float32)
            n = 0
            for prefix, result in transitions:
                p = self.tok.encode(prefix, add_bos=False, add_eos=False)
                c = self.tok.encode(result, add_bos=False, add_eos=False)
                if not p or not c:
                    continue
                loss = loss - span_logprob(self.net, p + c, len(c))  # maximise result logprob
                n += 1
            if n and loss.requires_grad:
                (loss / n).backward()
                self.muon.step()
                self.adam.step()
                if self.anchor is not None:
                    self.anchor.apply(self.net)  # proximal pull toward the reference checkpoint (anti-forgetting)
                self.net.normalize_weights()  # WAAAGH invariant: restore unit-norm geometry after each step
                last_loss = float(loss.detach() / n)
        return last_loss
