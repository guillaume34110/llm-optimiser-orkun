"""AnchorReg — proximal (L2-SP) anti-forgetting for Muon + AdamW online finetune.

Online RLVR / WM finetune update the shared `WaaaghNet` in place, drifting it away
from the pretrained checkpoint and forgetting the general competence that checkpoint
encodes. AnchorReg pulls weights back toward a reference snapshot taken right after
the checkpoint is loaded.

It does this in **parameter space**, not as a loss penalty. The WAAAGH stack trains
with Muon (orthogonalised momentum) + AdamW (RMS-normalised) — both are approximately
gradient-scale-invariant, so an L2-SP term added to the loss only tilts the step's
*direction* and gets normalised away; it does not pin the weights (and AdamW's
weight_decay drifts toward zero, not toward the reference). Instead, after each
optimiser step we apply an explicit proximal pull:

    θ ← θ + eta · F · (θ* − θ)

`eta` is the interpretable, optimiser-agnostic anchor strength: eta=0 leaves the step
untouched, eta=1 snaps straight back to the reference (frozen). The Fisher diagonal
`F` scales the per-parameter pull (EWC seam — keep eta·F ≤ 1 for stability); estimating
F from a representative batch is a later extension (the self-play loot is not the
pretraining distribution, so a Fisher faked from it would be worse than uniform F=1).

Usage: build the anchor from the freshly loaded net, hand it to the trainers. They call
`anchor.apply(net)` after the optimiser step and before `normalize_weights()`, **only on
steps that already take an optimiser step**, so the no-op contracts (empty transitions,
zero causal credit) are preserved and `anchor=None` stays byte-identical to the
un-anchored path.

Interaction with model soup (deliberate): the reference is NOT re-snapshotted after
`soup_sync`. Every arm anchors to the same warmup checkpoint, so the post-soup pull is
identical across arms (it shifts the soup mean toward the warmup, never tears the arms
apart) and total drift from the substrate stays bounded by eta. Re-anchoring to the
averaged weights would compound drift soup after soup.
"""
from __future__ import annotations

import torch


class AnchorReg:
    # WAAAGH 1.1 organ dials: per-token LERP modulation vectors, value-residual λ
    # and the stomach-write gate are the model's cheap adaptation capacity — pinning
    # them to the reference would defeat their purpose. The 2D writer projections
    # (stomach_writer.w{q,k,v}.weight) stay anchored like the rest of the backbone.
    WAAAGH11_FREE = ("w_mod_", ".lam", "stomach_writer.gate")

    def __init__(self, net, eta: float = 0.05, fisher: dict[str, torch.Tensor] | None = None,
                 exclude: tuple[str, ...] = ()):
        self.eta = float(eta)
        # detached reference snapshot of every trainable parameter (the starting point);
        # params matching `exclude` get no reference -> apply() leaves them free.
        self.ref = {name: p.detach().clone() for name, p in net.named_parameters()
                    if not any(s in name for s in exclude)}
        self.fisher = fisher  # optional per-param importance diagonal (EWC); None -> uniform F=1

    @torch.no_grad()
    def apply(self, net) -> None:
        for name, p in net.named_parameters():
            ref = self.ref.get(name)
            if ref is None or ref.shape != p.shape:
                continue  # param absent / reshaped since the snapshot -> not anchored
            pull = self.eta * (ref - p)
            if self.fisher is not None:
                f = self.fisher.get(name)
                if f is not None:
                    pull = f * pull
            p.add_(pull)
