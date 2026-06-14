"""RewardHead — predicts graded reward from a WM encoding, to rank imagined plans.

A small MLP encode-dim -> 1 with a sigmoid output in [0, 1]. `fit` regresses the
store's (wire -> graded) pairs with MSE; `predict` scores a (real or imagined) wire.
Planning uses it to rank candidates cheaply before the sandbox verifies the best
ones. Calibration (MAE vs real graded on held-out) is a Done-B exit criterion.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RewardHead:
    def __init__(self, wm, hidden: int = 64, lr: float = 1e-2, seed: int = 0):
        torch.manual_seed(seed)
        self.wm = wm
        dim = wm.net.cfg.dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _features(self, wire: str) -> torch.Tensor:
        return torch.tensor(self.wm.encode(wire), dtype=torch.float32)

    def predict(self, wire: str) -> float:
        with torch.no_grad():
            return float(torch.sigmoid(self.net(self._features(wire))))

    def fit(self, rows: list[dict], epochs: int = 100) -> None:
        if not rows:
            return
        xs = torch.stack([self._features(r["wire"]) for r in rows])
        ys = torch.tensor([[float(r["graded"])] for r in rows], dtype=torch.float32)
        for _ in range(epochs):
            self.opt.zero_grad()
            pred = torch.sigmoid(self.net(xs))
            loss = ((pred - ys) ** 2).mean()
            loss.backward()
            self.opt.step()
