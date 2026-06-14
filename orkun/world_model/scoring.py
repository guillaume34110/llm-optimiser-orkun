"""span_logprob — mean per-token logprob of a continuation span under a WaaaghNet.

`net(x)` returns logits (B, T, V) aligned to INPUT positions (stomach tokens are
sliced off inside the net), so the token at position p is predicted by logits[p-1].
The continuation is the LAST `cont_len` tokens of `ids`; we gather their logprobs
from the shifted prediction window and average. Returns a differentiable scalar so
RLVR and the WM trainer can backprop; callers that only need a number wrap it in
`torch.no_grad()` and call `float(...)`.
"""
from __future__ import annotations

import torch


def span_logprob(net, ids: list[int], cont_len: int) -> torch.Tensor:
    if cont_len < 1:
        raise ValueError("cont_len must be >= 1")
    if len(ids) <= cont_len:
        raise ValueError("need a non-empty prefix before the continuation span")
    device = next(net.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = net(x)[0]                                   # (T, V)
    logp = torch.log_softmax(logits.float(), dim=-1)
    T = len(ids)
    start = T - cont_len
    targets = torch.tensor(ids[start:T], dtype=torch.long, device=device)  # (cont_len,)
    pred = logp[start - 1 : T - 1]                       # (cont_len, V)
    tok_lp = pred.gather(1, targets.unsqueeze(1)).squeeze(1)               # (cont_len,)
    return tok_lp.mean()
