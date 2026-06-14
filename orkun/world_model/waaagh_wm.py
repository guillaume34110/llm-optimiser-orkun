"""WaaaghWorldModel — the WorldModelBackend over a (trained) WaaaghNet + tokenizer.

score: encode prefix+continuation, one forward, mean logprob of the continuation
span via `span_logprob` (no_grad). encode: a forward hook on the last block captures
the hidden state (B, stomach+T, dim); we pool the final real token (position -1) and
return it as floats — the hook means Orkish's waaagh.py is never edited. predict:
samples a continuation via the shared sampler (used by planning to imagine results).
Same net type as a trained Orkish checkpoint, so any trained LLM loads as the WM.
"""
from __future__ import annotations

import torch

from orkun.policy.sampler import sample
from orkun.world_model.scoring import span_logprob


class WaaaghWorldModel:
    def __init__(self, net, tok):
        self.net = net
        self.tok = tok

    def _device(self):
        return next(self.net.parameters()).device

    def score(self, prefix: str, continuation: str) -> float:
        p = self.tok.encode(prefix, add_bos=False, add_eos=False)
        c = self.tok.encode(continuation, add_bos=False, add_eos=False)
        if not p or not c:
            return 0.0
        with torch.no_grad():
            return float(span_logprob(self.net, p + c, len(c)))

    def encode(self, text: str) -> list[float]:
        ids = self.tok.encode(text, add_bos=False, add_eos=False)
        if not ids:
            return [0.0] * self.net.cfg.dim
        captured: dict[str, torch.Tensor] = {}

        def _hook(_module, _inputs, output):
            # WAAAGH 1.1 Block.forward returns (h, v_raw) — h is always element 0
            h = output[0] if isinstance(output, tuple) else output
            captured["h"] = h.detach()

        handle = self.net.blocks[-1].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                x = torch.tensor([ids], dtype=torch.long, device=self._device())
                self.net(x)
        finally:
            handle.remove()
        h_last = captured["h"][0, -1]            # final real token hidden, (dim,)
        return h_last.float().cpu().tolist()

    def predict(
        self,
        prefix: str,
        max_new: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int | None = None,
        stop_ids: frozenset[int] = frozenset(),
    ) -> str:
        ids = self.tok.encode(prefix, add_bos=False, add_eos=False)
        new_ids = sample(
            self.net, ids, max_new=max_new, temperature=temperature,
            top_p=top_p, stop_ids=stop_ids, seed=seed,
        )
        return self.tok.decode(new_ids, skip_special=True)
