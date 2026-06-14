"""WaaaghBackend — the local-torch LMBackend over a WaaaghNet + Orkish tokenizer.

Encodes the prompt, samples token ids (orkun.policy.sampler), decodes back to text.
Stop strings that the tokenizer knows as specials (e.g. "<|eos|>", "<|user|>") are
mapped to stop token ids so generation halts cleanly; any stop string is ALSO cut
from the decoded text as a safety net (handles multi-token / non-special stops).
This is the swap point: a different LLM is a different LMBackend, not a change here.
"""
from __future__ import annotations

import re

from orkun.policy.sampler import sample

_SPECIAL_RE = re.compile(r"<\|([a-z_]+)\|>")


class WaaaghBackend:
    def __init__(self, net, tokenizer):
        self.net = net
        self.tok = tokenizer

    def _stop_ids(self, stop_strings: list[str]) -> set[int]:
        ids: set[int] = set()
        for s in stop_strings:
            m = _SPECIAL_RE.fullmatch(s)
            if m and m.group(1) in self.tok.specials:
                ids.add(self.tok.specials[m.group(1)])
        return ids

    def generate(
        self,
        prompt: str,
        *,
        max_new: int,
        temperature: float,
        top_p: float,
        stop_strings: list[str],
        seed: int | None,
    ) -> str:
        prompt_ids = self.tok.encode(prompt, add_bos=False, add_eos=False)
        new_ids = sample(
            self.net,
            prompt_ids,
            max_new=max_new,
            temperature=temperature,
            top_p=top_p,
            stop_ids=self._stop_ids(stop_strings),
            seed=seed,
        )
        text = self.tok.decode(new_ids, skip_special=True)
        # safety net: cut at any stop string surviving in the decoded text
        for s in stop_strings:
            i = text.find(s)
            if i != -1:
                text = text[:i]
        return text
