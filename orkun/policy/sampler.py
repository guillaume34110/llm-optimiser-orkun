"""Autoregressive sampler over WaaaghNet — Orkish ships no inference path, so this is it.

KV-cached decode: `net.prefill` runs the prompt once (capturing the per-layer
[stomach ++ seq] kv caches), then `net.decode_step` advances one token at a time,
attending the new query against the cache instead of re-running the whole prefix.
This turns generation from O(n²) recompute into O(n) — the dominant per-cycle GPU
cost on the T4. Parity with the full forward is gated by Orkish tests/test_kv_decode.py
(cached logits match forward() to <1e-4, identical greedy argmax at every step).

Two more GPU optimisations, both no-ops on CPU so the loop stays deterministic in tests:

  * Autocast — the forward runs under torch.autocast on CUDA (bf16 where the GPU has
    bf16 tensor cores, else fp16). T4 is Turing (sm_75) → no bf16 tensor cores → fp16,
    which is where its tensor-core throughput lives. The model is autocast-safe by
    design (QK-norm keeps its fp32 Scale, `_attend` casts q,k back to v's dtype).
  * On-device sampling — logits, softmax, top-p and multinomial all stay on the model's
    device; only the single chosen token id crosses to host. This removes the per-token
    full-vocab device→host copy that previously stalled the GPU pipeline every step.

Supports greedy (temperature==0), temperature scaling, and nucleus (top-p) sampling.
Stops on any id in `stop_ids` (the stop token is NOT appended) or at `max_new`.
"""
from __future__ import annotations

import contextlib

import torch


def _autocast_ctx(device: torch.device):
    """Autocast on CUDA (bf16 if the GPU supports it, else fp16); no-op elsewhere."""
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[0] = True  # always keep the top token
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = sorted_probs[keep]
    total = filtered.sum()
    if total <= 0:
        return probs
    return filtered / total


def _top_p_filter_batch(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Row-wise nucleus filter for a (B, V) probability tensor.

    Same semantics as `_top_p_filter` applied independently per row: keep the
    smallest prefix of the sorted mass whose cumulative probability is <= top_p
    (always keeping each row's top token), renormalise. Rows that would collapse
    to zero mass fall back to their unfiltered distribution.
    """
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[..., 0] = True  # always keep each row's top token
    sorted_probs = sorted_probs * keep
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, sorted_probs)
    total = filtered.sum(dim=-1, keepdim=True)
    safe = total > 0
    return torch.where(safe, filtered / total.clamp_min(1e-12), probs)


@torch.no_grad()
def sample(
    net,
    prompt_ids: list[int],
    max_new: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    stop_ids: set[int] | frozenset[int] = frozenset(),
    seed: int | None = None,
) -> list[int]:
    """Generate up to `max_new` tokens after `prompt_ids`. Returns the NEW ids only."""
    device = next(net.parameters()).device
    max_ctx = net.cfg.max_seq_len
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # Prefill the prompt once (truncated to the model's context window); decode_step
    # then advances one token at a time against the cache. The rope tables are sized
    # for max_seq_len + stomach + a small margin, so cap the generated length too.
    ctx = list(prompt_ids)[-max_ctx:]
    x = torch.tensor([ctx], dtype=torch.long, device=device)
    with _autocast_ctx(device):
        logits, state = net.prefill(x)
    logits = logits[0, -1].float()  # stable softmax; stays on device

    # RoPE tables cover positions [0, max_seq_len + stomach + 64); decode_step t lands
    # the new token at seq pos len(ctx)+t, so the cache supports this many extra tokens.
    rope_cap = max_ctx + net.cfg.stomach_tokens + 64 - len(ctx)
    out: list[int] = []
    budget = min(max_new, rope_cap)
    for _ in range(max(budget, 0)):
        if temperature <= 0.0:
            nxt = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs = _top_p_filter(probs, top_p)
            nxt = int(torch.multinomial(probs, num_samples=1, generator=gen).item())
        if nxt in stop_ids:
            break
        out.append(nxt)
        if len(out) >= budget:
            break
        tok = torch.tensor([[nxt]], dtype=torch.long, device=device)
        with _autocast_ctx(device):
            logits = net.decode_step(tok, state)[0].float()
    return out


@torch.no_grad()
def sample_batch(
    net,
    prompt_ids: list[int],
    n: int,
    max_new: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    stop_ids: set[int] | frozenset[int] = frozenset(),
    seed: int | None = None,
) -> list[list[int]]:
    """Generate `n` independent continuations of the SAME prompt in one batched forward.

    The whole point is GPU occupancy: prefill runs once on an (n, T) batch and the
    decode loop advances all n streams together through `decode_step` (both batch over
    the leading dim), so n rollouts cost ~one rollout's worth of launches instead of n.

    Diversity comes from per-row multinomial sampling — with temperature>0 each row
    draws independently, so the rows fan out. With temperature==0 (greedy) every row is
    identical (and equals the single-stream `sample` greedy decode), which the parity
    test relies on. Returns a list of n lists, each the NEW ids for that stream; a stream
    that emits a stop id (not appended) freezes there while the others keep going.
    """
    device = next(net.parameters()).device
    max_ctx = net.cfg.max_seq_len
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    ctx = list(prompt_ids)[-max_ctx:]
    x = torch.tensor([ctx], dtype=torch.long, device=device).expand(n, -1)
    with _autocast_ctx(device):
        logits, state = net.prefill(x)
    logits = logits[:, -1].float()  # (n, V), on device

    rope_cap = max_ctx + net.cfg.stomach_tokens + 64 - len(ctx)
    budget = max(min(max_new, rope_cap), 0)
    outs: list[list[int]] = [[] for _ in range(n)]
    done = [False] * n
    for _ in range(budget):
        if temperature <= 0.0:
            nxt = torch.argmax(logits, dim=-1)  # (n,)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs = _top_p_filter_batch(probs, top_p)
            nxt = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)  # (n,)
        ids = nxt.tolist()
        for i, t in enumerate(ids):
            if done[i]:
                continue
            if t in stop_ids:
                done[i] = True
            else:
                outs[i].append(t)
        if all(done):
            break
        # Feed every row (including frozen ones) to keep the batch rectangular; frozen
        # rows' outputs are simply ignored above.
        tok = nxt.unsqueeze(1)  # (n, 1)
        with _autocast_ctx(device):
            logits = net.decode_step(tok, state).float()
    return outs
