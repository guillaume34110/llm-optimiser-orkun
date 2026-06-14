"""Model soup — cross-arm weight averaging via a shared directory.

Each arm periodically and atomically dumps its float weights to
``<shared>/arm_<i>.safetensors``, then averages every peer dump it can read
(its own included) and loads the mean back in-place. One strong model emerges
from N arms with no gradient communication — strictly cheaper than all-reduce and
robust to arms running at different speeds (a stale peer just biases the mean a
little, it never blocks). Integer buffers are left untouched; only floating-point
parameters are souped.

Two invariants around the load:
  * the mean of unit-norm matrices is NOT unit-norm, so `soup_sync` calls
    `normalize_weights()` right after loading the average — otherwise the arm
    generates on an off-manifold net until its next optimiser step (WAAAGH
    invariant violation of exactly the kind bugs.md documents).
  * AnchorReg keeps its reference at the WARMUP checkpoint across soups, by
    design: every arm anchors to the same warmup weights, so the post-soup pull
    is consistent across arms and bounds total drift; re-snapshotting to the
    averaged weights would let the soup random-walk away from the substrate.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def average_state_dicts(states: list[dict]) -> dict:
    """Element-wise mean of a list of state dicts (must share keys & shapes)."""
    if not states:
        raise ValueError("no state dicts to average")
    out: dict = {}
    for k in states[0]:
        acc = None
        for s in states:
            t = s[k].float()
            acc = t.clone() if acc is None else acc + t
        out[k] = acc / len(states)
    return out


def dump_arm_weights(net, shared_dir: Path, arm_idx: int) -> Path:
    """Atomically write this arm's CPU float weights to the shared soup dir.

    Writes to a pid-tagged temp file then ``os.replace`` (atomic on POSIX) so a
    peer mid-read never sees a half-written checkpoint.
    """
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    final = shared_dir / f"arm_{arm_idx}.safetensors"
    tmp = shared_dir / f"arm_{arm_idx}.tmp_{os.getpid()}.safetensors"
    floats = {k: v.detach().cpu().float() for k, v in net.state_dict().items() if v.is_floating_point()}
    save_file(floats, str(tmp))
    os.replace(tmp, final)
    return final


def read_peer_states(shared_dir: Path) -> list[dict]:
    """Load every ``arm_*.safetensors`` present; skip any partial/corrupt read."""
    states = []
    for p in sorted(Path(shared_dir).glob("arm_*.safetensors")):
        try:
            states.append(load_file(str(p)))
        except Exception:
            continue
    return states


def soup_sync(net, shared_dir: Path, arm_idx: int) -> int:
    """Dump our weights, average all peers, load the mean in-place.

    Returns the number of peer dumps mixed (1 ⇒ only our own present ⇒ no-op).
    """
    dump_arm_weights(net, shared_dir, arm_idx)
    states = read_peer_states(shared_dir)
    if len(states) < 2:
        return len(states)
    mean = average_state_dicts(states)
    device = next(net.parameters()).device
    net.load_state_dict({k: v.to(device) for k, v in mean.items()}, strict=False)
    # Mean of unit-norm matrices is not unit-norm: restore the WAAAGH geometry
    # before the net generates again (plain test nets have no such invariant).
    if hasattr(net, "normalize_weights"):
        net.normalize_weights()
    return len(states)
