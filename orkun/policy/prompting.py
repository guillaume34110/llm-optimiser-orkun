"""Single source of truth for the policy prompt wire.

Both OrkishPolicy (sequential rejection sampling) and GroupRollout (batched
GRPO rollout) condition generation on this exact prefix, and RLVR scores
logπ(action) on it too — REINFORCE is unbiased only when the scored context is
byte-identical to the sampled one. Keeping the builder in one place makes that
invariant structural instead of a copy-paste discipline.
"""
from __future__ import annotations

from scripts.verifier import Task


def prompt_wire(task: Task, system: str | None = None) -> str:
    """Generation prefix: bos [+ system] + user prompt + assistant header."""
    sys = task.system if getattr(task, "system", None) is not None else system
    parts = ["<|bos|>"]
    if sys is not None:
        parts.append("<|system|>" + sys)
    parts.append("<|user|>" + task.prompt)
    parts.append("<|assistant|>")
    return "".join(parts)
