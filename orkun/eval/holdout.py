"""Held-out construction for Done-B calibration.

`transitions_from_wire` reconstructs the per-turn (prefix, result) transitions of a
wire by growing the prefix turn-by-turn — the single source of truth for this logic,
re-used by `loop._curiosity_over_results`. `split_by_task` partitions store rows into
train / held-out by a deterministic hash of the task_id, so a task never lands on both
sides and the split is reproducible by seed. Certifies WM/head generalisation to unseen
instances of seen families.
"""
from __future__ import annotations

import hashlib

from infer.monkey_wire import AssistantMsg, ToolResult, UserMsg, parse


def transitions_from_wire(wire: str) -> list[tuple[str, str]]:
    """(prefix, result) transitions; prefix grows turn-by-turn.

    User text and the first assistant CALL text seed the prefix; each ToolResult's
    payload is paired with the prefix that precedes it, then appended to the prefix so
    the next result is scored against everything before it.
    """
    conv = parse(wire)
    user = next((t.text for t in conv.turns if isinstance(t, UserMsg)), "")
    assistant = next((t.text for t in conv.turns if isinstance(t, AssistantMsg)), "")
    prefix = user + assistant
    transitions: list[tuple[str, str]] = []
    for turn in conv.turns:
        if isinstance(turn, ToolResult):
            transitions.append((prefix, turn.payload_str))
            prefix = prefix + turn.payload_str
    return transitions


def _task_hash(seed: int, task_id: str) -> float:
    """Stable hash of (seed, task_id) into [0, 1). Stdlib-only, process-independent
    (Python's built-in hash() is salted per process and must not be used here)."""
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def split_by_task(rows: list[dict], frac: float = 0.2, seed: int = 0
                  ) -> tuple[list[dict], list[dict]]:
    """Partition rows into (train, held_out) by task_id hash.

    A task_id is held out iff hash(seed, task_id) < frac, so all rows of a task land on
    the same side and ~`frac` of distinct task_ids are held out. Reproducible by seed.
    """
    train: list[dict] = []
    held: list[dict] = []
    for row in rows:
        if _task_hash(seed, str(row["task"])) < frac:
            held.append(row)
        else:
            train.append(row)
    return train, held
