"""cf-distill vs vanilla corpus builder — the ablation lives in corpus composition.

Both modes emit wire-JSONL that Orkish's stock `pack_sft` consumes unchanged
(doctrine: keep the training path byte-identical; the experiment is the data, not
the loss). vanilla = one wire per verified trajectory. cf = the necessity-pruned
minimal causal trajectory, repeated proportionally to its max necessity so causal
trajectories receive more gradient (repetition is the only upweighting the stock
packer supports; token-level weighting is deferred to Phase B).

NOTE: since Orkish 83d05ac the stock packer dedups identical wires, which would
silently collapse the repetition boost to 1 copy. Pack Orkun corpora through
`orkun.train.pack.load_examples_boosted`, which re-expands duplicates after the
stock packer's filtering.

Pruning rebuilds the wire keeping only assistant CALL:s whose action was necessary
(parallel to `necessity`). Self-play wires (scripts.self_play._build_wire) end on a
ToolResult — they carry no final assistant report — but the stock packer
(scripts.pack_sft) rejects any wire whose last turn isn't a no-CALL assistant turn.
So BOTH arms run their wires through `_ensure_final_report`, which appends a
synthetic "Done." report when one is missing. That repair is the only shaping the
corpus builder does to the stored trajectory.
"""
from __future__ import annotations

import json
from pathlib import Path

from infer.monkey_wire import (
    AssistantMsg,
    Conversation,
    ToolResult,
    UserMsg,
    parse,
    serialize,
)

_EPS = 1e-9


def _ensure_final_report(wire: str) -> str:
    """Guarantee the wire ends on a no-CALL assistant turn (packer requirement).

    Self-play wires end on a ToolResult; append a synthetic final report so
    scripts.pack_sft.load_examples accepts them. A wire that already ends on a
    no-CALL assistant turn is returned unchanged.
    """
    conv = parse(wire)
    last = conv.turns[-1] if conv.turns else None
    if isinstance(last, AssistantMsg) and not last.tool_calls:
        return wire
    conv.turns.append(AssistantMsg(text="Done."))
    return serialize(conv)


def _prune_wire(wire: str, necessity: list[float]) -> str:
    """Rebuild the wire keeping only the necessary tool calls + their results.

    Stored wire shape: user, assistant(all CALL:s), one ToolResult per executed
    call. `necessity` is aligned 1:1 with the executed calls (= the ToolResults) by
    the loop. Pruning is sound ONLY when that alignment holds, so we require
    len(tool_calls) == len(necessity) == len(results); otherwise we conservatively
    keep the whole trajectory (a misaligned prune would corrupt the corpus). Either
    way the result is run through `_ensure_final_report`.
    """
    conv = parse(wire)
    if not necessity or not any(n > _EPS for n in necessity):
        return _ensure_final_report(wire)

    user_turn = next((t for t in conv.turns if isinstance(t, UserMsg)), None)
    call_turn = next((t for t in conv.turns if isinstance(t, AssistantMsg) and t.tool_calls), None)
    results = [t for t in conv.turns if isinstance(t, ToolResult)]
    if call_turn is None or not (len(call_turn.tool_calls) == len(necessity) == len(results)):
        return _ensure_final_report(wire)

    keep = [i for i, n in enumerate(necessity) if n > _EPS]
    kept_calls = [call_turn.tool_calls[i] for i in keep]
    kept_results = [results[i] for i in keep]
    call_lines = "\n".join(f'CALL: {c.name}(' + _fmt_args(c.args) + ')' for c in kept_calls)

    new = Conversation(system=conv.system, eos=True)
    if user_turn is not None:
        new.turns.append(UserMsg(text=user_turn.text))
    new.turns.append(AssistantMsg(text=call_lines))
    new.turns.extend(kept_results)
    new.turns.append(AssistantMsg(text="Done."))   # synthetic final report (packer requires it)
    return serialize(new)


def _fmt_args(args: dict) -> str:
    return ", ".join(f'{k}={v!r}' for k, v in args.items())


def build_corpus(rows: list[dict], mode: str, out_path: Path, boost: int = 4) -> int:
    """Write a wire-JSONL corpus. Returns the number of rows written."""
    if mode not in ("vanilla", "cf"):
        raise ValueError(f"mode must be 'vanilla' or 'cf', got {mode!r}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if mode == "vanilla":
                wires = [(_ensure_final_report(row["wire"]), 1)]
            else:
                pruned = _prune_wire(row["wire"], row.get("necessity", []))
                max_nec = max(row.get("necessity", [0.0]) or [0.0])
                copies = 1 + round(max_nec * boost)
                wires = [(pruned, copies)]
            for wire, copies in wires:
                for _ in range(copies):
                    f.write(json.dumps(
                        {"task": row.get("task"), "wire": wire, "source": f"orkun_{mode}"},
                        ensure_ascii=False,
                    ) + "\n")
                    written += 1
    return written
