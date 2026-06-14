"""Generate a procedural SFT corpus from the GoalFamily oracles.

Each family exposes sample(rng, difficulty) -> Task and oracle(task) -> [ToolCall].
For every sampled task we render the oracle solution into the monkey wire dialect,
execute it in the REAL sandbox (orkun.world.verifier.run_fresh), and keep only the
trajectories that genuinely pass every check. The emitted wire matches what the
self-play GroupRollout produces for a passing trajectory (all calls in one assistant
turn, one ToolResult per executed call) plus a terminal assistant report turn so the
result packs through Orkish scripts/pack_sft.py (which requires a final report).

Output: a JSONL of {"wire", "task", "completed": true} records, ready for pack_sft.

Usage:
    python -m orkun.scripts.gen_procedural_sft \\
        --orkish-repo /Users/.../Orkish \\
        --out orkun/data/sft_proc/traces.jsonl \\
        --difficulties 0 1 2 --per-cell 400 --max-total 5000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def _add_orkish_to_path(orkish_repo: Path) -> None:
    s = str(orkish_repo)
    if s not in sys.path:
        sys.path.insert(0, s)


def render_call(tc) -> str:
    """Render a ToolCall as the `CALL: name(k=repr(v), ...)` text the model must emit.

    repr() emits a Python literal that the wire parser (_parse_args -> ast.literal_eval)
    round-trips, and collapses newlines in code/content to single-line \\n escapes so the
    CALL: regex is not broken by real newlines.
    """
    args = ", ".join(f"{k}={v!r}" for k, v in tc.args.items())
    return f"CALL: {tc.name}({args})"


def _final_report(steps) -> str:
    """Terminal assistant turn (no CALL:). Reward-irrelevant — no check reads it.

    Echo the last non-empty stdout/content so the report is plausible; else "Done."."""
    for st in reversed(steps):
        res = st.result
        if isinstance(res, dict):
            val = res.get("stdout") or res.get("content")
            if val:
                return f"Done. {str(val).strip()[:120]}"
    return "Done."


def build_record(task, calls, steps):
    """Build a {"wire","task","completed"} record from a verified gold trajectory."""
    from infer.monkey_wire import AssistantMsg, Conversation, ToolResult, UserMsg, serialize

    assistant_text = "\n".join(render_call(c) for c in calls)
    conv = Conversation(system=task.system, eos=True)
    conv.turns.append(UserMsg(text=task.prompt))
    conv.turns.append(AssistantMsg(text=assistant_text))
    for st in steps:
        conv.turns.append(
            ToolResult(name=st.call.name, payload_str=json.dumps(st.result, ensure_ascii=False))
        )
    conv.turns.append(AssistantMsg(text=_final_report(steps)))
    return {"wire": serialize(conv), "task": task.prompt, "completed": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orkish-repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--difficulties", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-cell", type=int, default=400)
    ap.add_argument("--max-total", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--boost", type=str, default="",
        help="per-family per-cell multiplier, e.g. 'echo=4,write_file=2'. Generates MORE "
             "DISTINCT traces for the named families (concentrates the copy gradient) — never "
             "duplicates (echo payloads are unique by design; duplicating defeats the copy).")
    args = ap.parse_args()

    boost: dict[str, int] = {}
    for tok in (t.strip() for t in args.boost.split(",") if t.strip()):
        name, _, mult = tok.partition("=")
        boost[name.strip()] = int(mult)

    _add_orkish_to_path(args.orkish_repo)
    from infer.monkey_wire import parse_calls
    from orkun.goals.families import ALL_FAMILIES
    from orkun.world.verifier import run_fresh

    reg = json.loads((args.orkish_repo / "data" / "tool_registry.json").read_text())
    known = frozenset(t["name"] for t in reg["tools"]) | frozenset(["py_run"])

    rng = random.Random(args.seed)
    records: list[dict] = []
    seen: set[str] = set()
    kept = Counter()
    stats = Counter()  # per-family: sampled / failed_verify / failed_roundtrip / dup

    for fam in ALL_FAMILIES:
        cell = args.per_cell * boost.get(fam.name, 1)
        for d in args.difficulties:
            for _ in range(cell):
                stats[f"{fam.name}.sampled"] += 1
                task = fam.sample(rng, d)
                calls = fam.oracle(task)
                # round-trip: the rendered CALL text must parse back to the oracle calls
                assistant_text = "\n".join(render_call(c) for c in calls)
                parsed = parse_calls(assistant_text, known_tools=known)
                if [(p.name, p.args) for p in parsed] != [(c.name, c.args) for c in calls]:
                    stats[f"{fam.name}.failed_roundtrip"] += 1
                    continue
                rew = run_fresh(task, calls)
                if not rew.passed:
                    stats[f"{fam.name}.failed_verify"] += 1
                    continue
                rec = build_record(task, calls, rew.steps)
                h = hash(rec["wire"])
                if h in seen:
                    stats[f"{fam.name}.dup"] += 1
                    continue
                seen.add(h)
                records.append(rec)
                kept[fam.name] += 1

    rng.shuffle(records)
    if len(records) > args.max_total:
        records = records[: args.max_total]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"=== procedural SFT generated: {len(records)} records -> {args.out} ===")
    print("kept per family:", dict(kept))
    drops = {k: v for k, v in stats.items() if not k.endswith(".sampled")}
    if drops:
        print("drops:", dict(drops))
    missing = [f.name for f in ALL_FAMILIES if kept[f.name] == 0]
    if missing:
        print(f"WARNING: families with zero kept traces: {missing}")


if __name__ == "__main__":
    main()
