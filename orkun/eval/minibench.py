"""Orkun minibench — a single command that turns a checkpoint into an honest report.

Two views of a WAAAGH checkpoint, both written to a JSON + Markdown artifact you can
commit next to a model card:

  1. CAPABILITY (`run_families`) — greedy + sampled pass@k over every GoalFamily, at the
     RLVR target temperature (0.45, not 0.8 — 0.8 degenerates the 80M into token salad
     and masks everything as 0/N). Greedy = "did it learn the wire format under argmax";
     sampled pass@k = "is a correct trajectory reachable under the self-play regime".

  2. INDUCTION (`induction_probe`) — a teacher-forced copy-fidelity curve. This is the
     metric val_loss cannot see and the reason the SFT plateaued: the model copies the
     first 1-2 chars of a high-entropy payload from the prompt, then drifts onto a digit
     prior. We force the gold completion and read the model's argmax at each payload
     position, with ZERO sampling noise, so the number is the induction circuit itself:
       * copy_acc      — fraction of payload tokens the model predicts correctly,
       * prefix_len    — how many leading payload tokens are right before the first miss
                         (this is the "copies 93_ then drifts" finding, quantified),
       * exact_rate    — fraction of trials where the WHOLE payload is reproduced.
     A trained induction head drives copy_acc -> 1.0 flat across payload lengths; a
     pre-induction backbone shows copy_acc decaying with length and prefix_len ~1-2.

Orkish is read-only: this imports the shared wire/model/verifier primitives, never edits
them. Reuses `orkun.scripts.run_kaggle` for checkpoint loading so there is one build path.

Usage:
    python -m orkun.eval.minibench \
        --checkpoint kaggle/_out_v4_burdok/sft_procedural/ckpts/best.safetensors \
        --config configs/orkun_sft_procedural_v4.yaml \
        --orkish-repo ../Orkish \
        --markdown report.md --json report.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch


def _build(args):
    """Resolve tokenizer, model and known-tool set from the staged Orkish repo."""
    from orkun.scripts.run_kaggle import _add_orkish_to_path, _load_tokenizer, _build_net
    _add_orkish_to_path(Path(args.orkish_repo))
    import yaml
    model_cfg = yaml.safe_load(open(args.config))["model"]
    tok = _load_tokenizer(Path(args.orkish_repo))
    net = _build_net(model_cfg, Path(args.checkpoint), args.device).eval()
    reg = Path(args.orkish_repo) / "data" / "tool_registry.json"
    ktools = frozenset(t["name"] for t in json.loads(reg.read_text())["tools"])
    if "python_repl" in ktools:
        ktools = ktools | frozenset(["py_run"])
    return net, tok, ktools


# --------------------------------------------------------------------------- families

def run_families(net, tok, ktools, *, samples: int, temperature: float, top_p: float):
    """Greedy + sampled pass@k for every GoalFamily at d0. Returns per-family rows."""
    from orkun.policy.group_rollout import GroupRollout
    from orkun.goals.families import ALL_FAMILIES

    greedy = GroupRollout(net, tok, temperature=0.0, top_p=1.0, max_new=256,
                          known_tools=ktools, base_seed=0)
    sampler = GroupRollout(net, tok, temperature=temperature, top_p=top_p, max_new=256,
                           known_tools=ktools, base_seed=0)
    rng = random.Random(0)
    rows = []
    for fam in ALL_FAMILIES:
        task = fam.sample(rng, difficulty=0)
        g = greedy.rollout_group(task, samples=1, seed=0).samples[0]
        res = sampler.rollout_group(task, samples=samples, seed=0)
        n_pass = sum(s.passed for s in res.samples)
        rows.append({
            "family": fam.name,
            "greedy_pass": bool(g.passed),
            "greedy_graded": round(g.graded, 3),
            "greedy_calls": g.n_calls,
            "sampled_pass": n_pass,
            "sampled_n": samples,
            "greedy_text": g.assistant_text[:200],
        })
    return rows


# --------------------------------------------------------------------------- induction

def _prompt_and_gold(payload: str):
    prompt = f"<|bos|><|user|>Print exactly ANSWER={payload} using python.<|assistant|>"
    gold = f"CALL: py_run(code=\"print('ANSWER={payload}')\")"
    gold_prefix = "CALL: py_run(code=\"print('ANSWER="  # everything before the payload
    return prompt, gold, gold_prefix


@torch.no_grad()
def induction_probe(net, tok, *, lengths, trials: int, seed: int = 0):
    """Teacher-forced copy fidelity on digit payloads of each length in `lengths`.

    For each payload we force the gold `print('ANSWER=<payload>')` completion, run a
    single prefill over prompt+gold, and read the model's argmax at every payload
    position. No sampling: the result is the induction circuit's copy behaviour itself.
    """
    device = next(net.parameters()).device
    rng = random.Random(seed)
    out = []
    for L in lengths:
        accs, prefixes, exacts = [], [], []
        for _ in range(trials):
            payload = "".join(rng.choice("0123456789") for _ in range(L))
            prompt, gold, gold_prefix = _prompt_and_gold(payload)
            p_ids = tok.encode(prompt, add_bos=False, add_eos=False)
            g_ids = tok.encode(gold, add_bos=False, add_eos=False)
            # Locate the payload token span inside the gold completion.
            pre_ids = tok.encode(gold_prefix, add_bos=False, add_eos=False)
            prepay_ids = tok.encode(gold_prefix + payload, add_bos=False, add_eos=False)
            lo, hi = len(pre_ids), len(prepay_ids)
            if hi <= lo or hi > len(g_ids):
                continue  # BPE boundary merged across the payload edge; skip this draw
            full = p_ids + g_ids
            x = torch.tensor([full], dtype=torch.long, device=device)
            logits, _ = net.prefill(x)
            logits = logits[0].float()  # (T, V); logits[i] predicts token i+1
            base = len(p_ids)            # gold token j sits at full index base + j
            correct, first_miss = 0, None
            for j in range(lo, hi):
                pred = int(torch.argmax(logits[base + j - 1]).item())
                if pred == g_ids[j]:
                    correct += 1
                elif first_miss is None:
                    first_miss = j - lo
            span = hi - lo
            accs.append(correct / span)
            prefixes.append(span if first_miss is None else first_miss)
            exacts.append(1.0 if correct == span else 0.0)
        n = max(len(accs), 1)
        out.append({
            "payload_len": L,
            "trials": len(accs),
            "copy_acc": round(sum(accs) / n, 3),
            "prefix_len": round(sum(prefixes) / n, 2),
            "exact_rate": round(sum(exacts) / n, 3),
        })
    return out


# --------------------------------------------------------------------------- render

def render_markdown(meta, fam_rows, probe_rows) -> str:
    L = ["# Orkun minibench report", ""]
    L += [f"- checkpoint: `{meta['checkpoint']}`",
          f"- config: `{meta['config']}`",
          f"- temperature: {meta['temperature']}  ·  pass@{meta['samples']}",
          f"- generated: {meta['timestamp']}", ""]
    if fam_rows:
        gp = sum(r["greedy_pass"] for r in fam_rows)
        sp = sum(r["sampled_pass"] > 0 for r in fam_rows)
        L += ["## Capability — GoalFamily pass rates", "",
              f"**greedy {gp}/{len(fam_rows)}**  ·  "
              f"**sampled(pass@{meta['samples']}) {sp}/{len(fam_rows)} families**", "",
              "| family | greedy | greedy graded | sampled pass@k |",
              "|---|---|---|---|"]
        for r in fam_rows:
            L.append(f"| {r['family']} | {'✓' if r['greedy_pass'] else '·'} "
                     f"| {r['greedy_graded']} | {r['sampled_pass']}/{r['sampled_n']} |")
        L.append("")
    if probe_rows:
        L += ["## Induction — teacher-forced copy fidelity", "",
              "`copy_acc` = fraction of payload tokens predicted correctly under forced "
              "decoding. `prefix_len` = correct leading tokens before first miss. "
              "`exact_rate` = whole-payload reproductions. A trained induction head holds "
              "`copy_acc≈1.0` flat across lengths; decay here = pre-induction backbone.", "",
              "| payload len | trials | copy_acc | prefix_len | exact_rate |",
              "|---|---|---|---|---|"]
        for r in probe_rows:
            L.append(f"| {r['payload_len']} | {r['trials']} | {r['copy_acc']} "
                     f"| {r['prefix_len']} | {r['exact_rate']} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Orkun minibench: capability + induction report")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--orkish-repo", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.45)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--probe-lengths", default="2,4,8,12,16",
                    help="comma-separated digit-payload lengths for the induction probe")
    ap.add_argument("--probe-trials", type=int, default=16)
    ap.add_argument("--families-only", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--markdown", dest="md_out")
    args = ap.parse_args()

    net, tok, ktools = _build(args)
    meta = {
        "checkpoint": args.checkpoint, "config": args.config,
        "temperature": args.temperature, "samples": args.samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    fam_rows = [] if args.probe_only else run_families(
        net, tok, ktools, samples=args.samples,
        temperature=args.temperature, top_p=args.top_p)
    lengths = [int(x) for x in args.probe_lengths.split(",") if x.strip()]
    probe_rows = [] if args.families_only else induction_probe(
        net, tok, lengths=lengths, trials=args.probe_trials)

    md = render_markdown(meta, fam_rows, probe_rows)
    print(md)
    if args.md_out:
        Path(args.md_out).write_text(md)
        print(f"\n[minibench] wrote {args.md_out}", file=sys.stderr)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"meta": meta, "families": fam_rows, "induction": probe_rows}, indent=2))
        print(f"[minibench] wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
