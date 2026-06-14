"""Diagnostic: dump generation of the SFT-warmed best.safetensors on d0 tasks.

Two passes per family:
  * greedy (temperature=0, 1 sample) — does the model reproduce a parsable, correct
    CALL: trajectory under its argmax mode? This is the strict "did it learn the wire
    format" gate; a warmed model should pass the majority of d0 families here.
  * sampled (--temperature, --samples draws) — pass@k > 0 shows the format is
    reachable under the self-play sampling regime that feeds RLVR. The default
    matches configs/phase_a.yaml (0.45): gates must be measured at the TARGET
    regime temperature — 0.8 degenerates the 80M into token salad and masks
    everything as 0/N (V2 post-mortem).

Covers every GoalFamily. Reuses run_kaggle helpers; Orkish read-only.
"""
import sys, random, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--orkish-repo", required=True)
ap.add_argument("--samples", type=int, default=6)
ap.add_argument("--temperature", type=float, default=0.45)
ap.add_argument("--top-p", type=float, default=0.95)
args = ap.parse_args()

import yaml
from orkun.scripts.run_kaggle import _add_orkish_to_path, _load_tokenizer, _build_net
_add_orkish_to_path(Path(args.orkish_repo))

import json as _json
from orkun.policy.group_rollout import GroupRollout
from orkun.goals.families import ALL_FAMILIES

model_cfg = yaml.safe_load(open(args.config))["model"]
tok = _load_tokenizer(Path(args.orkish_repo))
net = _build_net(model_cfg, Path(args.checkpoint), "cpu")

reg = Path(args.orkish_repo) / "data" / "tool_registry.json"
ktools = frozenset(t["name"] for t in _json.loads(reg.read_text())["tools"])
if "python_repl" in ktools:
    ktools = ktools | frozenset(["py_run"])
print("known_tools:", sorted(ktools))

greedy = GroupRollout(net, tok, temperature=0.0, top_p=1.0, max_new=256,
                      known_tools=ktools, base_seed=0)
sampler = GroupRollout(net, tok, temperature=args.temperature, top_p=args.top_p,
                       max_new=256, known_tools=ktools, base_seed=0)

rng = random.Random(0)
greedy_pass = 0
sample_pass = 0
n_fams = len(ALL_FAMILIES)
for fam in ALL_FAMILIES:
    task = fam.sample(rng, difficulty=0)
    print("\n" + "=" * 70)
    print(f"FAMILY {fam.name} d0  task.id={task.id}")
    print("PROMPT WIRE:", repr(greedy._prompt_wire(task))[:300])

    # greedy (deterministic): 1 sample is representative
    g = greedy.rollout_group(task, samples=1, seed=0)
    gs = g.samples[0]
    print(f"--- GREEDY: passed={gs.passed} graded={gs.graded} n_calls={gs.n_calls}")
    print("    text:", repr(gs.assistant_text)[:300])
    if gs.passed:
        greedy_pass += 1

    # sampled pass@k
    res = sampler.rollout_group(task, samples=args.samples, seed=0)
    for i, s in enumerate(res.samples):
        print(f"--- sample {i}: passed={s.passed} graded={s.graded} n_calls={s.n_calls}")
        print("    text:", repr(s.assistant_text)[:200])
    n_pass = len(res.passed) if hasattr(res, "passed") else sum(s.passed for s in res.samples)
    print(f"  => sampled {n_pass}/{len(res.samples)} passed")
    if n_pass > 0:
        sample_pass += 1

print("\n" + "=" * 70)
print(f"SUMMARY: greedy {greedy_pass}/{n_fams} families pass  |  "
      f"sampled(pass@{args.samples}) {sample_pass}/{n_fams} families pass")
