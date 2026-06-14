# Orkun — RLVR & causal-curiosity self-play for a from-scratch 80M LLM

Orkun is the post-training and evaluation stack for **WAAAGH GROT-80M**, a custom
[nGPT](https://arxiv.org/abs/2410.01131)-style language model trained from scratch
(see the companion [Orkish](https://github.com/guillaume34110/llm-waaagh-Orkish) repo for the
architecture and pretraining). Orkun teaches that tiny backbone to **call tools** with
verifiable-reward RL, and — just as importantly — measures *honestly* where an 80M model
hits its ceiling.

> The model is small on purpose. The interesting work is the training loop and the
> diagnosis, not the leaderboard score.

## What's in here

| Area | Module | What it does |
|---|---|---|
| **Verifiable tasks** | `orkun/goals/families.py` | 9 `GoalFamily` generators (write_file, compute, arith, sequence, pipeline, echo, json_transform, bool_sat, fix_token), each with a difficulty curriculum and a programmatic checker |
| **RLVR** | `orkun/train/rlvr.py`, `train/reward.py`, `policy/group_rollout.py` | Group-relative rollouts → sandbox execution → verifier reward → policy update |
| **Anti-forgetting** | `orkun/train/anchor.py` | Proximal parameter-space pull `θ ← θ + η(θ*−θ)` — works with scale-invariant optimisers (Muon/AdamW) where an in-loss L2 penalty silently does nothing |
| **Self-play / curiosity** | `orkun/curiosity/`, `orkun/causal/`, `orkun/planning/` | Learning-progress curriculum, epistemic signals, counterfactual probing, lookahead/dreamer |
| **World model** | `orkun/world_model/` | Reward head + scoring over the WAAAGH substrate |
| **Sandbox** | `orkun/world/sandbox.py`, `world/verifier.py` | Real tempdir execution; reward is computed from actual tool effects, never simulated |
| **Eval** | `orkun/eval/minibench.py` | Capability (pass@k) **and** an induction copy-fidelity probe |
| **Demo** | `orkun/demo/` | Dependency-free web app: type a request, watch grot fire a real tool in a sandbox |

## The honest result (`orkun/eval/minibench.py`)

Two numbers matter more than val-loss:

**Capability** — pass@4 across the 9 families: the model reliably fires tools and solves
the `pipeline` family (read → compute → write), but collapses elsewhere.

**Induction probe** — teacher-forced copy fidelity. A trained induction head holds
`copy_acc ≈ 1.0` flat across payload lengths. Grot instead decays:

| payload len | copy_acc | exact_rate |
|---|---|---|
| 2  | 0.81 | 0.75 |
| 4  | 0.60 | 0.25 |
| 8  | 0.58 | 0.00 |
| 12 | 0.54 | 0.00 |

It copies the first ~1 token, then drifts. **Diagnosis: the bottleneck is the
pre-induction backbone, not the SFT/RLVR recipe** — no amount of post-training installs
an induction head that pretraining never grew. That conclusion drove the decision to
invest in pretraining before further post-training, and it's the kind of call this repo
is meant to show.

## Demo

```bash
python -m orkun.demo.server \
    --checkpoint path/to/best.safetensors \
    --config configs/orkun_sft_procedural_v4.yaml \
    --orkish-repo ../Orkish
# open http://127.0.0.1:8000
```

The page shows grot's **full token-by-token output** (not just the parsed call), executes
each tool in a fresh sandbox, and snapshots the sandbox filesystem **before/after** so the
effects are verifiable. Tools the model was actually trained on are marked; the rest are
shown dimmed (it was never taught them — it collapses everything onto `py_run`).

## Run the eval

```bash
pip install -e .
python -m orkun.eval.minibench \
    --checkpoint path/to/best.safetensors \
    --config configs/orkun_sft_procedural_v4.yaml \
    --orkish-repo ../Orkish
# writes runs/minibench_<tag>.{md,json}
```

## Layout

```
orkun/
  goals/        verifiable task families + difficulty + generator
  train/        rlvr, reward, anchor, soup, cf_distill, pack
  policy/       sampler, group rollout, backends (waaagh)
  curiosity/    learning-progress, epistemic, curriculum
  causal/       counterfactual probing
  planning/     dreamer, lookahead
  world/        sandbox + verifier
  world_model/  reward head + scoring
  eval/         minibench + ablations
  demo/         tool-call web demo
configs/        training configs (relative paths)
runs/           eval reports (checkpoints are git-ignored)
tests/          pytest suite
```

Heavy artifacts (checkpoints `*.safetensors`, packed corpora `*.npz`, run outputs) are
git-ignored and regenerable.

## License

MIT — see [LICENSE](LICENSE).
