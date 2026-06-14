# Orkun Phase A — Experiment Runbook

Prereqs: `pip install -e ../Orkish && pip install -e .`, a warm-start Orkish 80M
checkpoint loaded into `WaaaghNet` (random init also runs but solves little).

## 1. Curriculum ablation (LP vs random) — done-criterion "success rises via LP"

Drive the loop for both curricula and compare success curves. With a real policy,
`solve_fn(family)` wraps a per-family `OrkishPolicy.rollout`:

```python
import yaml, random
from model.tokenizer import OrkishTokenizer
from torch_impl.model.waaagh import WaaaghConfig, WaaaghNet
from orkun.goals.families import ALL_FAMILIES
from orkun.goals.generator import GoalGen
from orkun.policy.orkish_policy import make_orkish_policy
from orkun.eval.ablation import run_curriculum_ablation

cfg = yaml.safe_load(open("configs/phase_a.yaml"))
tok = OrkishTokenizer(cfg["substrate"]["tokenizer"])      # adjust to orkish_repo
net = WaaaghNet(WaaaghConfig(vocab_size=tok.vocab_size))  # load warm-start weights here
# swap the LLM by passing a different LMBackend here (e.g. an HTTP-API backend);
# make_orkish_policy wraps the local WaaaghBackend(net, tok).
pol = make_orkish_policy(net, tok, temperature=cfg["policy"]["temperature"])
gen = GoalGen(ALL_FAMILIES, seed=0)

def solve_fn(family):
    # draw one goal of this family and attempt it
    g = next(g for g in gen.propose({"difficulty": {}}, n=8) if g.family == family)
    return pol.rollout(g.task, samples=cfg["policy"]["samples"]) is not None

lp  = run_curriculum_ablation("lp",     [f.name for f in ALL_FAMILIES], solve_fn, cycles=200, k=4, seed=0)
rnd = run_curriculum_ablation("random", [f.name for f in ALL_FAMILIES], solve_fn, cycles=200, k=4, seed=0)
print("LP final:", lp.success_curve[-1], "Random final:", rnd.success_curve[-1])
assert lp.success_curve[-1] >= rnd.success_curve[-1]   # PASS criterion
```

## 2. Counterfactual corpus + cf-distill vs vanilla — done-criterion "cf > vanilla"

Run the loop to fill the store, then build both corpora and train with Orkish's
stock packer/trainer (unmodified), then re-measure success rate:

```bash
# loop fills data/orkun/loot.jsonl (run kunnin_cycle for cfg.loop.cycles)
python -m orkun.scripts.run_loop --config configs/phase_a.yaml   # thin driver around kunnin_cycle

# build both corpora from the same loot
python - <<'PY'
import json
from orkun.train.cf_distill import build_corpus
rows = [json.loads(l) for l in open("data/orkun/loot.jsonl")]
print("vanilla rows:", build_corpus(rows, "vanilla", "data/orkun/vanilla.jsonl", boost=4))
print("cf rows:",      build_corpus(rows, "cf",      "data/orkun/cf.jsonl",      boost=4))
PY

# pack each with Orkish's stock packer (byte-identical training path)
python -m scripts.pack_sft --in-path data/orkun/vanilla.jsonl --train-out data/orkun/vanilla_train.npz --val-out data/orkun/vanilla_val.npz
python -m scripts.pack_sft --in-path data/orkun/cf.jsonl      --train-out data/orkun/cf_train.npz      --val-out data/orkun/cf_val.npz

# fine-tune each (Orkish SFT, MLX on Mac or torch_impl on T4×2) from the same warm-start,
# then re-run section 1's success measurement on each resulting checkpoint.
# PASS criterion: success-rate(cf checkpoint) > success-rate(vanilla checkpoint).
```

## Done when
- LP success curve ends above random (section 1).
- `data/orkun/cf.jsonl` exists and is non-empty (counterfactual corpus generated).
- cf-distilled checkpoint beats vanilla-distilled on held-out family success (section 2).
- necessity scores correlate with true causal impact (already unit-proven in `test_counterfactual.py`; spot-check on a multi-step task).
