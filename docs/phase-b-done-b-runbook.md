# Orkun Phase B — Done-B Evaluation Runbook

Wires the `orkun/eval/` harness (merged, unit-proven) to real backends and renders the
Phase B pass/fail verdict. The harness modules are pure / seam-driven; this runbook is the
glue that feeds them a trained checkpoint, a filled store, and the live sandbox.

Prereqs: `pip install -e ../Orkish && pip install -e .`, a **trained** Orkish 80M checkpoint
loaded into `WaaaghNet` (the WM and reward head are meaningless on random init), and a loot
file produced by Phase A / Phase B self-play (`data/orkun/loot.jsonl`).

Thresholds live in `configs/eval.yaml` (sentinels — recalibrate after the first real run).

```python
import json, yaml
from model.tokenizer import OrkishTokenizer
from torch_impl.model.waaagh import WaaaghConfig, WaaaghNet
from orkun.world_model.waaagh_wm import WaaaghWorldModel

cfg = yaml.safe_load(open("configs/eval.yaml"))
tok = OrkishTokenizer(...)                                  # adjust to orkish_repo
net = WaaaghNet(WaaaghConfig(vocab_size=tok.vocab_size))    # load trained weights here
wm  = WaaaghWorldModel(net, tok)                            # WorldModelBackend over the net

rows = [json.loads(l) for l in open("data/orkun/loot.jsonl")]
```

## 1. WM calibration — held-out NLL + reward-head MAE

Split by `task_id` hash so the WM/head are scored on **unseen instances of seen families**.
Fit the reward head on the train side, then measure both metrics on the held-out side.

```python
from orkun.eval.holdout import split_by_task, transitions_from_wire
from orkun.eval.calibration import wm_nll, reward_head_mae
from orkun.world_model.reward_head import RewardHead

train, held = split_by_task(rows, frac=cfg["holdout_frac"], seed=cfg["holdout_seed"])

# (a) WM NLL/token over every real ToolResult transition in the held-out wires
held_transitions = [t for r in held for t in transitions_from_wire(r["wire"])]
nll = wm_nll(wm, held_transitions)                          # PASS if <= cfg["wm_nll_max"]

# (b) reward-head MAE: fit on train, measure |predict - graded| on held-out
head = RewardHead(wm)
head.fit(train, epochs=100)                                 # uses train rows' `graded`
mae = reward_head_mae(head, held)                           # PASS if <= cfg["reward_head_mae_max"]

print("held-out NLL/token:", nll, "| reward-head MAE:", mae)
```

`graded` is the sandbox's real grade — it stays ground truth, never predicted. Rows without
`graded` (e.g. Phase A rows) are skipped by `reward_head_mae`.

## 2. Sample efficiency — planning beats reactive (ablation i)

Two arms per task through the `attempt_fn` seam: a **reactive** arm (Phase A blind sampling
via `OrkishPolicy.rollout`) and a **planned** arm (WM-imagined ranking via `LookaheadPlanner`,
then `Dreamer.execute_verified` runs the top candidates for real). `n_real_exec` is the count
of **real sandbox executions** each arm consumed — the planner trades real rollouts for free
WM-imagined ones, so it should verify in fewer real executions.

```python
from orkun.eval.sample_efficiency import AttemptResult, run_sample_efficiency
from orkun.policy.orkish_policy import make_orkish_policy
from orkun.planning.lookahead import LookaheadPlanner
from orkun.planning.dreamer import Dreamer
from orkun.goals.families import ALL_FAMILIES
from orkun.goals.generator import GoalGen

policy  = make_orkish_policy(net, tok, temperature=0.8)
planner = LookaheadPlanner(policy, wm, head, k=cfg.get("plan_k", 4), horizon=3)
dreamer = Dreamer(known_tools=getattr(policy, "known_tools", None))

gen   = GoalGen(ALL_FAMILIES, seed=0)
tasks = [g.task for g in gen.propose(context={}, n=cfg.get("n_eval_tasks", 32))]
SAMPLES, TOP_N = cfg.get("samples", 8), cfg.get("top_n", 2)

def attempt_fn(task, mode):
    if mode == "reactive":
        # blind sampling: each of the `SAMPLES` rollouts is a real sandbox execution
        res = policy.rollout(task, samples=SAMPLES)
        return AttemptResult(solved=res is not None, n_real_exec=SAMPLES)
    # planned: imagine + rank under the WM (free), then execute only the top_n for real
    ranked = planner.imagine(task, n=SAMPLES)
    kept   = dreamer.execute_verified(task, ranked, top_n=TOP_N)
    return AttemptResult(solved=len(kept) > 0, n_real_exec=TOP_N)

report = run_sample_efficiency(tasks, attempt_fn, seed=0)
print("reactive mean exec:", report.reactive_mean_exec,
      "| planned mean exec:", report.planned_mean_exec,
      "| gain:", report.gain_ratio)                         # PASS if >= cfg["sample_gain_min"]
```

Means are over **all** tasks (a failed attempt still counts the executions it burned —
penalising failure-by-exhaustion); `report.per_task` carries the `(reactive, planned)` pair
per task for diagnostics. Tighten `n_real_exec` to the true per-attempt count by hooking
`run_fresh` if you need exact sandbox-call accounting rather than the configured budget.

## 3. Loop ablations — curiosity (ii) + WM-online (iii)

Drive `cycles` of `kunnin_cycle` per arm through the `cycle_fn` seam, toggling the flags the
arm exercises. Curiosity-on should cover more families; WM-online should show positive
curiosity progress where frozen stays flat.

`CycleReport` exposes `n_attempts`, `n_verified`, `picked_families` — not a per-cycle
progress scalar — so `cycle_fn` reads curiosity progress off the `EpistemicCuriosity`
instance after the cycle. That instance (and the LP / WM trainer) must **persist across an
arm's cycles**, so key the state by arm config.

```python
from orkun.eval.loop_ablation import CycleMetrics, run_loop_ablation, coverage
from orkun.loop import kunnin_cycle
from orkun.curiosity.learning_progress import LearningProgress
from orkun.curiosity.curriculum import LPCurriculum
from orkun.curiosity.epistemic import EpistemicCuriosity
from orkun.world_model.trainer import WMTrainer
from orkun.data.store import Store

families = [f.name for f in ALL_FAMILIES]
_arm_state: dict = {}    # per-arm persistent lp / curiosity / wm_trainer

def cycle_fn(arm_cfg, rng):
    key = tuple(sorted(arm_cfg.items()))
    st = _arm_state.setdefault(key, {
        "lp": LearningProgress(window=8),
        "curiosity": EpistemicCuriosity() if arm_cfg.get("curiosity") else None,
        "wm_trainer": WMTrainer(net, tok) if arm_cfg.get("wm_online") else None,
    })
    rep = kunnin_cycle(
        gen, st["lp"], LPCurriculum(epsilon=0.2, rng=rng), policy,
        Store(":memory:"), families, k=4, samples=SAMPLES, rng=rng,
        world_model_on=True, wm=wm, reward_head=head, planner=planner,
        curiosity=st["curiosity"], wm_trainer=st["wm_trainer"],
    )
    fam = rep.picked_families[0] if rep.picked_families else families[0]
    progress = st["curiosity"].progress(fam) if st["curiosity"] else 0.0
    return CycleMetrics(success=rep.n_verified > 0, progress=progress, family=fam)

# (ii) curiosity on/off — coverage
cur_arms = run_loop_ablation({"on": {"curiosity": True}, "off": {"curiosity": False}},
                             cycle_fn, cycles=60, families=families, seed=0)
assert coverage(cur_arms["on"], families) > coverage(cur_arms["off"], families)

# (iii) WM online vs frozen — curiosity progress
wm_arms = run_loop_ablation({"online": {"wm_online": True}, "frozen": {"wm_online": False}},
                            cycle_fn, cycles=60, families=families, seed=0)
assert sum(wm_arms["online"].progress_curve) > sum(wm_arms["frozen"].progress_curve)
```

**(iii) caveat — net weight isolation:** `WMTrainer(net, tok)` finetunes the *shared* `net`
in place, so an online arm would pollute the frozen arm's WM (and the policy/planner that
read the same net). For a clean ablation, give each arm its own checkpoint copy — load a
fresh `WaaaghNet` per arm and build a per-arm `wm` / `planner` inside the state dict above —
so "frozen" genuinely stays frozen.

## 4. Verdict

Feed the three measured scalars to `evaluate_done_b`; it compares each to its configured
threshold (`wm_nll`/`mae` upper-bounded, `sample_gain` lower-bounded) and ANDs the results.

```python
from orkun.eval.done_b import evaluate_done_b, load_eval_config

verdict = evaluate_done_b(load_eval_config("configs/eval.yaml"),
                          wm_nll_value=nll, reward_head_mae_value=mae,
                          sample_gain_ratio=report.gain_ratio)
for name, c in verdict.criteria.items():
    print(f"  {name}: {c.value:.3f} vs {c.threshold} -> {'PASS' if c.passed else 'FAIL'}")
print("DONE-B:", "PASS" if verdict.passed else "FAIL")
```

## Done when
- Held-out WM NLL/token ≤ `wm_nll_max` (section 1a).
- Reward-head MAE ≤ `reward_head_mae_max` on held-out (section 1b).
- `gain_ratio` = reactive_mean / planned_mean ≥ `sample_gain_min`, i.e. planning verifies in
  fewer real executions (section 2).
- Curiosity-on covers strictly more families than curiosity-off, and WM-online shows higher
  summed curiosity progress than frozen (section 3).
- `evaluate_done_b(...).passed is True` with the thresholds recalibrated for this checkpoint
  (section 4).
