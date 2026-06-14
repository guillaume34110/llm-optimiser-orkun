"""run_kaggle.py — 12-hour Kaggle 2×T4 training driver.

Saturates both T4 GPUs by running one independent training arm per device. Each arm
loads the checkpoint, builds the full Phase B stack (WM, RLVR, curiosity, AnchorReg),
and drives kunnin_cycle in a time-bounded loop. Parallel sandbox execution uses all
available CPU cores (GoalGen oracle + counterfactual ablation via ThreadPoolExecutor).

Kaggle usage (run from a notebook cell):
    !python -m orkun.scripts.run_kaggle \\
        --checkpoint /kaggle/input/orkish/ckpts/step_XXXX.safetensors \\
        --config     /kaggle/input/orkish/configs/waaagh_grot_80m.yaml \\
        --orkish-repo /kaggle/input/orkish \\
        --out        /kaggle/working/orkun_run

Outputs (per arm):
    <out>/arm_<N>/loot.jsonl                  trajectories verified this arm
    <out>/arm_<N>/ckpt_<step>.safetensors     checkpoint every --save-every-minutes
    <out>/arm_<N>/ckpt_final.safetensors      final checkpoint on clean exit
    <out>/loot_merged.jsonl                   merged rows from all arms (written at end)

After the run, use the checkpoint with Done-B:
    python -m orkun.scripts.run_done_b \\
        --checkpoint <out>/arm_0/ckpt_final.safetensors \\
        --config     <config> --loot <out>/loot_merged.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# helpers — identical to run_done_b.py (both copied, not shared, to keep
# each script self-contained and importable without cross-import cycles)
# ---------------------------------------------------------------------------

def _add_orkish_to_path(orkish_repo: Path) -> None:
    s = str(orkish_repo)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_tokenizer(orkish_repo: Path):
    from model.tokenizer import OrkishTokenizer
    return OrkishTokenizer(orkish_repo / "data" / "tokenizer" / "orkish-bpe-8k.json")


def _build_net(model_cfg: dict, checkpoint: Path, device: str):
    from torch_impl.model.waaagh import WaaaghConfig, WaaaghNet
    valid = {f.name for f in dataclasses.fields(WaaaghConfig)}
    kw = {k: v for k, v in model_cfg.items() if k in valid}
    kw.setdefault("attn_backend", "sdpa")
    cfg = WaaaghConfig(**kw)
    net = WaaaghNet(cfg)
    tensors = load_file(str(checkpoint))
    if any(k.startswith("model.") for k in tensors):
        state = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
    else:
        state = {k: v for k, v in tensors.items() if not k.startswith("optim.")}
    net.load_state_dict(state, strict=False)
    return net.to(device).train()


def _save_ckpt(net, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu() for k, v in net.state_dict().items()}, str(path))
    print(f"  saved {path.name}", flush=True)


# Hyperparameters shared with the experiment runbook. configs/phase_a.yaml is the
# single source of truth; these literals are only the fallback when the yaml is not
# staged (and they mirror its values so absence ≠ drift).
_PHASE_DEFAULTS = {
    "temperature": 0.45,
    "top_p": 0.95,
    "max_new": 256,
    "lp_window": 20,
    "epsilon": 0.2,
    "default_difficulty": 1,
}


def _load_phase_cfg(path: Path | None) -> dict:
    cfg = dict(_PHASE_DEFAULTS)
    if path is None or not Path(path).is_file():
        return cfg
    y = yaml.safe_load(Path(path).read_text()) or {}
    pol = y.get("policy", {})
    cfg["temperature"] = float(pol.get("temperature", cfg["temperature"]))
    cfg["top_p"] = float(pol.get("top_p", cfg["top_p"]))
    cfg["max_new"] = int(pol.get("max_new", cfg["max_new"]))
    cfg["lp_window"] = int(y.get("learning_progress", {}).get("window", cfg["lp_window"]))
    cfg["epsilon"] = float(y.get("curriculum", {}).get("epsilon", cfg["epsilon"]))
    cfg["default_difficulty"] = int(
        y.get("goals", {}).get("default_difficulty", cfg["default_difficulty"]))
    return cfg


# ---------------------------------------------------------------------------
# arm loop — one per GPU, runs in a separate process
# ---------------------------------------------------------------------------

def _arm_loop(
    arm_idx: int,
    device: str,
    *,
    checkpoint: str,
    model_cfg: dict,
    orkish_repo: str,
    out_dir: str,
    budget_seconds: float,
    n_goals: int,
    k: int,
    samples: int,
    save_interval_seconds: float,
    parallel_workers: int,
    anchor_eta: float,
    seed: int,
    soup_dir: str,
    soup_interval_seconds: float,
    soup_on: bool,
    profile_on: bool,
    phase_cfg: dict,
) -> None:
    """Single-arm training loop. Must be importable (module-level, no closures)."""
    orkish_path = Path(orkish_repo)
    _add_orkish_to_path(orkish_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from orkun.world_model.waaagh_wm import WaaaghWorldModel
    from orkun.world_model.trainer import WMTrainer
    from orkun.world_model.reward_head import RewardHead
    from orkun.train.rlvr import RLVRTrainer
    from orkun.train.anchor import AnchorReg
    from orkun.policy.orkish_policy import make_orkish_policy
    from orkun.policy.group_rollout import GroupRollout
    from orkun.train.soup import soup_sync
    from orkun.planning.lookahead import LookaheadPlanner
    from orkun.curiosity.learning_progress import LearningProgress
    from orkun.curiosity.curriculum import LPCurriculum
    from orkun.curiosity.epistemic import EpistemicCuriosity
    from orkun.goals.difficulty import DifficultyTracker
    from orkun.goals.families import ALL_FAMILIES
    from orkun.goals.generator import GoalGen
    from orkun.data.store import Store
    from orkun.loop import kunnin_cycle

    tok = _load_tokenizer(orkish_path)
    net = _build_net(model_cfg, Path(checkpoint), device)

    reg = orkish_path / "data" / "tool_registry.json"
    if reg.is_file():
        ktools = frozenset(t["name"] for t in json.loads(reg.read_text())["tools"])
    else:
        from infer.executors import EXECUTORS
        ktools = frozenset(EXECUTORS.keys())
    if "python_repl" in ktools:
        ktools = ktools | frozenset(["py_run"])

    wm = WaaaghWorldModel(net, tok)
    head = RewardHead(wm)
    anchor = AnchorReg(net, eta=anchor_eta, exclude=AnchorReg.WAAAGH11_FREE)
    wm_trainer = WMTrainer(net, tok, anchor=anchor)
    rlvr = RLVRTrainer(net, tok, anchor=anchor)
    policy = make_orkish_policy(
        net, tok,
        temperature=phase_cfg["temperature"], top_p=phase_cfg["top_p"],
        max_new=phase_cfg["max_new"], known_tools=ktools,
    )
    planner = LookaheadPlanner(policy, wm, head, k=4, horizon=3)
    # Batched group rollout = the perf core: N rollouts in one prefill + shared decode,
    # per-sample rewards for a group-relative (GRPO) baseline. distinct base_seed/arm.
    group = GroupRollout(net, tok,
                         temperature=phase_cfg["temperature"], top_p=phase_cfg["top_p"],
                         max_new=phase_cfg["max_new"],
                         known_tools=ktools, base_seed=seed + arm_idx * 997)
    lp = LearningProgress(window=phase_cfg["lp_window"])
    curiosity = EpistemicCuriosity()
    goal_gen = GoalGen(ALL_FAMILIES, seed=seed + arm_idx * 997)  # distinct per arm
    families = [f.name for f in ALL_FAMILIES]
    store = Store(out / "loot.jsonl")
    reward_history: list[float] = []
    # Difficulty starts at 0 regardless of the yaml default: the warmup checkpoint must
    # earn its way up; the tracker promotes per family on a ≥0.8 success window.
    difficulty = DifficultyTracker(start=0)

    t_start = time.time()
    t_last_save = t_start
    t_last_soup = t_start
    cycle = 0
    rows_verified = 0
    consecutive_failures = 0
    max_consecutive_failures = 5
    soup_path = Path(soup_dir)
    tag = f"[arm {arm_idx}|{device}]"
    print(f"{tag} ready — starting loop (budget {budget_seconds/3600:.1f}h, "
          f"group on, soup {'on' if soup_on else 'off'})", flush=True)

    while True:
        elapsed = time.time() - t_start
        if elapsed >= budget_seconds:
            break

        rng = random.Random(seed + arm_idx * 997 + cycle)
        curriculum = LPCurriculum(epsilon=phase_cfg["epsilon"], rng=rng)
        profile: dict | None = {} if profile_on else None

        t_cycle = time.time()
        try:
            rep = kunnin_cycle(
                goal_gen, lp, curriculum, policy, store, families,
                n_goals=n_goals, k=k, samples=samples, rng=rng,
                world_model_on=True,
                wm=wm, curiosity=curiosity, reward_head=head,
                planner=planner, wm_trainer=wm_trainer, rlvr=rlvr, tok=tok,
                reward_history=reward_history,
                parallel_workers=parallel_workers,
                group=group, profile=profile,
                difficulty=difficulty,
            )
            rows_verified += rep.n_verified
            consecutive_failures = 0
        except Exception:
            consecutive_failures += 1
            print(f"{tag} cycle {cycle} error "
                  f"({consecutive_failures}/{max_consecutive_failures} consecutive):",
                  flush=True)
            traceback.print_exc()
            if consecutive_failures >= max_consecutive_failures:
                # A persistent fault (OOM loop, corrupted state) would otherwise burn the
                # whole GPU budget printing tracebacks; save what we have and stop the arm.
                print(f"{tag} aborting after {consecutive_failures} consecutive failures",
                      flush=True)
                break

        # REWARD HEAD — fit on stored graded rows so planner/store reward_hat is live,
        # not the untrained head's constant sigmoid. Cheap (tiny head), every 5 cycles.
        if cycle % 5 == 4 and len(store) >= 8:
            try:
                graded_rows = [r for r in store.sample(64) if "graded" in r]
                if graded_rows:
                    head.fit(graded_rows, epochs=25)
            except Exception as exc:
                print(f"{tag} reward-head fit error (skipped): {exc}", flush=True)

        if profile is not None:
            dt = time.time() - t_cycle
            phases = " ".join(f"{k}={v:.1f}s" for k, v in sorted(profile.items()))
            print(f"{tag} cycle {cycle} prof | total={dt:.1f}s {phases}", flush=True)

        cycle += 1

        now = time.time()
        # MODEL SOUP — periodically average peer arms' weights in-place (one strong model).
        if soup_on and now - t_last_soup >= soup_interval_seconds:
            try:
                n_mixed = soup_sync(net, soup_path, arm_idx)
                if n_mixed > 1:
                    print(f"{tag} souped {n_mixed} arms", flush=True)
            except Exception as exc:
                print(f"{tag} soup error (skipped): {exc}", flush=True)
            t_last_soup = now

        if now - t_last_save >= save_interval_seconds:
            ckpt_path = out / f"ckpt_{cycle:06d}.safetensors"
            _save_ckpt(net, ckpt_path)
            h = (now - t_start) / 3600
            print(
                f"{tag} cycle {cycle} | {h:.2f}h | {rows_verified} rows verified",
                flush=True,
            )
            t_last_save = now

    _save_ckpt(net, out / "ckpt_final.safetensors")
    total_h = (time.time() - t_start) / 3600
    print(
        f"{tag} DONE — {cycle} cycles | {rows_verified} rows | {total_h:.2f}h",
        flush=True,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Orkun 12h Kaggle training driver (2×T4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, help="Path to .safetensors checkpoint")
    ap.add_argument("--config", default="../Orkish/configs/waaagh_grot_80m.yaml")
    ap.add_argument("--orkish-repo", default="../Orkish")
    ap.add_argument("--out", default="runs/kaggle")
    ap.add_argument("--budget-hours", type=float, default=11.5,
                    help="Wall-clock hours before stopping (30 min buffer vs 12h limit)")
    ap.add_argument("--n-arms", type=int, default=0,
                    help="Arms to spawn (0 = one per CUDA device, min 1)")
    ap.add_argument("--n-goals", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--samples", type=int, default=16,
                    help="Rollouts per goal — all drawn in ONE batched forward (T4 occupancy)")
    ap.add_argument("--save-every-minutes", type=float, default=30.0,
                    help="Checkpoint interval in minutes")
    ap.add_argument("--soup-every-minutes", type=float, default=10.0,
                    help="Cross-arm weight-averaging (model soup) interval in minutes")
    ap.add_argument("--no-soup", action="store_true",
                    help="Disable model soup (keep arms fully independent)")
    ap.add_argument("--profile", action="store_true",
                    help="Print per-cycle phase timing (propose/rollout/train)")
    ap.add_argument("--parallel-workers", type=int, default=8,
                    help="CPU threads for parallel sandbox (oracle + ablation)")
    ap.add_argument("--anchor-eta", type=float, default=0.05,
                    help="AnchorReg pull strength (0=free, 1=frozen)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase-config", default=None,
                    help="Phase yaml (policy temps, lp window, epsilon); "
                         "defaults to configs/phase_a.yaml next to the package if present")
    args = ap.parse_args(argv)

    orkish_repo = Path(args.orkish_repo).resolve()
    _add_orkish_to_path(orkish_repo)
    model_cfg = yaml.safe_load(Path(args.config).read_text())["model"]
    if args.phase_config is not None:
        phase_path = Path(args.phase_config)
    else:
        phase_path = Path(__file__).resolve().parents[2] / "configs" / "phase_a.yaml"
    phase_cfg = _load_phase_cfg(phase_path)
    print(f"phase config: {phase_path if phase_path.is_file() else 'defaults (yaml missing)'} "
          f"→ {phase_cfg}", flush=True)
    out = Path(args.out)

    n_gpus = torch.cuda.device_count()
    n_arms = args.n_arms if args.n_arms > 0 else max(1, n_gpus)
    gpu_list = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cpu"]
    devices = [gpu_list[i % len(gpu_list)] for i in range(n_arms)]

    soup_on = (not args.no_soup) and n_arms > 1
    print(
        f"Orkun Kaggle driver: {n_arms} arm(s) on {devices}, "
        f"budget {args.budget_hours}h, {args.parallel_workers} sandbox workers, "
        f"samples/goal {args.samples}, soup {'on' if soup_on else 'off'}",
        flush=True,
    )

    shared_kwargs = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        model_cfg=model_cfg,
        orkish_repo=str(orkish_repo),
        budget_seconds=args.budget_hours * 3600.0,
        n_goals=args.n_goals,
        k=args.k,
        samples=args.samples,
        save_interval_seconds=args.save_every_minutes * 60.0,
        parallel_workers=args.parallel_workers,
        anchor_eta=args.anchor_eta,
        seed=args.seed,
        soup_dir=str(out / "soup"),
        soup_interval_seconds=args.soup_every_minutes * 60.0,
        soup_on=soup_on,
        profile_on=args.profile,
        phase_cfg=phase_cfg,
    )

    if n_arms == 1:
        # avoid spawn overhead; also friendlier to Jupyter / debug
        _arm_loop(0, devices[0], out_dir=str(out / "arm_0"), **shared_kwargs)
    else:
        # spawn = CUDA-safe (fork + CUDA = undefined behaviour)
        ctx = mp.get_context("spawn")
        procs = []
        for i, device in enumerate(devices):
            p = ctx.Process(
                target=_arm_loop,
                args=(i, device),
                kwargs={**shared_kwargs, "out_dir": str(out / f"arm_{i}")},
                daemon=False,
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    # merge loot from all arms
    merged = out / "loot_merged.jsonl"
    arm_loots = sorted(out.glob("arm_*/loot.jsonl"))
    if arm_loots:
        with merged.open("w") as fout:
            for path in arm_loots:
                fout.write(path.read_text())
        n_rows = sum(1 for _ in merged.open())
        print(f"Merged loot → {merged} ({n_rows} rows from {len(arm_loots)} arm(s))", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
