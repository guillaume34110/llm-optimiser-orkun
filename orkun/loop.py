"""kunnin_cycle — the Orkun loop: propose → curriculum → rollout → verify → causal → store → LP.

Phase A path: GoalGen proposes verifiable goals; the curriculum picks families by
learning progress; the policy rolls out against the verifier; a verified trajectory
is scored for per-action necessity by counterfactual ablation, stored, and LP is
updated. Phase B (flag `world_model_on=True`) adds: a curiosity-augmented curriculum
score, an optional planner rollout (imagination proposes, sandbox disposes), surprise
+ prediction-progress from the world-model, online WM finetune, an optional RLVR
policy-gradient step with necessity-weighted causal credit, and the extended store
row. Flag off ⇒ byte-identical to Phase A. Collaborators are injected for testing.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from infer.monkey_wire import AssistantMsg, ToolResult, parse, parse_calls
from orkun.eval.holdout import transitions_from_wire
from orkun.causal.counterfactual import ablation_necessity as _ablation_seq
from orkun.data.store import Store
from orkun.goals.generator import GoalGen, ProposedGoal
from orkun.train.reward import advantage, baseline, r_total


@dataclass
class CycleReport:
    n_attempts: int
    n_verified: int
    picked_families: list[str]


def _goal_for_family(goals: list[ProposedGoal], family: str) -> ProposedGoal | None:
    for g in goals:
        if g.family == family:
            return g
    return None


def _curiosity_over_results(wire: str, wm, curiosity) -> tuple[float, list[tuple[str, str]]]:
    """Mean surprise over the wire's real tool results + the (prefix, result) transitions.

    Transition reconstruction lives in `orkun.eval.holdout.transitions_from_wire` (single
    source of truth, shared with the Done-B calibration harness); here we score surprise
    over those transitions. Returns (mean_surprise, transitions).
    """
    transitions = transitions_from_wire(wire)
    surprises = [curiosity.surprise(wm, prefix, result) for prefix, result in transitions]
    mean_surprise = sum(surprises) / len(surprises) if surprises else 0.0
    return mean_surprise, transitions


def _group_rewards(gr, family: str, wm, curiosity, beta: float):
    """Per-sample (sample, surprise, transitions, curiosity, reward) for a whole group.

    Every sample is scored on its OWN wire, so the group carries a spread of rewards —
    the raw material for a group-relative (GRPO) baseline. Failed samples (no parsed
    calls) have an empty wire ⇒ zero surprise / zero graded ⇒ reward near zero, which
    is exactly what should drag the group mean down so genuine successes stand out.
    """
    rows = []
    for s in gr.samples:
        if s.wire:
            surprise, transitions = _curiosity_over_results(s.wire, wm, curiosity)
        else:
            surprise, transitions = 0.0, []
        cur = curiosity.curiosity(family, surprise)
        reward = r_total(s.graded, cur, beta)
        rows.append((s, surprise, transitions, cur, reward))
    return rows


def kunnin_cycle(
    goal_gen: GoalGen,
    lp,
    curriculum,
    policy,
    store: Store,
    families: list[str],
    n_goals: int = 16,
    k: int = 4,
    samples: int = 8,
    rng: random.Random | None = None,
    *,
    world_model_on: bool = False,
    wm=None,
    curiosity=None,
    reward_head=None,
    planner=None,
    wm_trainer=None,
    rlvr=None,
    tok=None,
    beta: float = 0.1,
    lam: float = 0.5,
    reward_history: list[float] | None = None,
    parallel_workers: int = 0,
    group=None,
    profile: dict | None = None,
    difficulty=None,
    max_ablations_per_goal: int = 4,
) -> CycleReport:
    rng = rng or random.Random()
    reward_history = reward_history if reward_history is not None else []

    def _tick(bucket: str, dt: float) -> None:
        if profile is not None:
            profile[bucket] = profile.get(bucket, 0.0) + dt

    if parallel_workers > 0:
        from orkun.parallel.pool import parallel_propose, parallel_ablation_necessity
        _ablation = lambda task, calls: parallel_ablation_necessity(task, calls, max_workers=parallel_workers)
    else:
        parallel_propose = None
        _ablation = _ablation_seq

    # 1. PROPOSE — at the tracker's per-family difficulty when one is wired
    # (difficulty=None keeps the historical context={} ⇒ GoalGen default level).
    context = difficulty.context() if difficulty is not None else {}
    _t = time.perf_counter()
    if parallel_workers > 0:
        goals = parallel_propose(goal_gen, context=context, n=n_goals, max_workers=parallel_workers)
    else:
        goals = goal_gen.propose(context=context, n=n_goals)
    _tick("propose", time.perf_counter() - _t)

    # 2. CURRICULUM (combined score when curiosity is wired)
    lp_scores = lp.predict(families)
    if world_model_on and curiosity is not None:
        scores = {f: lp_scores.get(f, 0.0) + lam * curiosity.curiosity(f, 0.0) for f in families}
    else:
        scores = lp_scores
    chosen = curriculum.pick(families, scores, k=k)

    n_attempts = 0
    n_verified = 0
    group_on = world_model_on and group is not None and wm is not None and curiosity is not None
    wm_batch: list[tuple[str, str]] = []   # transitions accumulated across the whole cycle
    rlvr_batch: list[dict] = []            # policy-gradient samples accumulated across the cycle
    for family in chosen:
        goal = _goal_for_family(goals, family)
        if goal is None:
            continue
        n_attempts += 1

        # GROUP PATH — batched rollout + group-relative advantage + cycle-batched updates.
        if group_on:
            _t = time.perf_counter()
            gr = group.rollout_group(goal.task, samples=samples)
            rows = _group_rewards(gr, family, wm, curiosity, beta)
            _tick("rollout", time.perf_counter() - _t)
            rewards = [r for *_, r in rows]
            base_g = sum(rewards) / len(rewards) if rewards else 0.0  # group-relative baseline
            mean_surprise = sum(sp for _, sp, *_ in rows) / len(rows) if rows else 0.0
            curiosity.observe(family, mean_surprise)
            # WM learns from EVERY real transition — success or not, a tool result is world truth.
            for _s, _sp, transitions, _cur, _r in rows:
                wm_batch.extend(transitions)

            success = bool(gr.passed)
            if success:
                n_verified += 1
                # Shared prompt context: every sample was drawn from the same wire, so encode once.
                prompt_wire = group._prompt_wire(goal.task) if hasattr(group, "_prompt_wire") else ""
                prefix_ids = tok.encode(prompt_wire, add_bos=False, add_eos=False) if tok is not None else []
                # Ablation is the expensive step (one sandbox run per call per passing sample),
                # so it is capped: only the top `max_ablations_per_goal` passing samples (by
                # reward, stable order) get a real counterfactual; the tail reuses the mean
                # ablated necessity as its RLVR weight. Rank 0 is the stored (best) sample, so
                # the loot always carries a true cf vector.
                passing = sorted((r for r in rows if r[0].passed), key=lambda r: r[4], reverse=True)
                best_cf = None
                nec_ws: list[float] = []
                for rank, (s, surprise, _transitions, cur, reward) in enumerate(passing):
                    adv = reward - base_g
                    if rank < max_ablations_per_goal:
                        _t = time.perf_counter()
                        cf = _ablation(goal.task, s.calls)
                        _tick("train", time.perf_counter() - _t)
                        nec_w = max(cf.necessity) if cf.necessity else 0.0
                        nec_ws.append(nec_w)
                        if best_cf is None:
                            best_cf = cf
                    else:
                        nec_w = sum(nec_ws) / len(nec_ws) if nec_ws else 0.0
                    # weight = necessity × advantage; gather only samples that move the policy.
                    if rlvr is not None and tok is not None and prefix_ids and nec_w * adv != 0.0:
                        action_ids = tok.encode(s.assistant_text, add_bos=False, add_eos=False)
                        if action_ids:
                            rlvr_batch.append({
                                "prefix_ids": prefix_ids, "action_ids": action_ids,
                                "advantage": adv, "necessity": nec_w,
                            })
                # 7. STORE — the best passing sample represents the goal in the loot.
                s, surprise, _transitions, cur, reward = passing[0]
                cf = best_cf
                reward_hat = reward_head.predict(s.wire) if reward_head is not None else 0.0
                reward_history.append(reward)
                store.add(task_id=goal.task.id, wire=s.wire, n_calls=s.n_calls,
                          necessity=cf.necessity, source="self_play",
                          graded=cf.base_graded, surprise=surprise,
                          reward_hat=reward_hat, curiosity=cur)
            lp.update(family, success)
            if difficulty is not None:
                difficulty.update(family, success)
            continue

        # 3-4. ROLLOUT + VERIFY (planner if wired, else Phase A policy)
        rollout_obj = planner if (world_model_on and planner is not None) else policy
        result = rollout_obj.rollout(goal.task, samples=samples)
        success = result is not None
        if success:
            n_verified += 1
            # 5. CAUSAL — re-derive the executed calls 1:1 with the wire's ToolResults.
            conv = parse(result.wire)
            n_steps = sum(1 for t in conv.turns if isinstance(t, ToolResult))
            assistant = next((t for t in conv.turns if isinstance(t, AssistantMsg)), None)
            known_tools = getattr(policy, "known_tools", None)
            calls = parse_calls(assistant.text, known_tools=known_tools) if assistant else []
            calls = calls[:n_steps]
            cf = _ablation(goal.task, calls)

            if world_model_on and wm is not None and curiosity is not None:
                # 5b. CURIOSITY — surprise over real results + online WM finetune.
                mean_surprise, transitions = _curiosity_over_results(result.wire, wm, curiosity)
                curiosity.observe(family, mean_surprise)
                cur = curiosity.curiosity(family, mean_surprise)
                reward_hat = reward_head.predict(result.wire) if reward_head is not None else 0.0
                if wm_trainer is not None:
                    wm_trainer.update(transitions)
                # 8. TRAIN — RLVR on r_total with necessity-weighted causal credit.
                if rlvr is not None and tok is not None and assistant is not None:
                    reward = r_total(cf.base_graded, cur, beta)
                    adv = advantage(reward, baseline(reward_history))  # baseline from prior history, before appending
                    reward_history.append(reward)
                    nec_w = max(cf.necessity) if cf.necessity else 0.0
                    # REINFORCE is unbiased only when logπ(a) is scored on the SAME context the
                    # policy sampled from: the full prompt wire (control tokens included), not raw user text.
                    prompt_wire = policy._prompt_wire(goal.task) if hasattr(policy, "_prompt_wire") else ""
                    prefix_ids = tok.encode(prompt_wire, add_bos=False, add_eos=False)
                    action_ids = tok.encode(assistant.text, add_bos=False, add_eos=False)
                    if prefix_ids and action_ids:
                        rlvr.update([{
                            "prefix_ids": prefix_ids, "action_ids": action_ids,
                            "advantage": adv, "necessity": nec_w,
                        }])
                # 7. STORE — Phase A fields + Phase B fields.
                store.add(task_id=goal.task.id, wire=result.wire, n_calls=result.n_calls,
                          necessity=cf.necessity, source="self_play",
                          graded=cf.base_graded, surprise=mean_surprise,
                          reward_hat=reward_hat, curiosity=cur)
            else:
                # Phase A store call — byte-identical to Phase A.
                store.add(task_id=goal.task.id, wire=result.wire, n_calls=result.n_calls,
                          necessity=cf.necessity, source="self_play")
        # 9. LP UPDATE (+ difficulty progression when a tracker is wired)
        lp.update(family, success)
        if difficulty is not None:
            difficulty.update(family, success)

    # GROUP PATH — flush the cycle's accumulated updates in ONE batched step each:
    # WMTrainer sums its loss over the transition list; RLVRTrainer sums -weight·logπ
    # over the sample list. One update/cycle instead of one/goal kills the batch-1
    # REINFORCE variance and amortises the optimiser step.
    if group_on:
        _t = time.perf_counter()
        if wm_trainer is not None and wm_batch:
            wm_trainer.update(wm_batch)
        if rlvr is not None and rlvr_batch:
            rlvr.update(rlvr_batch)
        _tick("train", time.perf_counter() - _t)

    return CycleReport(n_attempts=n_attempts, n_verified=n_verified, picked_families=chosen)
