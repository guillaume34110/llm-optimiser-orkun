"""GroupRollout — Orkun-side batched group rollout exposing per-sample rewards.

Orkish `play_task` rejection-samples and returns only the FIRST verified trajectory,
which throws away exactly the signal a group-relative policy gradient (GRPO) needs:
the reward of every sample. This rolls the loop ourselves, entirely Orkun-side:

  * one batched forward — `sample_batch` generates all N continuations of the task in
    a single prefill + shared decode loop (the big T4 occupancy win), and
  * one fresh-sandbox verification per sample via `run_fresh`, keeping the graded
    reward and executed calls of EACH sample, not just the first that passes.

The wire of a sample is built the same way Orkish renders a verified trajectory
(prompt → assistant reply → one ToolResult turn per executed call), so downstream
consumers (curiosity transitions, reward head, store) are unchanged. Nothing here
edits Orkish; it only imports the shared wire/verifier primitives.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from infer.monkey_wire import AssistantMsg, Conversation, ToolResult, UserMsg, parse_calls, serialize
from scripts.verifier import Task

from orkun.policy.prompting import prompt_wire
from orkun.policy.sampler import sample_batch
from orkun.world.verifier import run_fresh

_SPECIAL_RE = re.compile(r"<\|([a-z_]+)\|>")
_STOP_STRINGS = ("<|eos|>", "<|user|>")


@dataclass
class SampleResult:
    """One rollout of a task: its decoded reply, executed calls, graded reward, wire."""
    assistant_text: str
    calls: list  # list[ToolCall] actually executed (truncated to the verified steps)
    graded: float
    passed: bool
    n_calls: int
    wire: str


@dataclass
class GroupResult:
    """All N samples for a single task; `passed` are the verified ones."""
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def passed(self) -> list[SampleResult]:
        return [s for s in self.samples if s.passed]

    @property
    def best(self) -> SampleResult | None:
        """Highest-graded sample (ties broken toward verified), or None if empty."""
        if not self.samples:
            return None
        return max(self.samples, key=lambda s: (s.graded, s.passed))


class GroupRollout:
    def __init__(
        self,
        net,
        tok,
        system: str | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new: int = 256,
        known_tools=None,
        base_seed: int = 0,
    ):
        self.net = net
        self.tok = tok
        self.system = system
        self.temperature = temperature
        self.top_p = top_p
        self.max_new = max_new
        self.known_tools = known_tools
        self.base_seed = base_seed
        self._calls = 0  # advances the sampling seed each call so groups vary across cycles

    def _prompt_wire(self, task: Task) -> str:
        return prompt_wire(task, self.system)

    def _stop_ids(self) -> set[int]:
        ids: set[int] = set()
        for s in _STOP_STRINGS:
            m = _SPECIAL_RE.fullmatch(s)
            if m and m.group(1) in self.tok.specials:
                ids.add(self.tok.specials[m.group(1)])
        return ids

    def _decode(self, ids: list[int]) -> str:
        text = self.tok.decode(ids, skip_special=True)
        for s in _STOP_STRINGS:  # safety net for multi-token / non-special stops
            i = text.find(s)
            if i != -1:
                text = text[:i]
        return text

    @staticmethod
    def _build_wire(task: Task, assistant_text: str, steps) -> str:
        conv = Conversation(system=task.system, eos=True)
        conv.turns.append(UserMsg(text=task.prompt))
        conv.turns.append(AssistantMsg(text=assistant_text))
        for st in steps:
            conv.turns.append(
                ToolResult(name=st.call.name, payload_str=json.dumps(st.result, ensure_ascii=False))
            )
        return serialize(conv)

    def rollout_group(self, task: Task, samples: int = 8, seed: int | None = None) -> GroupResult:
        """Generate `samples` continuations in one batched forward, verify each, keep all."""
        if seed is None:
            seed = self.base_seed + self._calls
            self._calls += 1
        prompt_ids = self.tok.encode(self._prompt_wire(task), add_bos=False, add_eos=False)
        batch_ids = sample_batch(
            self.net,
            prompt_ids,
            n=samples,
            max_new=self.max_new,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_ids=self._stop_ids(),
            seed=seed,
        )
        out: list[SampleResult] = []
        for ids in batch_ids:
            text = self._decode(ids)
            calls = parse_calls(text, known_tools=self.known_tools)
            if not calls:
                out.append(SampleResult(text, [], 0.0, False, 0, ""))
                continue
            r = run_fresh(task, calls)
            wire = self._build_wire(task, text, r.steps)
            out.append(
                SampleResult(text, calls[: len(r.steps)], r.graded, r.passed, len(r.steps), wire)
            )
        return GroupResult(samples=out)
