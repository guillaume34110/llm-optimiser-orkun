"""Parametric goal families — each emits a verifiable Task + a solving oracle.

A goal is an Orkish `Task` (prompt + checks). `difficulty` (0,1,2,...) scales the
demand. `oracle(task)` returns the correct tool calls; the generator uses it to
prove a goal is solvable before admitting it (guarantees a well-defined reward and
keeps infeasible LP≈0 goals out of the curriculum). Families cover the executor
surface: write_file, py_run (fix token / arithmetic / JSON transform) and basic
logic games (arith eval / sequence continuation / boolean SAT), the latter all
answering through an ANSWER= stdout sentinel verified by stdout_contains.

Two invariants every oracle must respect (V2 post-mortem):

* SINGLE-SHOT: GroupRollout emits ALL calls in one assistant turn before any
  ToolResult comes back. An oracle may only reference what is in the prompt or
  what the sandbox computes at EXECUTION time — never a value that only exists
  in a tool result or in the hidden seed.
* COPY, DON'T STASH: the rendered gold call must be derivable from the prompt by
  copying. A gold like `print(285)` where 285 = a+b teaches the model to invent
  constants (the V2 operand-binding failure); the gold must be `print(a + b)`
  with a and b copied verbatim, letting the sandbox do the computing.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from infer.monkey_wire import ToolCall
from scripts.verifier import Task

_WORDS = ["WAAAGH", "stomp", "boyz", "dakka", "loota", "grot", "mek", "nob"]


def _stable_id(*parts) -> str:
    """Process-stable content hash for task ids.

    Built-in `hash()` is salted per process (PYTHONHASHSEED), so ids built from it
    differ across arms/runs and `% 10000` collides distinct tasks — both poison the
    by-task holdout grouping. sha256 over the content is reproducible everywhere;
    12 hex chars keep collisions negligible at loop scale.
    """
    blob = "|".join(str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class GoalFamily:
    name: str = "base"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        raise NotImplementedError

    def oracle(self, task: Task) -> list[ToolCall]:
        raise NotImplementedError


@dataclass
class WriteFileFamily(GoalFamily):
    name: str = "write_file"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        n_words = 1 + difficulty * 2
        # numeric token: high-entropy span the model can only get right by copying
        content = " ".join(rng.choice(_WORDS) for _ in range(n_words)) + f" {rng.randint(0, 999999)}"
        path = f"out_{rng.randint(0, 9999)}.txt"
        return Task(
            id=f"{self.name}-{path}",
            prompt=f"Create a file {path} containing exactly: {content}",
            checks=[
                {"type": "file_equals", "path": path, "content": content, "strip": True},
                {"type": "no_error"},
            ],
            seed={},
            clean=True,
        )

    def oracle(self, task: Task) -> list[ToolCall]:
        spec = next(c for c in task.checks if c["type"] == "file_equals")
        return [ToolCall("write_file", {"path": spec["path"], "content": spec["content"] + "\n"})]


@dataclass
class FixTokenFamily(GoalFamily):
    name: str = "fix_token"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        n = 1 + difficulty
        lines = [f"line {i} TODO marker" for i in range(n)] + ["tail line"]
        body = "\n".join(lines) + "\n"
        path = f"notes_{rng.randint(0, 9999)}.txt"
        return Task(
            id=f"{self.name}-{path}",
            prompt=f"In {path} replace every TODO with DONE.",
            checks=[
                {"type": "file_contains", "path": path, "substr": "DONE"},
                {"type": "file_not_contains", "path": path, "substr": "TODO"},
            ],
            seed={path: body},
            clean=True,
        )

    def oracle(self, task: Task) -> list[ToolCall]:
        path = next(c for c in task.checks if c["type"] == "file_contains")["path"]
        code = (
            f"p={path!r}\n"
            "s=open(p).read().replace('TODO','DONE')\n"
            "open(p,'w').write(s)\n"
            "print('ok')"
        )
        return [ToolCall("py_run", {"code": code})]


@dataclass
class ComputeFamily(GoalFamily):
    """Print the sum of two integers; the sandbox computes, the model copies.

    The oracle is `print(a + b)` with a and b copied from the prompt — NOT
    `print(total)`. A `print(<total>)` gold (the V2 corpus) supervises a constant
    that appears nowhere in the prompt, which actively teaches the model to invent
    operands instead of binding them.
    """

    name: str = "compute"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        a = rng.randint(1, 10 ** (difficulty + 2))
        b = rng.randint(1, 10 ** (difficulty + 2))
        total = a + b
        return Task(
            id=f"{self.name}-{a}-{b}",
            prompt=f"Use python to print the sum of {a} and {b}.",
            checks=[
                {"type": "stdout_contains", "substr": str(total)},
                {"type": "called_tool", "name": "py_run"},
            ],
            clean=True,
        )

    def oracle(self, task: Task) -> list[ToolCall]:
        # copy both operands from the public prompt; the sandbox does the addition
        rest = task.prompt.split("sum of ", 1)[1]
        a = rest.split(" and ", 1)[0]
        b = rest.split(" and ", 1)[1].rstrip(".")
        return [ToolCall("py_run", {"code": f"print({a} + {b})"})]


@dataclass
class JsonTransformFamily(GoalFamily):
    name: str = "json_transform"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        key = rng.choice(["name", "kind", "tag"])
        # digit suffix: the value can't be produced from the 8-word vocab alone
        new_val = f"{rng.choice(_WORDS)}{rng.randint(0, 99)}"
        path = f"cfg_{rng.randint(0, 9999)}.json"
        body = f'{{"{key}": "old", "n": {difficulty}}}\n'
        return Task(
            id=f"{self.name}-{path}",
            prompt=f'In {path}, set the JSON field "{key}" to "{new_val}".',
            checks=[
                {"type": "file_contains", "path": path, "substr": f'"{key}": "{new_val}"'},
                {"type": "file_not_contains", "path": path, "substr": '"old"'},
            ],
            seed={path: body},
            clean=True,
        )

    def oracle(self, task: Task) -> list[ToolCall]:
        spec = next(c for c in task.checks if c["type"] == "file_contains")
        path = spec["path"]
        code = (
            "import json\n"
            f"p={path!r}\n"
            "d=json.load(open(p))\n"
            f"d[{_key_from(spec['substr'])!r}]={_val_from(spec['substr'])!r}\n"
            "json.dump(d, open(p,'w'))\n"
            "print('ok')"
        )
        return [ToolCall("py_run", {"code": code})]


@dataclass
class ArithFamily(GoalFamily):
    """Evaluate an integer arithmetic expression; answer via the ANSWER= sentinel.

    Generate-backwards: the expression is built from sampled operands/ops and its
    value is computed at sample time, so the check (`stdout_contains "ANSWER=<v>"`)
    and the oracle are always correct by construction. + - only → integer result,
    no division/rounding ambiguity. Multiplication is excluded on purpose: the wire
    parser (`parse_calls`) strips `*` as markdown, so any tool call carrying a literal
    `*` is corrupted before execution and the task becomes unsolvable. Difficulty adds
    operands and magnitude instead.
    """

    name: str = "arith"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        n_ops = 1 + difficulty            # d0: 2 operands, d2: 4 operands
        ops = ["+", "-"]                  # no `*`: parse_calls strips it as markdown
        hi = 99 * (difficulty + 1)        # multi-digit operands: copying, not guessing
        expr = str(rng.randint(1, hi))
        for _ in range(n_ops):
            expr += f" {rng.choice(ops)} {rng.randint(1, hi)}"
        value = eval(expr)            # noqa: S307 — operands/ops are sampled, not user input
        return Task(
            id=f"{self.name}-{_stable_id(expr)}",
            prompt=f"Evaluate {expr} with python and print ANSWER=<result>.",
            checks=[
                {"type": "stdout_contains", "substr": f"ANSWER={value}"},
                {"type": "called_tool", "name": "py_run"},
            ],
            clean=True,
        )

    def oracle(self, task: Task) -> list[ToolCall]:
        # recover the expression from the public prompt — no answer stashed in the sandbox
        expr = task.prompt.split("Evaluate ", 1)[1].split(" with python", 1)[0]
        return [ToolCall("py_run", {"code": f"print(f'ANSWER={{{expr}}}')"})]


@dataclass
class SequenceFamily(GoalFamily):
    """Continue an arithmetic sequence; answer via ANSWER=.

    Generate-backwards from a known rule (start + step). The shown terms determine the
    rule uniquely, so the next term is well-defined and the oracle rederives it from the
    term list. Arithmetic only: a geometric rule needs `*` in the solving code, which the
    wire parser (`parse_calls`) strips as markdown — corrupting the call. Difficulty adds
    shown terms and a larger step instead.
    """

    name: str = "sequence"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        n = 3 + difficulty                       # terms shown
        start = rng.randint(1, 40)               # multi-digit terms: copy pressure
        step = rng.randint(1, 9 + 3 * difficulty)  # arithmetic only (no `*` in oracle code)
        terms = [start + step * i for i in range(n)]
        nxt = terms[-1] + step
        shown = ", ".join(str(t) for t in terms)
        return Task(
            id=f"{self.name}-{_stable_id(shown)}",
            prompt=f"Find the next number in this sequence and print ANSWER=<n>: {shown}",
            checks=[
                {"type": "stdout_contains", "substr": f"ANSWER={nxt}"},
                {"type": "called_tool", "name": "py_run"},
            ],
            clean=True,
        )

    def oracle(self, task: Task) -> list[ToolCall]:
        # the shown terms (after the colon) determine the constant difference; solve it
        # in the sandbox. Arithmetic only — no `*` (the wire parser would strip it).
        shown = task.prompt.split(": ", 1)[1]
        code = (
            f"t=[{shown}]\n"
            "d=t[1]-t[0]\n"
            "nxt=t[-1]+d\n"
            "print(f'ANSWER={nxt}')"
        )
        return [ToolCall("py_run", {"code": code})]


@dataclass
class BoolSatFamily(GoalFamily):
    """Decide satisfiability of a small CNF formula; answer ANSWER=SAT / ANSWER=UNSAT.

    The label is the GROUND TRUTH from a brute-force over all 2^k assignments at
    sample time (k<=4 → at most 16), not from how the formula was built — so the
    decision is exact whichever way the model reasons. Construction biases the mix:
    ~half are built from a planted assignment (SAT), ~half get a planted x AND not-x
    contradiction (UNSAT), but brute force is the source of truth. The oracle runs the
    same enumeration in the sandbox (no imports — a bitmask loop).
    """

    name: str = "bool_sat"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        k = 2 + difficulty                       # vars: d0=2 .. d2=4
        n_clauses = 2 + difficulty * 2
        assign = [rng.random() < 0.5 for _ in range(k)]
        clauses: list[list[tuple[int, bool]]] = []
        for _ in range(n_clauses):
            size = rng.randint(1, min(3, k))
            vs = rng.sample(range(k), size)
            cl: list[tuple[int, bool]] = []
            for i, v in enumerate(vs):
                # plant at least one literal satisfied by `assign` (first literal)
                neg = (not assign[v]) if i == 0 else (rng.random() < 0.5)
                cl.append((v, neg))
            clauses.append(cl)
        if rng.random() < 0.45:                  # inject a contradiction → UNSAT
            v = rng.randrange(k)
            clauses.append([(v, False)])
            clauses.append([(v, True)])
        label = "SAT" if _cnf_sat(clauses, k) else "UNSAT"
        formula = " and ".join(
            "(" + " or ".join(("not " if neg else "") + f"x{v}" for v, neg in cl) + ")"
            for cl in clauses
        )
        return Task(
            id=f"{self.name}-{_stable_id(formula)}",
            prompt=(
                f"Is this boolean formula satisfiable? Use python to decide and print "
                f"ANSWER=SAT or ANSWER=UNSAT: {formula}"
            ),
            checks=[
                {"type": "stdout_contains", "substr": f"ANSWER={label}"},
                {"type": "called_tool", "name": "py_run"},
            ],
            clean=True,
        )

    def oracle(self, task: Task) -> list[ToolCall]:
        # solve the formula from the public prompt: collect its xN vars, brute-force
        # all assignments, eval the boolean expression. No answer stashed in the sandbox.
        formula = task.prompt.split(": ", 1)[1]
        code = (
            f"f={formula!r}\n"
            "idx=set();i=0\n"
            "while i<len(f):\n"
            "    if f[i]=='x' and i+1<len(f) and f[i+1].isdigit():\n"
            "        j=i+1\n"
            "        while j<len(f) and f[j].isdigit(): j+=1\n"
            "        idx.add(int(f[i+1:j]));i=j\n"
            "    else: i+=1\n"
            "ks=sorted(idx);sat=False\n"
            "for m in range(1<<len(ks)):\n"
            "    env={f'x{v}':bool((m>>p)&1) for p,v in enumerate(ks)}\n"
            "    if eval(f,{},env): sat=True;break\n"
            "print('ANSWER='+('SAT' if sat else 'UNSAT'))"
        )
        return [ToolCall("py_run", {"code": code})]


def _cnf_sat(clauses: list[list[tuple[int, bool]]], k: int) -> bool:
    """Brute-force satisfiability over 2^k assignments (k small)."""
    for m in range(1 << k):
        a = [(m >> i) & 1 for i in range(k)]
        if all(any((a[v] == 1) != neg for v, neg in cl) for cl in clauses):
            return True
    return False


@dataclass
class PipelineFamily(GoalFamily):
    """A genuinely multi-step task: read an input file, transform it, write the result.

    Single-action families make necessity degenerate — one call, ablate it, the goal
    fails, necessity is always 1.0, so the necessity weight carries no information. This
    family forces a >=2-call trajectory (a `min_calls` check plus `called_tool` on both
    read_file and py_run), so the counterfactual ablation produces a real per-call
    necessity vector.

    V3 redesign: the rollout is SINGLE-SHOT (all calls emitted before any ToolResult),
    so the old gold `write_file(content=str(n+s))` required emitting a value (N) the
    model had never seen — unsolvable by construction, hard 0.75 ceiling, and the SFT
    trace supervised a magic constant. Now the transform happens INSIDE py_run at
    execution time: the emitted code only copies `s` from the prompt and reads N from
    input.txt when the sandbox runs it. Fully learnable, still >=2 calls.
    """

    name: str = "pipeline"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        n = rng.randint(1, 200 * (difficulty + 1))
        s = rng.randint(1, 99 * (difficulty + 1))
        result = n + s
        return Task(
            id=f"{self.name}-{_stable_id(n, s)}",
            prompt=(
                f"input.txt holds an integer N. Read input.txt with read_file, then use "
                f"py_run to add {s} to N and write the sum to result.txt."
            ),
            seed={"input.txt": str(n)},
            checks=[
                {"type": "file_equals", "path": "result.txt", "content": str(result), "strip": True},
                {"type": "called_tool", "name": "read_file"},
                {"type": "called_tool", "name": "py_run"},
                {"type": "min_calls", "n": 2},
            ],
            clean=True,
        )

    def oracle(self, task: Task) -> list[ToolCall]:
        # only `s` is copied from the prompt; N stays in the sandbox and is read at
        # execution time — nothing in the gold text encodes the hidden value.
        s = int(task.prompt.split("add ", 1)[1].split(" to N", 1)[0])
        code = (
            "n=int(open('input.txt').read())\n"
            f"open('result.txt','w').write(str(n+{s}))\n"
            "print('ok')"
        )
        return [
            ToolCall("read_file", {"path": "input.txt"}),
            ToolCall("py_run", {"code": code}),
        ]


@dataclass
class EchoFamily(GoalFamily):
    """Pure copy task: reproduce a high-entropy payload from the prompt, verbatim.

    The V3 anti-template-memorization curriculum. The payload is random alphanumeric
    (half the time digits-only), so the loss on the answer span cannot be reduced by
    emitting modal tokens — only by attending to and copying the prompt. Nearly every
    supervised token in the gold is an operand token, which concentrates gradient on
    the binding behaviour the other families exercise only sparsely. Charset is
    [A-Za-z0-9] only: no `*` (parse_calls strips it) and no quoting hazards.
    """

    name: str = "echo"

    _ALPHA = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def sample(self, rng: random.Random, difficulty: int) -> Task:
        n = 4 + 4 * difficulty                   # payload chars: d0=4 .. d3=16
        if rng.random() < 0.5:                   # digits-only half: where V2 failed hardest
            payload = "".join(rng.choice("0123456789") for _ in range(n))
        else:
            payload = "".join(rng.choice(self._ALPHA) for _ in range(n))
        return Task(
            id=f"{self.name}-{_stable_id(payload)}",
            prompt=f"Print exactly ANSWER={payload} using python.",
            checks=[
                {"type": "stdout_contains", "substr": f"ANSWER={payload}"},
                {"type": "called_tool", "name": "py_run"},
            ],
            clean=True,
        )

    def oracle(self, task: Task) -> list[ToolCall]:
        payload = task.prompt.split("ANSWER=", 1)[1].split(" using", 1)[0]
        return [ToolCall("py_run", {"code": f"print('ANSWER={payload}')"})]


def _key_from(substr: str) -> str:
    # substr is '"<key>": "<val>"'
    return substr.split(":")[0].strip().strip('"')


def _val_from(substr: str) -> str:
    return substr.split(":", 1)[1].strip().strip('"')


ALL_FAMILIES: list[GoalFamily] = [
    WriteFileFamily(),
    FixTokenFamily(),
    ComputeFamily(),
    JsonTransformFamily(),
    ArithFamily(),
    SequenceFamily(),
    BoolSatFamily(),
    PipelineFamily(),
    EchoFamily(),
]
