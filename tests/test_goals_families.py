import random

from infer.monkey_wire import ToolCall
from scripts.verifier import Task
from orkun.goals.families import (
    ALL_FAMILIES,
    ArithFamily,
    BoolSatFamily,
    ComputeFamily,
    EchoFamily,
    FixTokenFamily,
    PipelineFamily,
    SequenceFamily,
    WriteFileFamily,
)
from orkun.world.verifier import run_fresh


def test_every_family_oracle_solves_its_own_task():
    rng = random.Random(0)
    for family in ALL_FAMILIES:
        for difficulty in (0, 1, 2):
            task = family.sample(rng, difficulty)
            assert isinstance(task, Task)
            assert task.checks, "task must carry at least one check"
            calls = family.oracle(task)
            assert all(isinstance(c, ToolCall) for c in calls)
            r = run_fresh(task, calls)
            assert r.passed, f"{family.name} d={difficulty} oracle failed: {r.failed_checks}"


def test_difficulty_scales_write_file_length():
    rng = random.Random(1)
    short = WriteFileFamily().sample(rng, difficulty=0)
    long = WriteFileFamily().sample(rng, difficulty=2)
    # difficulty 2 demands a longer exact-content payload than difficulty 0
    short_len = next(c["content"] for c in short.checks if c["type"] == "file_equals")
    long_len = next(c["content"] for c in long.checks if c["type"] == "file_equals")
    assert len(long_len) > len(short_len)


def test_family_names_unique():
    names = [f.name for f in ALL_FAMILIES]
    assert len(names) == len(set(names))


def test_logic_families_registered():
    names = {f.name for f in ALL_FAMILIES}
    assert {"arith", "sequence", "bool_sat", "pipeline", "echo"} <= names


def test_pipeline_is_multi_step_with_informative_necessity():
    """The whole point of pipeline: a >=2-call solution whose necessity is a real vector."""
    from orkun.causal.counterfactual import ablation_necessity

    fam = PipelineFamily()
    task = fam.sample(random.Random(0), difficulty=1)
    calls = fam.oracle(task)
    assert len(calls) >= 2
    cf = ablation_necessity(task, calls)
    assert len(cf.necessity) == len(calls)        # per-call vector, not a degenerate scalar
    assert all(nec > 0.0 for nec in cf.necessity)  # every step is required to pass all checks
    # read vs write contribute differently → necessity actually discriminates between steps
    assert len(set(round(n, 6) for n in cf.necessity)) > 1


def test_arith_difficulty_scales_operand_count():
    rng = random.Random(2)
    easy = ArithFamily().sample(rng, difficulty=0)
    hard = ArithFamily().sample(rng, difficulty=2)
    # more operands at higher difficulty → a longer prompt expression
    assert len(hard.prompt) > len(easy.prompt)


def test_logic_answers_use_sentinel():
    """Every logic family verifies via the ANSWER= sentinel, never a bare value."""
    rng = random.Random(3)
    for family in (ArithFamily(), SequenceFamily(), BoolSatFamily()):
        task = family.sample(rng, difficulty=1)
        substr = next(c["substr"] for c in task.checks if c["type"] == "stdout_contains")
        assert substr.startswith("ANSWER=")


def test_oracles_copy_operands_never_stash_answers():
    """V2 post-mortem invariant: every numeric value in a gold call's text must be
    copyable from the prompt (or be a structural constant), never a derived answer.
    compute: gold is `print(a + b)`, not `print(total)`. pipeline: the hidden N and
    the result n+s never appear in the gold text (the sandbox computes at exec time)."""
    rng = random.Random(7)

    task = ComputeFamily().sample(rng, difficulty=1)
    code = ComputeFamily().oracle(task)[0].args["code"]
    a, rest = task.prompt.split("sum of ", 1)[1].split(" and ", 1)
    b = rest.rstrip(".")
    assert a in code and b in code
    total = next(c["substr"] for c in task.checks if c["type"] == "stdout_contains")
    assert total not in code.replace(a, "").replace(b, "")

    task = PipelineFamily().sample(rng, difficulty=1)
    calls = PipelineFamily().oracle(task)
    n = task.seed["input.txt"]
    result = next(c["content"] for c in task.checks if c["type"] == "file_equals")
    rendered = " ".join(str(c.args) for c in calls)
    assert f"str(n+" in rendered          # transform happens at execution time
    assert result not in rendered          # the answer is never in the gold text
    assert n not in rendered.replace(result, "")


def test_pipeline_is_solvable_single_shot():
    """The rollout emits all calls before seeing any ToolResult — the gold must not
    require knowledge of the seed file's content at generation time."""
    fam = PipelineFamily()
    rng = random.Random(11)
    for d in (0, 1, 2):
        task = fam.sample(rng, d)
        for call in fam.oracle(task):
            # nothing in the emitted text depends on the hidden N
            assert task.seed["input.txt"] not in str(call.args.values())


def test_echo_payload_is_high_entropy_and_copyable():
    fam = EchoFamily()
    rng = random.Random(5)
    payloads = set()
    for _ in range(30):
        task = fam.sample(rng, difficulty=1)
        payload = task.prompt.split("ANSWER=", 1)[1].split(" using", 1)[0]
        payloads.add(payload)
        assert payload.isalnum() and "*" not in payload   # parse_calls-safe charset
        code = fam.oracle(task)[0].args["code"]
        assert payload in code                            # gold copies the payload
    assert len(payloads) == 30                            # no modal payload to memorize


def test_echo_difficulty_scales_payload_length():
    fam = EchoFamily()
    rng = random.Random(3)
    p0 = fam.sample(rng, 0).prompt.split("ANSWER=", 1)[1].split(" using", 1)[0]
    p3 = fam.sample(rng, 3).prompt.split("ANSWER=", 1)[1].split(" using", 1)[0]
    assert len(p3) > len(p0)


def test_bool_sat_emits_both_outcomes():
    """Across seeds the family must produce both SAT and UNSAT, else it is degenerate."""
    fam = BoolSatFamily()
    labels = set()
    for s in range(40):
        task = fam.sample(random.Random(s), difficulty=1)
        substr = next(c["substr"] for c in task.checks if c["type"] == "stdout_contains")
        labels.add(substr)
    assert {"ANSWER=SAT", "ANSWER=UNSAT"} <= labels
