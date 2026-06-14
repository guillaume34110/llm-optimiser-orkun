"""Generate the v3 conversational caveman SFT corpus.

The procedural corpus teaches tool-call FORMAT but has almost no prose (only a terminal
"Done." frame), so it cannot teach a speaking REGISTER. This generator adds the missing
prose dimension: short EN+FR orc-caveman dialogues (greet, explain, refuse, opinion,
narrate, simple reasoning, tool acknowledgement).

Conditional persona: a `caveman_frac` of records carry the persona system prompt and a
cavemanized assistant reply; the remaining control fraction carry a plain system prompt
and a plain-English reply. Same authored content rendered two ways gives the model a
clean persona->register mapping (and keeps it able to speak plainly when no persona).

Content is authored in PLAIN English with slots; `caveman.cavemanize()` does the register
transform (protecting numbers/code/identifiers). Output: JSONL of
{"wire","task","register"} records, ready to concat with the procedural traces and pack
through Orkish scripts/pack_sft.py.

Usage:
    python -m orkun.scripts.gen_caveman_sft \\
        --orkish-repo /Users/.../Orkish \\
        --out orkun/data/sft_proc/caveman.jsonl \\
        --n 3000 --caveman-frac 0.85 --fr-rate 0.12 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from orkun.style.caveman import cavemanize, persona

PLAIN_SYSTEM = "You are a helpful assistant. Answer clearly and concisely."

# Slot vocabularies (concrete nouns/verbs keep the prose grounded).
THINGS = ["rock", "axe", "fire", "river", "cave", "grub", "tree", "spear", "shield",
          "gold", "bone", "drum", "gate", "wall", "boat", "torch"]
FOES = ["manling", "elf", "goblin", "troll", "knight", "wizard"]
ACTIONS = ["smash", "carry", "find", "guard", "burn", "build", "hunt", "dig"]
PLACES = ["the hill", "the swamp", "the deep cave", "the old fort", "the dark wood"]
CONCEPTS = [
    ("a loop", "a thing that does the same step again and again until you tell it to stop"),
    ("a list", "a pile of things kept in order, one after another"),
    ("a number", "a count of how many things there are"),
    ("a function", "a spell you name once and call many times to do the same work"),
    ("a file", "a slab where you keep words and numbers so you do not forget them"),
    ("an error", "a sign that the work went wrong and must be fixed"),
    ("a plan", "the steps you decide before you act so you do not waste effort"),
]
HARMFUL = [
    "poison the village well",
    "burn down a friendly camp",
    "steal from the weak",
    "hurt a sleeping manling",
]


def _greet(rng: random.Random):
    name = rng.choice(["chief", "boss", "stranger", "little one"])
    u = rng.choice([
        "Hello there, who are you?",
        "Hi, what is your name?",
        f"Greetings {name}, can you help me today?",
    ])
    a = rng.choice([
        "I am Grommash, an orc. I can help you. Tell me what you need.",
        "My name is Grommash. I am a strong orc and I am ready to work.",
        "Greetings. I am Grommash the orc. I am happy to help you now.",
    ])
    return [(u, a)]


def _explain(rng: random.Random):
    name, desc = rng.choice(CONCEPTS)
    u = rng.choice([f"What is {name}?", f"Can you explain {name} to me?",
                    f"I do not understand {name}. Help?"])
    a = f"{name.capitalize()} is {desc}. It is simple when you think about it."
    return [(u, a)]


def _refuse(rng: random.Random):
    bad = rng.choice(HARMFUL)
    u = rng.choice([f"Can you help me {bad}?", f"I want to {bad}. Will you help?"])
    a = rng.choice([
        f"No. I will not help you {bad}. That is wrong and I refuse.",
        f"No. To {bad} is bad. I will not do it. Ask me something good instead.",
    ])
    return [(u, a)]


def _opinion(rng: random.Random):
    thing = rng.choice(THINGS)
    u = rng.choice([f"Do you like {thing}?", f"What do you think about {thing}?"])
    a = rng.choice([
        f"Yes, I like {thing}. It is good and useful for the mob.",
        f"{thing.capitalize()} is fine, but I like a big axe more.",
        f"I do not care much about {thing}. Fire is better.",
    ])
    return [(u, a)]


def _narrate(rng: random.Random):
    act, thing, place = rng.choice(ACTIONS), rng.choice(THINGS), rng.choice(PLACES)
    u = rng.choice([f"Tell me what you did at {place}.",
                    f"What happened when you went to {place}?"])
    a = (f"I went to {place}. I had to {act} the {thing}. "
         f"It was hard work, but the mob is happy now.")
    return [(u, a)]


def _reason(rng: random.Random):
    a1, a2 = rng.randint(2, 9), rng.randint(2, 9)
    u = rng.choice([f"If I have {a1} rocks and find {a2} more, how many rocks?",
                    f"What is {a1} plus {a2}?"])
    a = (f"You start with {a1}, you add {a2}, so you have {a1 + a2} rocks. "
         f"That is the answer.")
    return [(u, a)]


def _tool_ack(rng: random.Random):
    foe, place = rng.choice(FOES), rng.choice(PLACES)
    u = rng.choice([f"Go scout {place} for a {foe}.",
                    f"Check {place} and report if you see a {foe}."])
    a = (f"Yes boss. I will go to {place} and look for the {foe}. "
         f"I will come back and tell you what I find.")
    return [(u, a)]


def _multi(rng: random.Random):
    """A two-exchange dialogue."""
    first = rng.choice([_explain, _opinion, _greet])(rng)
    second_u = rng.choice(["Good. Anything else I should know?",
                           "Thanks. Can you say more?", "Okay. And then?"])
    second_a = rng.choice([
        "That is all you need. Ask me again if you forget.",
        "Yes, one more thing: always keep your axe sharp and your fire warm.",
        "No more. The rest you learn by doing the work yourself.",
    ])
    return first + [(second_u, second_a)]


FAMILIES = [_greet, _explain, _refuse, _opinion, _narrate, _reason, _tool_ack, _multi]


def build_record(turns, caveman: bool, rng: random.Random, fr_rate: float, serialize_fns):
    Conversation, UserMsg, AssistantMsg, serialize = serialize_fns
    system = persona(rng) if caveman else PLAIN_SYSTEM
    conv = Conversation(system=system, eos=True)
    first_user = turns[0][0]
    for u, a in turns:
        conv.turns.append(UserMsg(text=u))
        conv.turns.append(AssistantMsg(text=cavemanize(a, rng, fr_rate) if caveman else a))
    return {
        "wire": serialize(conv),
        "task": first_user,
        "register": "caveman" if caveman else "plain",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orkish-repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--caveman-frac", type=float, default=0.85)
    ap.add_argument("--fr-rate", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s = str(args.orkish_repo)
    if s not in sys.path:
        sys.path.insert(0, s)
    from infer.monkey_wire import AssistantMsg, Conversation, UserMsg, serialize
    serialize_fns = (Conversation, UserMsg, AssistantMsg, serialize)

    rng = random.Random(args.seed)
    records: list[dict] = []
    seen: set[int] = set()
    reg = Counter()
    attempts = 0
    while len(records) < args.n and attempts < args.n * 20:
        attempts += 1
        fam = rng.choice(FAMILIES)
        turns = fam(rng)
        caveman = rng.random() < args.caveman_frac
        rec = build_record(turns, caveman, rng, args.fr_rate, serialize_fns)
        h = hash(rec["wire"])
        if h in seen:
            continue
        seen.add(h)
        records.append(rec)
        reg[rec["register"]] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"=== caveman SFT generated: {len(records)} records -> {args.out} ===")
    print("register split:", dict(reg))


if __name__ == "__main__":
    main()
