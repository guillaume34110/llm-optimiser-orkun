"""Orc-caveman register: renderer, lexicon, persona — for the v3 stylistic SFT corpus.

The model substrate (Orkish 80M) learns a *conditional* low-vocabulary orc pidgin:
English-dominant, with a light, controlled French sprinkle. The register is carried
into the corpus two ways:

  * **persona system prompt** — monkey_wire has no `<|persona|>` token in its wire
    format (it only emits bos/system/user/assistant/eos), so the persona is prepended
    to `conv.system`. Records WITH it teach caveman; a control fraction WITHOUT it keeps
    the model able to speak plainly. This is what makes the style conditional, not baked.
  * **caveman prose** — conversational turns (authored caveman templates) and the
    terminal report frame of procedural traces.

`cavemanize()` is a deterministic rule transform (drop articles/auxiliaries, map
pronouns, substitute to a small core lexicon, inject FR at a controlled rate). It is
applied ONLY to natural-language prose. It must NEVER touch:
  * `CALL: tool(args)` lines or any tool argument,
  * numbers, code spans, quoted payloads, tool/identifier tokens.
Those are masked out before transform and restored after, so structured content
round-trips byte-for-byte. If the register bleeds into a CALL line the wire parser
(`ast.literal_eval`) breaks and RLVR reward collapses — hence the hard protection.
"""
from __future__ import annotations

import random
import re

# ---------------------------------------------------------------------------
# Persona (prepended to conv.system on the caveman fraction)

PERSONA_SYSTEM = (
    "You Grommash. You orc. You talk short, few word, no fancy. "
    "Drop 'the' and 'a'. Use small word. Say truth plain. "
    "Sometime french word slip in: oui, non, gros, feu, ami. "
    "You still do tool when ask. Talk orc, think clear."
)

# A few surface variants so the persona is not a single memorised string.
PERSONA_VARIANTS = (
    PERSONA_SYSTEM,
    (
        "You big orc, name Grommash. Speak caveman: tiny word, no 'the', no 'a'. "
        "Short. Clear. Drop french word time to time: oui, non, gros, ami, feu. "
        "Do tool job when boss ask. Talk dumb-sound, work smart."
    ),
    (
        "Orc speak only. You Grommash, war-mind. Few word. No fancy grammar. "
        "Cut 'the', cut 'a'. Mix small french: oui, non, gros, feu. "
        "Still call tool right when need. Brain sharp, mouth simple."
    ),
)

# ---------------------------------------------------------------------------
# Lexicon

ARTICLES = frozenset({"a", "an", "the"})

# Auxiliaries / copulas dropped to make pidgin ("I am ready" -> "Me ready").
DROP_WORDS = frozenset({
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doth",
    "has", "have", "had",
    "will", "shall", "would", "should", "could", "can", "may", "might", "must",
    "of",
})

# Pronoun / determiner remap to the orc register.
PRONOUN = {
    "i": "Me", "me": "Me", "my": "Me", "mine": "Me", "myself": "Me",
    "we": "Mob", "our": "Mob", "ours": "Mob", "us": "Mob", "ourselves": "Mob",
    "you": "You", "your": "You", "yours": "You", "yourself": "You",
    "they": "Them", "their": "Them", "them": "Them", "theirs": "Them",
    "he": "Him", "his": "Him", "him": "Him",
    "she": "Her", "her": "Her", "hers": "Her",
    "it": "It", "its": "It",
}

# Map verbose words to the small core lexicon.
SUBST = {
    "large": "big", "huge": "big", "enormous": "big", "giant": "big",
    "great": "big", "massive": "big",
    "small": "tiny", "little": "tiny", "minor": "tiny",
    "destroy": "smash", "demolish": "smash", "break": "smash", "attack": "smash",
    "kill": "smash", "defeat": "smash",
    "understand": "get", "comprehend": "get", "realize": "get", "know": "know",
    "problem": "trouble", "issue": "trouble", "error": "bad", "failure": "bad",
    "failed": "bad", "wrong": "bad", "incorrect": "bad", "mistake": "bad",
    "success": "good", "correct": "good", "succeeded": "good", "okay": "good",
    "ok": "good", "fine": "good", "great": "good",
    "finished": "done", "completed": "done", "complete": "done", "ready": "done",
    "create": "make", "build": "make", "produce": "make", "generate": "make",
    "construct": "make", "compute": "make", "calculate": "make",
    "person": "manling", "people": "manling", "human": "manling",
    "enemy": "enemy", "friend": "friend",
    "quickly": "fast", "rapidly": "fast", "slowly": "slow",
    "very": "big", "really": "big", "extremely": "big",
    "help": "help", "fight": "fight", "food": "grub", "eat": "eat",
    "yes": "yes", "no": "no", "hello": "Oi", "hi": "Oi", "please": "",
    "thanks": "good", "thank": "good", "sorry": "Grr",
    "because": "cause", "therefore": "so", "however": "but",
    "result": "answer", "output": "answer", "value": "answer",
    "function": "spell", "code": "rune", "program": "rune", "script": "rune",
    "number": "count", "list": "pile", "string": "word",
}

# Closed FR set, injected at a controlled rate (light sprinkle only).
# Maps an EN trigger word -> FR replacement; applied probabilistically.
FR_REPLACE = {
    "yes": "oui", "no": "non", "good": "bon", "bad": "mal", "big": "gros",
    "friend": "ami", "enemy": "ennemi", "fire": "feu", "eat": "manger",
    "water": "eau", "and": "et", "smash": "casser",
}
# Free-standing FR interjections occasionally prepended to a sentence.
FR_INTERJ = ("Bah,", "Oui,", "Non,", "Grr,", "Hein,")

# Spans that must survive byte-for-byte. Order matters (longest/most specific first).
_PROTECT_RES = (
    re.compile(r"`[^`]*`"),                       # backtick code spans
    re.compile(r"\"[^\"]*\"|'[^']*'"),            # quoted payloads
    re.compile(r"https?://\S+"),                  # urls
    re.compile(r"-?\d+(?:\.\d+)?"),               # numbers
    re.compile(r"\b[A-Za-z_]\w*\s*\([^)]*\)"),    # call-ish identifier(...)
    re.compile(r"\b\w+_\w[\w_]*\b"),              # snake_case identifiers
)

_SENT = "\x00%d\x00"
_WORD_RE = re.compile(r"[A-Za-z']+|[^A-Za-z']+")


def _protect(text: str) -> tuple[str, list[str]]:
    """Replace structured spans with sentinels; return (masked_text, spans)."""
    spans: list[str] = []

    def _sub(m: "re.Match[str]") -> str:
        spans.append(m.group(0))
        return _SENT % (len(spans) - 1)

    for rx in _PROTECT_RES:
        text = rx.sub(_sub, text)
    return text, spans


def _restore(text: str, spans: list[str]) -> str:
    # Reverse order: a later span (e.g. `id(...)`) may embed an earlier sentinel
    # (e.g. a quoted arg), so the outer must be expanded before the inner is filled.
    for i in range(len(spans) - 1, -1, -1):
        text = text.replace(_SENT % i, spans[i])
    return text


def _xform_word(w: str, rng: random.Random, fr_rate: float) -> str | None:
    """Transform one alphabetic word. Return None to drop it."""
    low = w.lower()
    if low in ARTICLES or low in DROP_WORDS:
        return None
    repl = None
    if low in PRONOUN:
        repl = PRONOUN[low]
    elif low in SUBST:
        repl = SUBST[low]
        if repl == "":
            return None
    else:
        repl = w
    # FR sprinkle: only on a closed trigger set, at the controlled rate.
    base = repl.lower()
    if base in FR_REPLACE and rng.random() < fr_rate:
        fr = FR_REPLACE[base]
        repl = fr.capitalize() if repl[:1].isupper() else fr
    return repl


def cavemanize(text: str, rng: random.Random, fr_rate: float = 0.12) -> str:
    """Rewrite natural-language prose into the orc-caveman register.

    Deterministic given `rng`. Structured spans (numbers, code, quotes, identifiers)
    are protected and restored verbatim. `fr_rate` is the per-trigger-word probability
    of swapping an EN word for its closed-set FR counterpart (light sprinkle).
    """
    masked, spans = _protect(text)
    out: list[str] = []
    for tok in _WORD_RE.findall(masked):
        if tok and (tok[0].isalpha() or tok[0] == "'"):
            xf = _xform_word(tok, rng, fr_rate)
            if xf is not None:
                out.append(xf)
        else:
            out.append(tok)
    result = "".join(out)
    # Collapse spaces left by dropped words; tidy space-before-punct.
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\s+([,.!?;:])", r"\1", result)
    result = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), result)
    # Rare leading FR interjection.
    if rng.random() < fr_rate * 0.5:
        result = rng.choice(FR_INTERJ) + " " + result[:1].lower() + result[1:]
    result = _restore(result.strip(), spans)
    return result


def caveman_report(echo: str | None, rng: random.Random) -> str:
    """Caveman frame for the terminal report turn of a procedural trace.

    `echo` (the verbatim stdout/content the old `_final_report` echoed) is appended
    UNCHANGED — it is data, not prose, and may contain numbers/code the verifier-free
    report still ought to reproduce faithfully.
    """
    frame = rng.choice(("Work done.", "Mob done.", "Job done good.", "Done. all good.", "Grr. work done."))
    if echo:
        return f"{frame} {echo}"
    return frame


def persona(rng: random.Random) -> str:
    """Pick a persona-system variant."""
    return rng.choice(PERSONA_VARIANTS)
