"""Boost-preserving wrapper around the stock Orkish packer's `load_examples`.

cf-distill upweights causal trajectories by REPETITION (`copies = 1 + round(max_nec
* boost)` identical wires) — the only upweighting the stock packer supports. Orkish
commit 83d05ac added duplicate-wire skipping to `scripts.pack_sft.load_examples`,
which silently nullifies that boost: every repeated copy lands in `skipped["dup"]`
and the cf arm degenerates to vanilla.

Orkish is read-only (substrate doctrine), so the repair lives here: count each
wire's multiplicity straight from the JSONL, run the stock (deduplicating,
validating) loader unchanged, then re-expand every kept example to its original
copy count. The output is exactly what `load_examples` returned before 83d05ac —
same records, same order up to expansion — so `build_dataset` and everything
downstream stay byte-identical.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.filter_traces import extract_wire
from scripts.pack_sft import load_examples


def _wire_counts(paths: list[Path]) -> Counter:
    """Multiplicity of every wire across the input JSONLs (same key the dedup uses)."""
    counts: Counter = Counter()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counts[extract_wire(json.loads(line))] += 1
    return counts


def load_examples_boosted(paths: list[Path], **kwargs) -> list[dict]:
    """`load_examples` with repetition-boost restored.

    Validation/filtering (incomplete, no_final_report, parse_error) is the stock
    loader's, untouched; only the dedup is compensated by re-expanding each kept
    example to its source multiplicity. A wire the loader rejected is never
    re-expanded (it was filtered, not deduped).
    """
    counts = _wire_counts([Path(p) for p in paths])
    out: list[dict] = []
    for ex in load_examples([Path(p) for p in paths], **kwargs):
        out.extend([ex] * max(1, counts.get(ex["wire"], 1)))
    return out
