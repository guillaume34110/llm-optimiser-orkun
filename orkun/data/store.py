"""Store — append-only JSONL of verified + counterfactual trajectories.

Rows are SFT-packer compatible (`wire` + `task` keys, last turn a final report)
plus Orkun fields (`necessity`, `source`). The store is the loot pile the trainer
samples from. JSONL (not .npz) so it streams and the Orkish packer can read it
directly; packing to .npz happens in the trainer step (Task 10).
"""
from __future__ import annotations

import json
import random
from pathlib import Path


class Store:
    def __init__(self, path: Path, rng: random.Random | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []
        self.rng = rng or random.Random(0)
        # Reload an existing loot file so a restarted arm (crash, resumed session)
        # keeps sampling from its full history instead of starting empty.
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._rows.append(json.loads(line))

    def add(
        self,
        task_id: str,
        wire: str,
        n_calls: int,
        necessity: list[float],
        source: str,
        *,
        graded: float | None = None,
        surprise: float | None = None,
        reward_hat: float | None = None,
        curiosity: float | None = None,
    ) -> None:
        row = {
            "task": task_id,        # pack_sft.load_examples reads obj.get("task")
            "wire": wire,
            "n_calls": n_calls,
            "necessity": necessity,
            "source": source,
        }
        # Phase B fields are written ONLY when supplied, so the flag-off path stays
        # byte-identical to Phase A (and old rows remain valid).
        for key, val in (("graded", graded), ("surprise", surprise),
                         ("reward_hat", reward_hat), ("curiosity", curiosity)):
            if val is not None:
                row[key] = val
        self._rows.append(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def ready_for_update(self, min_rows: int) -> bool:
        return len(self._rows) >= min_rows

    def sample(self, n: int) -> list[dict]:
        if n >= len(self._rows):
            return list(self._rows)
        return self.rng.sample(self._rows, n)

    def __len__(self) -> int:
        return len(self._rows)
