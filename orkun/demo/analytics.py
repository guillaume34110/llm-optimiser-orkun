"""Basic, dependency-free analytics for the public demo.

Every page visit and every model run is appended as one JSON line to an
append-only log on the persistent volume (next to the checkpoint, so it survives
redeploys). The admin dashboard reads the whole file back and aggregates it in
Python — the volumes here are tiny (a demo), so a JSONL + full re-read is the
honest, real implementation: no extra service, no DB to provision, real bytes on
the real persistent disk.

Threading: the demo runs on ThreadingHTTPServer, so appends are serialised with
a lock. Reads take the same lock to get a consistent snapshot.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

# Cap how much we keep / scan so the file can't grow without bound on a long-lived
# deploy. Oldest lines beyond this are dropped on the next write.
_MAX_EVENTS = 20000
_RECENT = 60          # events surfaced in the dashboard table
_PROMPT_MAX = 200     # truncate stored prompts


class Analytics:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # --- write -------------------------------------------------------------
    def log(self, kind: str, ip: str, ua: str, **extra) -> None:
        ev = {"ts": time.time(), "kind": kind, "ip": ip or "?", "ua": (ua or "")[:300]}
        if "prompt" in extra and isinstance(extra["prompt"], str):
            extra["prompt"] = extra["prompt"][:_PROMPT_MAX]
        ev.update(extra)
        line = json.dumps(ev, ensure_ascii=False)
        with self.lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    # --- read --------------------------------------------------------------
    def _load(self) -> list:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
        except (OSError, ValueError):
            return []
        if len(rows) > _MAX_EVENTS:
            rows = rows[-_MAX_EVENTS:]
            # best-effort trim of the file so it stops growing
            with self.lock:
                try:
                    with open(self.path, "w", encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                except OSError:
                    pass
        return rows

    def summary(self) -> dict:
        with self.lock:
            rows = self._load()
        now = time.time()
        day = now - 86400
        visits = [r for r in rows if r.get("kind") == "visit"]
        runs = [r for r in rows if r.get("kind") == "run"]
        ips = {r.get("ip") for r in rows if r.get("ip")}
        tool_counts: dict[str, int] = {}
        ok_runs = 0
        for r in runs:
            if r.get("ok"):
                ok_runs += 1
            for t in (r.get("tools") or []):
                tool_counts[t] = tool_counts.get(t, 0) + 1
        ts_all = [r.get("ts", 0) for r in rows if r.get("ts")]
        recent = [{
            "ts": r.get("ts"),
            "kind": r.get("kind"),
            "ip": r.get("ip"),
            "ua": r.get("ua", ""),
            "prompt": r.get("prompt", ""),
            "tools": r.get("tools", []),
            "ok": r.get("ok"),
            "ms": r.get("ms"),
            "n_tokens": r.get("n_tokens"),
        } for r in rows[-_RECENT:][::-1]]
        return {
            "total_events": len(rows),
            "total_visits": len(visits),
            "total_runs": len(runs),
            "ok_runs": ok_runs,
            "unique_ips": len(ips),
            "visits_24h": sum(1 for r in visits if r.get("ts", 0) >= day),
            "runs_24h": sum(1 for r in runs if r.get("ts", 0) >= day),
            "first_seen": min(ts_all) if ts_all else None,
            "last_seen": max(ts_all) if ts_all else None,
            "tool_breakdown": dict(sorted(tool_counts.items(), key=lambda kv: -kv[1])),
            "recent": recent,
        }
