"""Multi-model registry for the public demo.

Several checkpoints — mid-training, final, post-Orkun — live side-by-side on the
persistent volume under ``/data/models/``. Each is one *slot*: a ``.safetensors``
file plus an entry in ``manifest.json`` (id, label, description, size, uploaded
timestamp). This is real bytes on the real persistent disk — no DB, no mock.

The server keeps AT MOST ONE :class:`~orkun.demo.server.Engine` loaded at a time
(small 2 GB VPS), so :class:`AppState` lazily (re)builds the engine when a visitor
selects a different model and evicts the previous one. This module only owns the
on-disk registry; the engine cache lives in ``AppState``.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

# A slot id is what goes in the filename + URL, so keep it filesystem/URL-safe.
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def valid_id(slot_id: str) -> bool:
    return bool(_ID_RE.match(slot_id or ""))


class ModelRegistry:
    """On-disk catalogue of model slots under ``root`` (the persistent volume)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = self.root / "manifest.json"
        self.lock = threading.Lock()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # --- manifest io -------------------------------------------------------
    def _read(self) -> dict:
        try:
            return json.loads(self.manifest.read_text())
        except (OSError, ValueError):
            return {"models": {}}

    def _write(self, data: dict) -> None:
        tmp = self.manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(self.manifest)

    def file_for(self, slot_id: str) -> Path:
        # Defense in depth: never let a slot id escape the models dir, even if a
        # caller forgot to validate it. A bad id raises rather than touching disk.
        if not valid_id(slot_id):
            raise ValueError(f"unsafe model id: {slot_id!r}")
        p = (self.root / f"{slot_id}.safetensors").resolve()
        if p.parent != self.root.resolve():
            raise ValueError(f"unsafe model id: {slot_id!r}")
        return p

    # --- queries -----------------------------------------------------------
    def path(self, slot_id: str) -> Path | None:
        """Existing weights file for ``slot_id``, or None."""
        with self.lock:
            entry = self._read().get("models", {}).get(slot_id)
        if not entry:
            return None
        p = self.file_for(slot_id)
        return p if p.is_file() else None

    def list(self) -> list[dict]:
        """All slots (manifest order), each with id/label/description/size/present."""
        with self.lock:
            models = self._read().get("models", {})
        out = []
        for sid, e in models.items():
            p = self.file_for(sid)
            present = p.is_file()
            out.append({
                "id": sid,
                "label": e.get("label", sid),
                "description": e.get("description", ""),
                "size": p.stat().st_size if present else 0,
                "present": present,
                "uploaded_at": e.get("uploaded_at"),
            })
        return out

    # --- mutations ---------------------------------------------------------
    def register(self, slot_id: str, label: str, description: str, size: int) -> None:
        """Record (or update) a slot's metadata. The weights file must already be
        in place at :meth:`file_for`."""
        with self.lock:
            data = self._read()
            data.setdefault("models", {})[slot_id] = {
                "label": label or slot_id,
                "description": description or "",
                "size": size,
                "uploaded_at": time.time(),
            }
            self._write(data)

    def remove(self, slot_id: str) -> bool:
        with self.lock:
            data = self._read()
            existed = data.get("models", {}).pop(slot_id, None) is not None
            if existed:
                self._write(data)
        if existed:  # only unlink a slot we actually tracked; file_for re-checks safety
            try:
                self.file_for(slot_id).unlink()
            except (OSError, ValueError):
                pass
        return existed

    def migrate_legacy(self, legacy_path: Path, slot_id: str = "grot",
                       label: str = "GROT-80M") -> None:
        """One-time adoption of the old single-file checkpoint into a slot so a
        previously-deployed weights blob keeps working after the multi-model
        upgrade. No-op if the legacy file is absent or already in the registry."""
        legacy_path = Path(legacy_path)
        if not legacy_path.is_file():
            return
        with self.lock:
            data = self._read()
            if data.get("models"):
                return  # registry already populated — leave it alone
        dst = self.file_for(slot_id)
        if not dst.exists():
            try:
                if legacy_path.resolve() != dst.resolve():
                    dst.write_bytes(legacy_path.read_bytes())
            except OSError:
                return
        self.register(slot_id, label, "Original demo checkpoint.", dst.stat().st_size)
