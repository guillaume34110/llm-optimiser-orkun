"""Snapshot/restore for the Orkish Sandbox filesystem — enables exact counterfactual replay.

The Orkish sandbox is deterministic (a temp dir + cleaned subprocess env), so a
byte-for-byte copy of its root captures the full mutable world state relevant to
the verifier (files + cwd contents). Snapshot copies the tree aside; restore wipes
the live root and copies it back. No env/cwd state survives across executor calls
in Orkish, so the filesystem IS the state.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from infer.executors import Sandbox


def snapshot(sb: Sandbox) -> Path:
    """Copy the sandbox root into a fresh temp dir; return its path (the snapshot handle)."""
    dst = Path(tempfile.mkdtemp(prefix="orkun_snap_")).resolve()
    shutil.copytree(sb.root, dst / "root")
    return dst


def restore(sb: Sandbox, snap: Path) -> None:
    """Replace the live sandbox root contents with the snapshot's."""
    shutil.rmtree(sb.root, ignore_errors=True)
    shutil.copytree(snap / "root", sb.root)


def free_snapshot(snap: Path) -> None:
    """Delete a snapshot handle's backing files."""
    shutil.rmtree(snap, ignore_errors=True)
