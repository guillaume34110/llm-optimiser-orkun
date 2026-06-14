from pathlib import Path

from infer.executors import Sandbox, execute
from infer.monkey_wire import ToolCall
from orkun.world.sandbox import restore, snapshot, free_snapshot


def test_snapshot_restore_round_trips_filesystem():
    sb = Sandbox.create(seed=True)
    try:
        snap = snapshot(sb)
        # mutate after snapshot
        execute("write_file", {"path": "mutation.txt", "content": "after"}, sb)
        assert (sb.root / "mutation.txt").is_file()
        restore(sb, snap)
        # mutation gone, original seed back
        assert not (sb.root / "mutation.txt").exists()
        assert (sb.root / "README.md").is_file()
        free_snapshot(snap)
    finally:
        sb.cleanup()


def test_restore_recreates_deleted_files():
    sb = Sandbox.create(seed=True)
    try:
        snap = snapshot(sb)
        (sb.root / "README.md").unlink()
        assert not (sb.root / "README.md").exists()
        restore(sb, snap)
        assert (sb.root / "README.md").is_file()
        free_snapshot(snap)
    finally:
        sb.cleanup()
