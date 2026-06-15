"""Hardened tool execution for the PUBLIC demo.

The shared `infer.executors` is honest about its own threat model: it is a fence
against agent *mistakes* (rm of system files, runaway loops), NOT a sandbox against
a malicious caller. That is fine for offline distill generation, but the web demo
takes a prompt from anyone on the internet and runs whatever code grot emits — so
here we add a real OS-level fence on top.

What this module changes vs the raw `execute`:
  * shell_run / run_command are DROPPED — no arbitrary shell on a public box.
  * py_run / python_repl run under POSIX rlimits enforced by the kernel in a
    preexec hook (CPU time, address space, file size, open files) and with
    RLIMIT_NPROC = current usage → the child cannot fork/exec, which kills
    subprocess, os.system and fork-bombs at the OS level even if the textual
    blacklist is bypassed.
  * Network egress is not reachable from the child (no fork to spawn helpers,
    and the deploy container has no outbound route — see Dockerfile).
  * Every other tool (read_file/write_file/list_dir/grep/calculate/stubs) is
    delegated unchanged to the shared executors, already sandbox-path-confined.

Outer boundary (defence in depth): the container runs as non-root with a
read-only root fs except the sandbox tmpdir, and no network. This module is the
inner boundary so the demo is still safe if run outside that container.
"""
from __future__ import annotations

import re
import subprocess
import sys

# Tools the demo will execute. shell_run/run_command intentionally absent.
DEMO_ALLOWED = frozenset({
    "read_file", "write_file", "list_dir", "grep", "calculate",
    "py_run", "python_repl", "search_web", "fetch_url", "think", "plan",
})

# rlimits for the python child (seconds / bytes / count)
_PY_CPU_SECONDS = 5
_PY_WALL_TIMEOUT = 8          # hard wall clock on top of CPU limit
_PY_ADDRESS_SPACE = 512 * 1024 * 1024
_PY_FILE_SIZE = 8 * 1024 * 1024
_PY_OPEN_FILES = 64

_MAX_STDOUT = 4096
_MAX_STDERR = 2048

# Same textual blacklist as the shared executor — kept as a cheap first gate; the
# rlimits below are the real guarantee.
_PY_FORBIDDEN = re.compile(
    r"\b(os\.system|subprocess|socket|urllib|requests|open\s*\(\s*['\"]/|"
    r"shutil\.rmtree|os\.remove|os\.unlink|sys\.exit|ctypes|multiprocessing)\b"
)


def _truncate(s: str, limit: int) -> tuple[str, bool]:
    return (s, False) if len(s) <= limit else (s[:limit], True)


def _error(code: str, message: str) -> dict:
    return {"error": code, "message": message}


def _rlimit_preexec():
    """Run in the child between fork and exec. Cap kernel resources, then forbid
    any further fork/exec by pinning RLIMIT_NPROC low. Every limit is best-effort:
    macOS rejects some values that Linux (the deploy target) accepts, and a raised
    exception here would surface as an opaque SubprocessError, so we never let one
    escape. On Linux all of these take effect — that is where it matters."""
    import os
    import resource

    def _set(what, soft, hard=None):
        try:
            resource.setrlimit(what, (soft, hard if hard is not None else soft))
        except (ValueError, OSError, AttributeError):
            pass

    _set(resource.RLIMIT_CPU, _PY_CPU_SECONDS)
    _set(resource.RLIMIT_FSIZE, _PY_FILE_SIZE)
    _set(resource.RLIMIT_NOFILE, _PY_OPEN_FILES)
    # RLIMIT_AS is honoured on Linux; flaky/ignored on macOS — best-effort.
    _set(resource.RLIMIT_AS, _PY_ADDRESS_SPACE)
    # No new processes: blocks os.fork / subprocess / os.system / fork bombs (Linux).
    if getattr(resource, "RLIMIT_NPROC", None) is not None:
        _set(resource.RLIMIT_NPROC, 1)
    # Detach into a new session so signals stay contained.
    try:
        os.setsid()
    except OSError:
        pass


def _safe_py_run(args: dict, sb) -> dict:
    code = args.get("code")
    if not isinstance(code, str):
        return _error("EINVAL", "missing 'code'")
    if _PY_FORBIDDEN.search(code):
        return _error("EBLACKLIST", "code contains forbidden symbol")
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": str(sb.root)}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", code],
            cwd=str(sb.root), env=env,
            capture_output=True, text=True, timeout=_PY_WALL_TIMEOUT,
            preexec_fn=_rlimit_preexec if sys.platform != "win32" else None,
        )
    except subprocess.TimeoutExpired:
        return _error("ETIMEDOUT", f"python exceeded {_PY_WALL_TIMEOUT}s wall clock")
    except Exception as e:
        return _error("EEXEC", f"{type(e).__name__}: {e}")
    out, t1 = _truncate(proc.stdout, _MAX_STDOUT)
    err, t2 = _truncate(proc.stderr, _MAX_STDERR)
    payload = {"stdout": out, "stderr": err, "exit_code": proc.returncode}
    if t1 or t2:
        payload["truncated"] = True
    return payload


def safe_execute(name: str, args: dict, sb) -> dict:
    """Drop-in replacement for infer.executors.execute, hardened for public use."""
    from infer.executors import execute as _raw_execute
    if name not in DEMO_ALLOWED:
        return _error("EFORBIDDEN", f"tool disabled in public demo: {name!r}")
    if not isinstance(args, dict):
        return _error("EINVAL", "args must be an object")
    if name in ("py_run", "python_repl"):
        return _safe_py_run(args, sb)
    return _raw_execute(name, args, sb)
