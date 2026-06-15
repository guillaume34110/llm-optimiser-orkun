"""WAAAGH GROT-80M demo — a tiny, dependency-free web app that shows the model doing the
only thing it was trained to do: emit tool calls.

The model is a custom nGPT architecture (no ONNX/transformers.js path), so the only honest
way to demo it in a browser is to serve the REAL model from a tiny Python backend and
render its output in a page. This uses the stdlib `http.server` only — no FastAPI, no
build step. Run it, open the URL, type a request, watch grot fire a tool with real output.

Pipeline per request (all reused, tested primitives — nothing is faked):
    user text -> wire prompt -> WaaaghNet greedy decode -> parse_calls
              -> execute each call in a fresh Sandbox -> real tool output back to the page

Run (turnkey — no flags needed once weights are at orkun/demo/assets/grot-80m.safetensors
and the Orkish repo sits next to this one or at /opt/Orkish):
    python -m orkun.demo.server
    # then open http://127.0.0.1:8000

Tool execution is hardened for public exposure — see orkun/demo/safe_exec.py.
"""
from __future__ import annotations

import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent          # orkun repo root (…/orkun/demo/ -> repo)


def _ckpt_path(cli: str | None) -> Path:
    """The canonical weights location (does not need to exist yet — /admin uploads
    here onto the persistent volume). Order: --checkpoint, $GROT_CKPT, bundled assets/."""
    import os
    return Path(cli or os.environ.get("GROT_CKPT")
                or str(_HERE / "assets" / "grot-80m.safetensors"))


def _find_checkpoint(cli: str | None) -> str | None:
    """Return an existing checkpoint to load at boot, or None (admin uploads later)."""
    import os
    cands = [
        cli, os.environ.get("GROT_CKPT"),
        str(_HERE / "assets" / "grot-80m.safetensors"),
        str(_REPO / "kaggle/_out_v4_burdok/sft_procedural/ckpts/best.safetensors"),
    ]
    for c in cands:
        if c and Path(c).is_file():
            return c
    return None


def _resolve_config(cli: str | None) -> str:
    import os
    cands = [cli, os.environ.get("GROT_CONFIG"),
             str(_REPO / "configs/orkun_sft_procedural_v4.yaml")]
    for c in cands:
        if c and Path(c).is_file():
            return c
    raise SystemExit("[grot] config not found; pass --config.")


def _resolve_orkish(cli: str | None) -> str:
    """Locate the Orkish repo (tokenizer + tool_registry + model arch)."""
    import os
    cands = [cli, os.environ.get("ORKISH_REPO"),
             str(_REPO.parent / "Orkish"), "/opt/Orkish"]
    for c in cands:
        if c and (Path(c) / "data" / "tool_registry.json").is_file():
            return c
    raise SystemExit(
        "[grot] Orkish repo not found. Clone llm-waaagh-Orkish next to this repo, "
        "set $ORKISH_REPO, or pass --orkish-repo.")

# One hand-made prompt grot nails cleanly, then real in-distribution tasks drawn from the
# GoalFamilies the model was trained on. Paraphrases break it (it is brittle to exact
# tokens), so the curated buttons use the families' own canonical wording + seed files —
# that is what reliably makes grot fire a tool with real sandbox output.
_HAND = [{"label": "Print ANSWER=42", "prompt": "Print exactly ANSWER=42 using python.", "seed": {}}]
_SCENARIO_FAMILIES = [
    ("compute", "sum two numbers"),
    ("arith", "evaluate an expression"),
    ("sequence", "next in sequence"),
    ("pipeline", "read + compute + write"),
    ("write_file", "create a file"),
]


def build_scenarios():
    """Draw one canonical d0 task per chosen family (fixed seed = stable buttons)."""
    import random
    from orkun.goals.families import ALL_FAMILIES
    by_name = {f.name: f for f in ALL_FAMILIES}
    out = list(_HAND)
    rng = random.Random(0)
    for name, label in _SCENARIO_FAMILIES:
        fam = by_name.get(name)
        if fam is None:
            continue
        t = fam.sample(rng, difficulty=0)
        out.append({"label": label, "prompt": t.prompt, "seed": dict(t.seed)})
    return out


class Engine:
    """Loads the model once; turns a user message into parsed + executed tool calls."""

    def __init__(self, checkpoint: str, config: str, orkish_repo: str, device: str):
        from orkun.scripts.run_kaggle import _add_orkish_to_path, _load_tokenizer, _build_net
        _add_orkish_to_path(Path(orkish_repo))
        import yaml
        self.orkish_repo = Path(orkish_repo)
        model_cfg = yaml.safe_load(open(config))["model"]
        self.tok = _load_tokenizer(self.orkish_repo)
        self.net = _build_net(model_cfg, Path(checkpoint), device).eval()
        reg = self.orkish_repo / "data" / "tool_registry.json"
        regj = json.loads(reg.read_text())["tools"]
        # Honesty flag: grot was only ever trained (via the GoalFamilies) on these tools.
        # The others exist in the registry and execute for real if grot calls them, but it
        # was never taught to — in practice it collapses everything onto py_run.
        trained = {"read_file", "write_file", "python_repl", "py_run"}
        self.tools = [{"name": t["name"], "desc": t.get("description", ""),
                       "trained": t["name"] in trained} for t in regj]
        self.ktools = frozenset(t["name"] for t in regj)
        if "python_repl" in self.ktools:
            self.ktools = self.ktools | frozenset(["py_run"])
            # py_run is the wire alias the model emits for python_repl — surface it as a
            # tool chip so the UI can light it up when grot fires it.
            self.tools.append({"name": "py_run", "trained": True,
                               "desc": "Run Python (alias of python_repl). Grot's hammer."})
        self.stop_ids = {self.tok.specials[k] for k in ("eos", "user")}
        self.scenarios = build_scenarios()

    def _decode(self, ids):
        text = self.tok.decode(ids, skip_special=True)
        for s in ("<|eos|>", "<|user|>"):
            i = text.find(s)
            if i != -1:
                text = text[:i]
        return text

    def run(self, prompt: str, seed: dict, samples: int, temperature: float):
        """Generate, parse, and EXECUTE the tool calls. Returns a render-ready dict.

        The sandbox is a real tempdir on disk. We snapshot its file tree BEFORE and AFTER
        the tool calls run and return both, so the page can prove the effects are real:
        read_file reads files that are actually present, write_file makes a file you can see,
        list_dir reflects the true directory.
        """
        from infer.executors import Sandbox
        from infer.monkey_wire import parse_calls
        from orkun.demo.safe_exec import safe_execute as execute
        from orkun.policy.sampler import sample as sample_one

        wire = f"<|bos|><|user|>{prompt}<|assistant|>"
        ids = self.tok.encode(wire, add_bos=False, add_eos=False)
        # Greedy first (temperature 0). If it yields no valid call and the caller allows
        # retries, draw a few sampled attempts and keep the first that parses a call.
        max_new = 256
        text, out_ids, attempts = "", [], 0
        for attempt in range(max(1, samples)):
            temp = 0.0 if attempt == 0 else temperature
            out = sample_one(self.net, ids, max_new=max_new, temperature=temp,
                             top_p=0.95, stop_ids=self.stop_ids, seed=attempt)
            text = self._decode(out)
            out_ids = out
            attempts = attempt + 1
            if parse_calls(text, known_tools=self.ktools):
                break

        # The FULL raw generation. assistant_text is the cleaned string; `tokens` is the
        # honest token-by-token stream grot actually emitted (each id decoded on its own),
        # with the terminal <|eos|> appended when that is what stopped it — so the page can
        # show every token, not just the parsed call.
        raw_text = self.tok.decode(out_ids, skip_special=False)
        n_tokens = len(out_ids)
        tokens = [self.tok.decode([i], skip_special=False) for i in out_ids]
        stop_reason = "hit 256-token cap" if n_tokens >= max_new else "stopped on <|eos|>"
        if n_tokens < max_new:
            tokens = tokens + ["<|eos|>"]
        calls = parse_calls(text, known_tools=self.ktools)
        rendered = []
        # Always create the sandbox + seed it, even when there is no call, so the page can
        # show the starting files (what read_file/list_dir would see).
        sb = Sandbox.create(seed=False)
        try:
            for rel, content in (seed or {}).items():
                p = sb.resolve(rel)
                if p is not None:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content)
            fs_before = _snapshot_fs(sb.root)
            for c in calls[:16]:
                result = execute(c.name, c.args, sb)
                rendered.append({
                    "name": c.name,
                    "args": c.args,
                    "ok": "error" not in result,
                    "output": _pretty(result),
                })
            fs_after = _snapshot_fs(sb.root)
        finally:
            sb.cleanup()
        return {"assistant_text": text, "raw_text": raw_text, "tokens": tokens,
                "n_tokens": n_tokens, "stop_reason": stop_reason,
                "calls": rendered, "n_calls": len(rendered), "attempts": attempts,
                "fs_before": fs_before, "fs_after": fs_after}


def _snapshot_fs(root: Path, max_bytes: int = 4000) -> list:
    """List every file in the sandbox with its real size + (text) content."""
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = f"<binary, {len(raw)} bytes>"
        trunc = len(text) > max_bytes
        out.append({"path": rel, "size": len(raw),
                    "content": text[:max_bytes] + ("…" if trunc else "")})
    return out


def _pretty(result: dict) -> str:
    if "error" in result:
        return f"{result.get('error', 'ERROR')}: {result.get('message', '')}".strip()
    if "stdout" in result:
        body = result.get("stdout", "")
        err = result.get("stderr", "")
        return body + (f"\n[stderr] {err}" if err else "")
    return json.dumps(result, ensure_ascii=False, indent=2)


class AppState:
    """Holds the (optionally absent) engine and rebuilds it when admin uploads weights.

    The server boots even with no checkpoint on disk so the operator can deploy the
    image empty and upload the 319 MB weights once, via /admin, onto a persistent
    volume — keeping the big blob out of git and out of the image."""

    def __init__(self, config: str, orkish_repo: str, device: str, ckpt_path: Path):
        import threading
        self.config = config
        self.orkish_repo = orkish_repo
        self.device = device
        self.ckpt_path = ckpt_path
        self.engine: Engine | None = None
        self.lock = threading.Lock()
        self.last_error: str | None = None

    def try_boot(self, checkpoint: str | None):
        if checkpoint and Path(checkpoint).is_file():
            try:
                self.load(checkpoint)
            except Exception as e:  # boot must not die on a bad ckpt — admin can re-upload
                self.last_error = f"{type(e).__name__}: {e}"
                print(f"[grot] boot load failed: {self.last_error}", flush=True)

    def load(self, checkpoint: str):
        eng = Engine(checkpoint, self.config, self.orkish_repo, self.device)
        with self.lock:
            self.engine = eng
            self.last_error = None
        print(f"[grot] model loaded <- {checkpoint}", flush=True)

    def status(self) -> dict:
        p = self.ckpt_path
        return {
            "model_loaded": self.engine is not None,
            "ckpt_path": str(p),
            "ckpt_present": p.is_file(),
            "ckpt_size": p.stat().st_size if p.is_file() else 0,
            "last_error": self.last_error,
        }


def _admin_secret() -> str | None:
    """Admin is enabled only when ADMIN_PASSWORD is set."""
    import os
    return os.environ.get("ADMIN_PASSWORD") or None


def _make_token(user: str, secret: str) -> str:
    import hashlib
    import hmac
    return hmac.new(secret.encode(), f"grot:{user}".encode(), hashlib.sha256).hexdigest()


def make_handler(state: AppState):
    import hmac
    import os
    index_html = (_HERE / "static" / "index.html").read_text()
    admin_html = (_HERE / "static" / "admin.html").read_text()
    admin_user = os.environ.get("ADMIN_USER", "admin")
    MAX_UPLOAD = 2 * 1024 * 1024 * 1024  # 2 GB hard cap on the upload body

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="application/json", headers=None):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        # --- admin auth ---------------------------------------------------
        def _authed(self) -> bool:
            secret = _admin_secret()
            if not secret:
                return False
            from http.cookies import SimpleCookie
            raw = self.headers.get("Cookie", "")
            tok = SimpleCookie(raw).get("grot_admin")
            return bool(tok and hmac.compare_digest(tok.value, _make_token(admin_user, secret)))

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(n) if n else b""

        # --- routing ------------------------------------------------------
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, index_html, "text/html; charset=utf-8")
            if self.path in ("/admin", "/admin/"):
                return self._send(200, admin_html, "text/html; charset=utf-8")
            if self.path == "/admin/status":
                if not _admin_secret():
                    return self._send(503, json.dumps({"error": "admin disabled (set ADMIN_PASSWORD)"}))
                if not self._authed():
                    return self._send(401, json.dumps({"error": "login required"}))
                return self._send(200, json.dumps(state.status()))
            if self.path == "/api/meta":
                if state.engine is None:
                    return self._send(200, json.dumps({"tools": [], "scenarios": [],
                                                       "model_loaded": False}))
                return self._send(200, json.dumps({
                    "tools": state.engine.tools, "scenarios": state.engine.scenarios,
                    "model_loaded": True}))
            return self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if self.path == "/admin/login":
                return self._do_login()
            if self.path == "/admin/logout":
                return self._send(200, json.dumps({"ok": True}),
                                  headers={"Set-Cookie": "grot_admin=; Max-Age=0; Path=/; HttpOnly"})
            if self.path == "/admin/upload":
                return self._do_upload()
            if self.path == "/api/run":
                return self._do_run()
            return self._send(404, json.dumps({"error": "not found"}))

        def _do_login(self):
            secret = _admin_secret()
            if not secret:
                return self._send(503, json.dumps({"error": "admin disabled (set ADMIN_PASSWORD)"}))
            try:
                req = json.loads(self._read_body() or b"{}")
            except Exception:
                return self._send(400, json.dumps({"error": "bad json"}))
            ok_user = hmac.compare_digest(str(req.get("user", "")), admin_user)
            ok_pass = hmac.compare_digest(str(req.get("password", "")), secret)
            if not (ok_user and ok_pass):
                return self._send(401, json.dumps({"error": "bad credentials"}))
            tok = _make_token(admin_user, secret)
            cookie = f"grot_admin={tok}; Path=/; HttpOnly; SameSite=Strict; Max-Age=86400"
            return self._send(200, json.dumps({"ok": True}), headers={"Set-Cookie": cookie})

        def _do_upload(self):
            if not _admin_secret() or not self._authed():
                return self._send(401, json.dumps({"error": "login required"}))
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_UPLOAD:
                return self._send(413, json.dumps({"error": f"bad size {n}"}))
            dst = state.ckpt_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".upload")
            got = 0
            try:
                with open(tmp, "wb") as f:
                    while got < n:
                        chunk = self.rfile.read(min(1 << 20, n - got))
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                # Validate it actually loads as a model before swapping it in.
                state.load(str(tmp))
                os.replace(tmp, dst)
                # reload from the final path so the engine points at the persisted file
                state.load(str(dst))
                return self._send(200, json.dumps({"ok": True, **state.status()}))
            except Exception as e:
                traceback.print_exc()
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return self._send(400, json.dumps({"error": f"invalid checkpoint: {e}"}))

        def _do_run(self):
            if state.engine is None:
                return self._send(503, json.dumps({
                    "error": "model not loaded — admin must upload weights at /admin"}))
            try:
                req = json.loads(self._read_body() or b"{}")
                result = state.engine.run(
                    prompt=req.get("prompt", ""),
                    seed=req.get("seed", {}),
                    samples=int(req.get("samples", 6)),
                    temperature=float(req.get("temperature", 0.45)),
                )
                return self._send(200, json.dumps(result))
            except Exception as e:  # never crash the demo on a bad request
                traceback.print_exc()
                return self._send(500, json.dumps({"error": str(e)}))

    return Handler


def main():
    import os
    ap = argparse.ArgumentParser(description="WAAAGH GROT-80M tool-call demo server")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--orkish-repo", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default=os.environ.get("GROT_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GROT_PORT", "8000")))
    args = ap.parse_args()

    config = _resolve_config(args.config)
    orkish_repo = _resolve_orkish(args.orkish_repo)
    ckpt_path = _ckpt_path(args.checkpoint)
    state = AppState(config, orkish_repo, args.device, ckpt_path)
    state.try_boot(_find_checkpoint(args.checkpoint))
    admin = "ON" if _admin_secret() else "OFF (set ADMIN_PASSWORD)"
    print(f"[grot] orkish={orkish_repo}\n[grot] ckpt_target={ckpt_path}\n"
          f"[grot] model_loaded={state.engine is not None}  admin={admin}", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[grot] WAAAGH ready -> http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[grot] shutting down")


if __name__ == "__main__":
    main()
