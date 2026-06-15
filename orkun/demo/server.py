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


def _resolve_checkpoint(cli: str | None) -> str:
    """Find the model weights without flags. Order: --checkpoint, $GROT_CKPT,
    bundled assets/, the kaggle training output."""
    import os
    cands = [
        cli, os.environ.get("GROT_CKPT"),
        str(_HERE / "assets" / "grot-80m.safetensors"),
        str(_REPO / "kaggle/_out_v4_burdok/sft_procedural/ckpts/best.safetensors"),
    ]
    for c in cands:
        if c and Path(c).is_file():
            return c
    raise SystemExit(
        "[grot] no checkpoint found. Put weights at orkun/demo/assets/grot-80m.safetensors, "
        "set $GROT_CKPT, or pass --checkpoint.")


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


def make_handler(engine: Engine):
    index_html = (_HERE / "static" / "index.html").read_text()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, index_html, "text/html; charset=utf-8")
            if self.path == "/api/meta":
                return self._send(200, json.dumps({
                    "tools": engine.tools, "scenarios": engine.scenarios}))
            return self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if self.path != "/api/run":
                return self._send(404, json.dumps({"error": "not found"}))
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                result = engine.run(
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

    checkpoint = _resolve_checkpoint(args.checkpoint)
    config = _resolve_config(args.config)
    orkish_repo = _resolve_orkish(args.orkish_repo)
    print(f"[grot] ckpt={checkpoint}\n[grot] orkish={orkish_repo}\n[grot] loading model ...",
          flush=True)
    engine = Engine(checkpoint, config, orkish_repo, args.device)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"[grot] WAAAGH ready -> http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[grot] shutting down")


if __name__ == "__main__":
    main()
