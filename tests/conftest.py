"""Shared fixtures. Locates the real Orkish tokenizer + tool registry via ORKISH_REPO."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _orkish_repo() -> Path:
    p = Path(os.environ.get("ORKISH_REPO", Path(__file__).resolve().parents[2] / "Orkish"))
    return p.resolve()


def pytest_configure(config):
    """Add Orkish repo to sys.path early (before collection) so torch_impl is importable."""
    repo = _orkish_repo()
    if repo.is_dir():
        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)


@pytest.fixture(scope="session")
def orkish_repo() -> Path:
    repo = _orkish_repo()
    if not repo.is_dir():
        pytest.skip(f"Orkish repo not found at {repo} (set ORKISH_REPO)")
    # Add Orkish to path so torch_impl, model, infer, scripts can be imported
    sys.path.insert(0, str(repo))
    return repo


@pytest.fixture(scope="session")
def tokenizer(orkish_repo: Path):
    from model.tokenizer import OrkishTokenizer

    tok_path = orkish_repo / "data" / "tokenizer" / "orkish-bpe-8k.json"
    if not tok_path.is_file():
        pytest.skip(f"tokenizer not found at {tok_path}")
    return OrkishTokenizer(tok_path)


@pytest.fixture(scope="session")
def known_tools(orkish_repo: Path):
    """Tool names the wire parser will accept; falls back to the executor set."""
    import json

    reg = orkish_repo / "data" / "tool_registry.json"
    if reg.is_file():
        data = json.loads(reg.read_text())
        tools = frozenset(t["name"] for t in data["tools"])
    else:
        from infer.executors import EXECUTORS
        tools = frozenset(EXECUTORS.keys())
    # Include py_run as an alias for python_repl (Orkun families use py_run,
    # Orkish may expose it as python_repl; ensure both are accepted)
    if "python_repl" in tools:
        tools = tools | frozenset(["py_run"])
    return tools


@pytest.fixture
def tiny_net(tokenizer):
    """A randomly-initialised WaaaghNet small enough for CPU sampling tests."""
    import torch
    from torch_impl.model.waaagh import WaaaghConfig, WaaaghNet

    torch.manual_seed(0)
    cfg = WaaaghConfig(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        mlp_hidden=128,
        max_seq_len=256,
        stomach_tokens=4,
        attn_backend="sdpa",
    )
    net = WaaaghNet(cfg).eval()
    return net
