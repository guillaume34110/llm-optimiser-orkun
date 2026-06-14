from orkun.policy.base import LMBackend
from orkun.policy.backends.waaagh import WaaaghBackend


def test_waaagh_backend_satisfies_protocol(tokenizer, tiny_net):
    backend = WaaaghBackend(tiny_net, tokenizer)
    assert isinstance(backend, LMBackend)  # structural check


def test_generate_returns_str(tokenizer, tiny_net):
    backend = WaaaghBackend(tiny_net, tokenizer)
    out = backend.generate(
        "<|bos|><|user|>hello<|assistant|>",
        max_new=8, temperature=0.0, top_p=0.95, stop_strings=[], seed=0,
    )
    assert isinstance(out, str)


def test_generate_truncates_at_stop_string(tokenizer, tiny_net):
    backend = WaaaghBackend(tiny_net, tokenizer)
    out = backend.generate(
        "<|bos|><|user|>hi<|assistant|>",
        max_new=16, temperature=0.0, top_p=0.95, stop_strings=["<|user|>"], seed=0,
    )
    assert "<|user|>" not in out


def test_greedy_deterministic(tokenizer, tiny_net):
    backend = WaaaghBackend(tiny_net, tokenizer)
    kw = dict(max_new=8, temperature=0.0, top_p=0.95, stop_strings=[], seed=0)
    assert backend.generate("<|bos|><|user|>x<|assistant|>", **kw) == \
           backend.generate("<|bos|><|user|>x<|assistant|>", **kw)
