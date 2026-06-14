import torch

from orkun.policy.sampler import sample, sample_batch


def test_greedy_is_deterministic(tiny_net):
    prompt = [1, 2, 3, 4]
    a = sample(tiny_net, prompt, max_new=8, temperature=0.0)
    b = sample(tiny_net, prompt, max_new=8, temperature=0.0)
    assert a == b
    assert len(a) <= 8
    assert all(isinstance(t, int) for t in a)


def test_stops_on_stop_id(tiny_net):
    prompt = [5, 6, 7]
    first = sample(tiny_net, prompt, max_new=1, temperature=0.0)[0]
    out = sample(tiny_net, prompt, max_new=8, temperature=0.0, stop_ids={first})
    assert out == []


def test_respects_max_new(tiny_net):
    out = sample(tiny_net, [1, 2], max_new=5, temperature=1.0, top_p=0.9)
    assert len(out) <= 5


def test_truncates_prompt_to_max_seq_len(tiny_net):
    long_prompt = list(range(tiny_net.cfg.max_seq_len + 20))
    out = sample(tiny_net, long_prompt, max_new=3, temperature=0.0)
    assert len(out) <= 3


def _uncached_greedy(net, prompt, max_new):
    """Reference: full-prefix recompute, greedy. The KV-cached sampler must match this."""
    ids = list(prompt)
    out = []
    for _ in range(max_new):
        x = torch.tensor([ids[-net.cfg.max_seq_len:]], dtype=torch.long)
        with torch.no_grad():
            logits = net(x)[0, -1]
        nxt = int(torch.argmax(logits).item())
        ids.append(nxt)
        out.append(nxt)
    return out


def test_cached_greedy_matches_uncached(tiny_net):
    """KV-cached decode must produce the same greedy ids as full-prefix recompute."""
    prompt = [3, 14, 15, 9, 26, 5]
    cached = sample(tiny_net, prompt, max_new=12, temperature=0.0)
    reference = _uncached_greedy(tiny_net, prompt, max_new=12)
    assert cached == reference


def test_batch_greedy_matches_single(tiny_net):
    """Batched greedy: every row must equal the single-stream greedy decode."""
    prompt = [3, 14, 15, 9, 26, 5]
    single = sample(tiny_net, prompt, max_new=10, temperature=0.0)
    batch = sample_batch(tiny_net, prompt, n=4, max_new=10, temperature=0.0)
    assert len(batch) == 4
    assert all(row == single for row in batch)


def test_batch_respects_max_new(tiny_net):
    batch = sample_batch(tiny_net, [1, 2], n=3, max_new=5, temperature=1.0, top_p=0.9)
    assert len(batch) == 3
    assert all(len(r) <= 5 for r in batch)


def test_batch_stops_on_stop_id(tiny_net):
    prompt = [5, 6, 7]
    first = sample(tiny_net, prompt, max_new=1, temperature=0.0)[0]
    batch = sample_batch(tiny_net, prompt, n=3, max_new=8, temperature=0.0, stop_ids={first})
    assert batch == [[], [], []]


def test_batch_seed_is_deterministic(tiny_net):
    a = sample_batch(tiny_net, [1, 2, 3], n=4, max_new=8, temperature=1.0, top_p=0.95, seed=123)
    b = sample_batch(tiny_net, [1, 2, 3], n=4, max_new=8, temperature=1.0, top_p=0.95, seed=123)
    assert a == b


def test_batch_sampling_diversifies(tiny_net):
    """temperature>0 with independent per-row multinomial → rows are not all identical."""
    rows = sample_batch(tiny_net, [1, 2, 3, 4], n=8, max_new=12, temperature=1.5, top_p=0.99, seed=7)
    assert any(rows[i] != rows[0] for i in range(1, len(rows)))
