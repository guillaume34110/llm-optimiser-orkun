import torch

from orkun.world_model.scoring import span_logprob


def test_span_logprob_matches_manual(tiny_net):
    ids = [1, 2, 3, 4, 5, 6]
    cont_len = 2
    got = span_logprob(tiny_net, ids, cont_len)
    # manual reference: token at position p is predicted by logits[p-1]
    logits = tiny_net(torch.tensor([ids]))[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    T = len(ids)
    start = T - cont_len
    ref = sum(logp[start - 1 + j, ids[start + j]].item() for j in range(cont_len)) / cont_len
    assert abs(float(got) - ref) < 1e-4


def test_span_logprob_prefers_greedy_token(tiny_net):
    prefix = [1, 2, 3, 4, 5]
    nxt = int(torch.argmax(tiny_net(torch.tensor([prefix]))[0, -1]))  # max-logit next id
    other = (nxt + 7) % tiny_net.cfg.vocab_size
    hi = float(span_logprob(tiny_net, prefix + [nxt], 1))
    lo = float(span_logprob(tiny_net, prefix + [other], 1))
    assert hi >= lo


def test_span_logprob_is_differentiable(tiny_net):
    got = span_logprob(tiny_net, [1, 2, 3, 4], 2)
    assert got.requires_grad
