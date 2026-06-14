"""Model soup — cross-arm weight averaging via a shared directory."""
import torch

from orkun.train.soup import (
    average_state_dicts,
    dump_arm_weights,
    read_peer_states,
    soup_sync,
)


def test_average_is_elementwise_mean():
    a = {"w": torch.tensor([0.0, 2.0]), "b": torch.tensor([10.0])}
    b = {"w": torch.tensor([4.0, 6.0]), "b": torch.tensor([20.0])}
    mean = average_state_dicts([a, b])
    assert torch.allclose(mean["w"], torch.tensor([2.0, 4.0]))
    assert torch.allclose(mean["b"], torch.tensor([15.0]))


def test_average_single_is_identity():
    a = {"w": torch.tensor([1.0, 2.0, 3.0])}
    mean = average_state_dicts([a])
    assert torch.allclose(mean["w"], a["w"])


def test_dump_is_atomic_no_tmp_left(tmp_path):
    net = torch.nn.Linear(3, 2)
    path = dump_arm_weights(net, tmp_path, arm_idx=0)
    assert path.is_file()
    # no leftover temp files — write was atomic
    assert not list(tmp_path.glob("*.tmp_*"))
    assert sorted(p.name for p in tmp_path.glob("arm_*.safetensors")) == ["arm_0.safetensors"]


def test_dump_skips_non_float_buffers(tmp_path):
    net = torch.nn.Linear(2, 2)
    net.register_buffer("step", torch.tensor(7, dtype=torch.long))
    dump_arm_weights(net, tmp_path, arm_idx=0)
    state = read_peer_states(tmp_path)[0]
    assert "step" not in state  # integer buffers are not souped
    assert "weight" in state and state["weight"].is_floating_point()


def test_soup_sync_pulls_arm_toward_peer_mean(tmp_path):
    torch.manual_seed(0)
    a = torch.nn.Linear(4, 4)
    b = torch.nn.Linear(4, 4)
    with torch.no_grad():
        a.weight.fill_(0.0)
        b.weight.fill_(2.0)

    # b dumps first so a can read it; then a syncs and lands on the mean.
    dump_arm_weights(b, tmp_path, arm_idx=1)
    n_mixed = soup_sync(a, tmp_path, arm_idx=0)

    assert n_mixed == 2
    assert torch.allclose(a.weight, torch.ones_like(a.weight))  # mean(0, 2) == 1


def test_soup_sync_restores_waaagh_norm_invariant(tmp_path):
    # WAAAGH invariant: the mean of unit-norm matrices is NOT unit-norm, so
    # soup_sync must call normalize_weights() after loading the averaged state.
    class NormNet(torch.nn.Linear):
        def __init__(self):
            super().__init__(4, 4)
            self.normalized = 0

        def normalize_weights(self):
            self.normalized += 1

    a, b = NormNet(), NormNet()
    with torch.no_grad():
        a.weight.fill_(0.0)
        b.weight.fill_(2.0)
    dump_arm_weights(b, tmp_path, arm_idx=1)
    n_mixed = soup_sync(a, tmp_path, arm_idx=0)
    assert n_mixed == 2
    assert a.normalized == 1          # invariant restored exactly once, after the mix


def test_soup_sync_alone_is_noop(tmp_path):
    torch.manual_seed(1)
    a = torch.nn.Linear(4, 4)
    before = a.weight.detach().clone()
    n_mixed = soup_sync(a, tmp_path, arm_idx=0)
    assert n_mixed == 1                       # only our own dump present
    assert torch.allclose(a.weight, before)   # nothing to mix → unchanged
