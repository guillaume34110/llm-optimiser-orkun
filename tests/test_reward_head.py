from orkun.world_model.reward_head import RewardHead
from orkun.world_model.waaagh_wm import WaaaghWorldModel


def _rows():
    # distinct wires with distinct graded targets — a learnable mapping
    return [
        {"wire": "wire alpha good", "graded": 1.0},
        {"wire": "wire bravo bad", "graded": 0.0},
        {"wire": "wire charlie best", "graded": 1.0},
        {"wire": "wire delta worst", "graded": 0.0},
    ]


def test_predict_in_unit_interval(tiny_net, tokenizer):
    head = RewardHead(WaaaghWorldModel(tiny_net, tokenizer))
    p = head.predict("any wire text")
    assert 0.0 <= p <= 1.0


def test_fit_reduces_mae(tiny_net, tokenizer):
    head = RewardHead(WaaaghWorldModel(tiny_net, tokenizer))
    rows = _rows()
    before = sum(abs(head.predict(r["wire"]) - r["graded"]) for r in rows) / len(rows)
    head.fit(rows, epochs=200)
    after = sum(abs(head.predict(r["wire"]) - r["graded"]) for r in rows) / len(rows)
    assert after < before
