from orkun.train.reward import advantage, baseline, r_total


def test_r_total_increases_with_graded_and_curiosity():
    assert r_total(0.8, 0.5, beta=0.1) > r_total(0.3, 0.5, beta=0.1)
    assert r_total(0.5, 0.9, beta=0.1) > r_total(0.5, 0.1, beta=0.1)


def test_beta_scales_intrinsic_term():
    assert r_total(0.5, 1.0, beta=0.0) == 0.5
    assert abs(r_total(0.5, 1.0, beta=0.2) - 0.7) < 1e-9


def test_baseline_is_mean_and_zero_when_empty():
    assert baseline([]) == 0.0
    assert abs(baseline([1.0, 2.0, 3.0]) - 2.0) < 1e-9


def test_advantage_is_reward_minus_baseline():
    assert advantage(1.5, 0.5) == 1.0
    assert advantage(0.2, 0.5) == -0.3
